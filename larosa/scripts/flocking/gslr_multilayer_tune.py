"""GSLR stage-2, step 1: multi-layer retuning (topic: groupwise-flocking-tuning,
follow-up to gslr-stage1 / e98d734).

Stage 1 established (single layer 16 only): arm A2 (design B masked recon +
gauge-invariant l2,1 + W_gate GN) passes the Go threshold (>=1.5x within-group
overlap vs untuned, held-out maskrecon non-worsening) at g=64/lambda_rel=1,
and is a partial/mixed pass at g=16 (1.305x, below 1.5x). Stage 2's question
is whether retuning EVERY layer with this recipe actually improves wikitext-2
PPL under group masking versus the untuned model -- this script produces the
retuned weights; gslr_group_ppl.py (separate script) measures PPL.

Design constraint from the stage-2 request (must hold, not just a style
choice): retuning layer i's calibration activations must come from the
ORIGINAL DENSE model, never from a previously-retuned layer's output.
Sequential retuned-to-retuned propagation was stage-1's precursor's ("L2")
dead end (error accumulation) -- this script never runs the live model
through anything but its original frozen weights. Concretely: we never
write retuned weights back into the live `model` object; Wg/Wu/Wd for layer
i are extracted once per layer as plain tensors, retuned out-of-place with
the functions in gslr_layer_tune.py (imported, not duplicated), and the live
model (always original weights) is reused only to regenerate X via forward
hooks for the NEXT layer.

One (g, lambda_rel) config per invocation (arm fixed to A2, the only
Go-passing arm from stage 1). Layers 0..30 by default -- layer 31 is
excluded per spec (kept at original weights; stage-1 spec's own note: its
activations are already outlier-concentrated, Gini 0.705, measured
elsewhere). For lambda_rel=1.0 (stage-1's grid max, known to be the least
stable point), if a layer's held-out group-mask-recon error comes out worse
than that SAME layer's own A0 baseline (the anchor-ridge non-worsening
invariant stage-1 treated as a hard requirement) or produces non-finite
weights, that single layer is redone at lambda_rel=0.3 and the fallback is
recorded in its meta -- never silently kept.

GPU memory: the live model (bf16, ~13GB) and the retuning math (fp32,
several GB of N x d buffers per stage-1's OOM postmortem) are never resident
together -- the model is moved to CPU before each layer's retuning step and
back to GPU before the next layer's activation collection. This mirrors
stage-1's collect_activations(), which explicitly deleted the model before
retuning for the same reason (see e98d734's commit message).

Resumable: a layer whose <out>/layer_{i}.pt AND layer_{i}.meta.json already
exist is skipped (so a killed/OOM job can be resubmitted with the same --out
and only the missing layers re-run).
"""

import argparse
import json
import math
import os
import time

import torch

import gslr_layer_tune as glt


def collect_layer_activations(model, layer_idx, train_ids, test_ids, device):
    """Full frozen-model forward pass, hook on layer_idx's mlp input only.
    Verbatim logic of gslr_layer_tune.collect_activations, parameterized by
    layer and reusing an already-loaded model (never mutated)."""
    layer = model.model.layers[layer_idx]
    Wg0 = layer.mlp.gate_proj.weight.detach().float().cpu()
    Wu0 = layer.mlp.up_proj.weight.detach().float().cpu()
    Wd0 = layer.mlp.down_proj.weight.detach().float().cpu()
    act_name = model.config.hidden_act

    captured = []
    hook = layer.mlp.register_forward_pre_hook(
        lambda mod, inp: captured.append(inp[0].detach().squeeze(0).to(torch.float16).cpu()))

    def run(ids):
        captured.clear()
        with torch.no_grad():
            for i in range(ids.shape[0]):
                model(ids[i:i + 1].to(device))
        return torch.cat(captured, 0)

    X_train = run(train_ids)
    X_test = run(test_ids)
    hook.remove()
    return X_train, X_test, Wg0, Wu0, Wd0, act_name


def is_unstable(m_a2, m_a0, g, Wu, Wg_, Wd):
    finite = all(torch.isfinite(t).all().item() for t in (Wu, Wg_, Wd))
    if not finite:
        return True
    return m_a2[f"group_mask_recon_relerr_g{g}"] > m_a0[f"group_mask_recon_relerr_g{g}"]


def process_layer(model, layer_idx, train_ids, test_ids, g, lam, args, out_dir):
    outpath = os.path.join(out_dir, f"layer_{layer_idx}.pt")
    metapath = os.path.join(out_dir, f"layer_{layer_idx}.meta.json")
    if os.path.exists(outpath) and os.path.exists(metapath):
        glt.log(f"layer {layer_idx}: output exists, skipping (resume)")
        return

    dev = args.device
    model.to(dev)
    X_train, X_test, Wg0, Wu0, Wd0, act_name = collect_layer_activations(
        model, layer_idx, train_ids, test_ids, dev)
    # retuning math never needs the live model resident -- move it off-GPU
    # before allocating the N x d fp32 buffers (stage-1 OOM postmortem).
    model.to("cpu")
    glt.empty_cache()

    X_train = X_train.to(dev).float()
    X_test = X_test.to(dev).float()
    Wg, Wu0d, Wd0d = Wg0.to(dev), Wu0.to(dev), Wd0.to(dev)
    f = glt.act_fn(act_name)
    colnorm0 = Wd0d.pow(2).sum(0).sqrt()

    I0_train = glt.intermediate(X_train, f(glt.linear_up(X_train, Wg)), Wu0d)
    I0_test = glt.intermediate(X_test, f(glt.linear_up(X_test, Wg)), Wu0d)
    Y_train = glt.chunked_mm(I0_train, Wd0d.T)
    Y_test = glt.chunked_mm(I0_test, Wd0d.T)
    del I0_train, I0_test
    d = Wu0d.shape[0]
    K = int(math.floor((1 - args.sparsity) * d))

    m_a0 = glt.run_arm("A0", X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test,
                        K, g, 0.0, args, args.train_seqs)

    lam_used = lam
    m_a2, Wu, Wg_, Wd = glt.run_arm(
        "A2", X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test,
        K, g, lam, args, args.train_seqs, return_weights=True)

    fallback = False
    if lam == 1.0 and is_unstable(m_a2, m_a0, g, Wu, Wg_, Wd):
        fallback = True
        glt.log(f"layer {layer_idx}: lam=1.0 unstable "
                f"(maskrecon {m_a2[f'group_mask_recon_relerr_g{g}']:.4f} vs "
                f"A0 {m_a0[f'group_mask_recon_relerr_g{g}']:.4f}, or non-finite) "
                f"-> retrying at lam=0.3")
        lam_used = 0.3
        m_a2, Wu, Wg_, Wd = glt.run_arm(
            "A2", X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test,
            K, g, lam_used, args, args.train_seqs, return_weights=True)

    torch.save({"wg": Wg_.cpu(), "wu": Wu.cpu(), "wd": Wd.cpu()}, outpath)

    overlap_ratio = (m_a2[f"within_group_overlap_g{g}"] /
                      max(m_a0[f"within_group_overlap_g{g}"], 1e-12))
    meta = {
        "layer": layer_idx, "g": g, "lambda_rel_requested": lam, "lambda_rel_used": lam_used,
        "fallback_triggered": fallback, "arm": "A2", "K": K, "d": d,
        "sparsity": args.sparsity, "git_commit": glt.git_hash(), "seed": args.seed,
        "model": args.model, "train_seqs": args.train_seqs, "test_seqs": args.test_seqs,
        "seqlen": args.seqlen, "mu_up": args.mu_up, "mu_gate": args.mu_gate,
        "lambda_down": args.lambda_down, "outer": args.outer, "irls": args.irls,
        "cg_iters": args.cg_iters, "cg_tol": args.cg_tol,
        "overlap_ratio_vs_a0": overlap_ratio,
        "metrics_a0": m_a0, "metrics_a2": m_a2,
    }
    with open(metapath, "w") as fh:
        json.dump(meta, fh, indent=2)

    glt.log(
        f"layer {layer_idx} g={g} lam_used={lam_used}{' (fallback)' if fallback else ''}: "
        f"overlap={m_a2[f'within_group_overlap_g{g}']:.4f} "
        f"(A0={m_a0[f'within_group_overlap_g{g}']:.4f}, x{overlap_ratio:.3f}) "
        f"maskrecon={m_a2[f'group_mask_recon_relerr_g{g}']:.4f} "
        f"(A0={m_a0[f'group_mask_recon_relerr_g{g}']:.4f}) "
        f"static_frac={m_a2[f'static_mask_frac_g{g}']:.3f} "
        f"cross/within={m_a2[f'cross_within_ratio_g{g}']:.3f} "
        f"drift(u/g/d)={m_a2['wup_drift']:.4f}/{m_a2['wgate_drift']:.4f}/{m_a2['wdown_drift']:.4f}")

    del X_train, X_test, Y_train, Y_test, Wu, Wg_, Wd, Wg, Wu0d, Wd0d, colnorm0
    glt.empty_cache()


def selftest():
    """CPU-only: checks process_layer's resume skip and the fallback trigger
    logic against synthetic tensors (no GPU, no real model needed for the
    fallback predicate itself)."""
    torch.manual_seed(0)

    # is_unstable: worse maskrecon -> True
    m_a2_bad = {"group_mask_recon_relerr_g8": 0.9}
    m_a0 = {"group_mask_recon_relerr_g8": 0.7}
    finite = (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4))
    assert is_unstable(m_a2_bad, m_a0, 8, *finite) is True
    m_a2_ok = {"group_mask_recon_relerr_g8": 0.65}
    assert is_unstable(m_a2_ok, m_a0, 8, *finite) is False
    nonfinite = (torch.tensor([float("nan")]), torch.randn(4, 4), torch.randn(4, 4))
    assert is_unstable(m_a2_ok, m_a0, 8, *nonfinite) is True
    print("selftest 1/2 OK (is_unstable: worse-maskrecon and non-finite both trip fallback)")

    # resume: process_layer must no-op (not raise) when outputs already exist
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "layer_3.pt"), "w").close()
        open(os.path.join(td, "layer_3.meta.json"), "w").close()

        class Boom:
            def to(self, *a, **k):
                raise AssertionError("process_layer should have returned before touching the model")
        process_layer(Boom(), 3, None, None, 8, 1.0, None, td)
    print("selftest 2/2 OK (resume: existing layer_i outputs skip re-computation)")

    print("selftest ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--layers", default="0-30", help="inclusive range 'lo-hi'")
    ap.add_argument("--g", type=int, required=False)
    ap.add_argument("--lambda_rel", type=float, required=False)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--train_seqs", type=int, default=32)
    ap.add_argument("--test_seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--outer", type=int, default=2)
    ap.add_argument("--irls", type=int, default=3)
    ap.add_argument("--cg_iters", type=int, default=25)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    # same anchor-strength rationale/history as gslr_layer_tune.py (stage-1
    # report section 2): PoC defaults (0.03/0.03/0.01) overfit the held-out
    # mask pattern; these are the stage-1 fix, CLI-overridable.
    ap.add_argument("--mu_up", type=float, default=3.0)
    ap.add_argument("--mu_gate", type=float, default=3.0)
    ap.add_argument("--lambda_down", type=float, default=1.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    assert args.out, "--out is required"
    assert args.g is not None, "--g is required"
    assert args.lambda_rel is not None, "--lambda_rel is required"
    torch.manual_seed(args.seed)
    args.group_sizes = str(args.g)  # run_arm/metrics read this field
    os.makedirs(args.out, exist_ok=True)

    lo, hi = (int(v) for v in args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def token_stream(split, n_seqs):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
        need = n_seqs * args.seqlen
        assert ids.numel() >= need, f"{split}: {ids.numel()} < {need} tokens"
        return ids[:need].view(n_seqs, args.seqlen)

    train_ids = token_stream("train", args.train_seqs)
    test_ids = token_stream("test", args.test_seqs)
    torch.save({"train_ids": train_ids, "test_ids": test_ids},
               os.path.join(args.out, "calib_tokens.pt"))
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    glt.log(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.eval()

    for li in layer_ids:
        t0 = time.time()
        process_layer(model, li, train_ids, test_ids, args.g, args.lambda_rel, args, args.out)
        glt.log(f"=== layer {li} done in {time.time() - t0:.1f}s ===")

    glt.log("all layers done.")


if __name__ == "__main__":
    main()
