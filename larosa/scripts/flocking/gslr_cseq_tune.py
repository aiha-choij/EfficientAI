"""GSLR-CSEQ E1: composition-aware sequential (student-stream) B1d
calibration (topic: groupwise-flocking-tuning, request 20260819-060430-
gslr-cseq-e1). Re-examines the July local-loss-refit "L2" Dead End
(sequential/deployed-stream calibration compounds errors) under the
current anchored-ridge B1D solver, which anchors toward the ORIGINAL
weights W^{d,0} (not toward zero -- the L2 verdict's critique C1 confound)
and in the group-mask setting (not per-token).

Design (request section "Part B -- E1"), alpha-damped GPTQ-style sweep:
  1. Teacher activations x_t: ONE dense forward of the calibration set,
     hooked on every target layer's mlp input simultaneously (never
     changes across layers/alpha -- collect_teacher_all_layers does this
     in a single pass, unlike gslr25_tune.py's per-layer collect_layer_
     activations, which reran the full dense forward once per layer).
  2. Process layers l=0..30 in order. Layers < l already have their FINAL
     retuned W_down installed as a deployment MaskedMLP (group-masked
     forward, reused verbatim from gslr_group_ppl.py) -- collect_student_
     at_layer runs a forward pass through this partially-deployed model,
     stopping (via a StopForward exception raised inside the capturing
     pre-hook) right before layer l's mlp, to get the student-stream FFN
     input x~_t without wasting compute on layers > l (O(L) total forward
     cost across the sweep, not O(L^2)).
  3. x_hat_t = (1-alpha)*x_t + alpha*x~_t. alpha=0 -> x_hat == x_t exactly,
     collapsing every downstream step to B1D's own computation bit-for-bit
     (the mandatory sanity arm).
  4. On x_hat: intermediate i_hat = gate(x_hat)*up(x_hat) (ORIGINAL Wu0/Wg0
     -- this algorithm only ever retunes W_down, matching B1D's Delta
     w_up=Delta w_gate=0 design). Score/mask normally come from i_hat too
     (mask_source="student", default); the mask-ablation arm
     (mask_source="teacher") builds the mask from i_teacher instead while
     STILL using i_hat for the regressor z_t -- isolates input-shift
     (regressor distribution) from mask-shift (which neurons get selected)
     contributions to any alpha effect.
  5. Anchored ridge target y*_t = W^{d,0} @ i_teacher_t (the TEACHER
     output at the TEACHER input -- independent of alpha, computed once
     per layer) -- the same target B1D itself uses. Solved with the
     IDENTICAL Lambda-weighted anchored ridge as B1D (refit_wdown_weighted,
     same mu/Lambda-weighting), so alpha=0 is not just "close to" B1D, it
     is the same closed-form system.
  6. Install the retuned layer as a deployment MaskedMLP in the live model
     (mutates model.model.layers[l].mlp) so later layers' student-stream
     collection sees it -- "continue the student forward through the
     refit layer" (request step 6).

RAMS combination (--rams_reweight): a static per-neuron score reweight
w_j = (unique_j/median_j unique_j)^rams_alpha is multiplied into the group
score before top-K, on top of whatever alpha/mask_source arm is running.
unique_j and its Cholesky-Schur-complement computation are ported verbatim
from larosa/scripts/flocking/gslr_rams_e2.py (topic refit-aware-mask-
selection, branch auto/rams-e1, commit dee0414) -- see compute_unique's
docstring for the math. Ported rather than imported because that script
lives on a different topic's branch/worktree (auto/rams-e1) that this
topic must not depend on or disturb.

Output format is IDENTICAL to gslr25_tune.py's B1D layer_i.pt ({wg, wu,
wd} with wg==wg0, wu==wu0 always -- only wd is ever retuned), so
gslr_group_ppl.py evaluates E1 output with ZERO code changes.
"""

import argparse
import json
import math
import os
import time

import torch

import gslr_layer_tune as glt
import gslr_group_ppl as ggp
import gslr25_lambda as g25l


class StopForward(Exception):
    pass


# ---------------------------------------------------------------- data

def token_stream(tok, split, n_seqs, seqlen):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n_seqs * seqlen
    assert ids.numel() >= need, f"{split}: {ids.numel()} < {need} tokens"
    return ids[:need].view(n_seqs, seqlen)


def collect_teacher_all_layers(model, ids, layers, device):
    """One dense forward pass per calibration sequence, hooked on every
    target layer's mlp input simultaneously. Returns {layer_idx: (N, h)
    fp16 cpu tensor}."""
    captured = {li: [] for li in layers}
    hooks = []
    for li in layers:
        def mk(li):
            def hook(mod, inp):
                captured[li].append(inp[0].detach().squeeze(0).to(torch.float16).cpu())
            return hook
        hooks.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i:i + 1].to(device))
    for h in hooks:
        h.remove()
    return {li: torch.cat(captured[li], 0) for li in layers}


def collect_student_at_layer(model, layer_idx, ids, device):
    """Forward pass through whatever is CURRENTLY installed in the live
    model (layers < layer_idx already swapped to deployment MaskedMLP by
    earlier E1 iterations; layer_idx itself and beyond still original),
    stopping right before layer_idx's mlp via StopForward -- avoids running
    layers > layer_idx this iteration."""
    captured = []

    def hook(mod, inp):
        captured.append(inp[0].detach().squeeze(0).float().cpu())
        raise StopForward()

    h = model.model.layers[layer_idx].mlp.register_forward_pre_hook(hook)
    out = []
    try:
        with torch.no_grad():
            for i in range(ids.shape[0]):
                captured.clear()
                try:
                    model(ids[i:i + 1].to(device))
                except StopForward:
                    pass
                out.append(captured[0])
    finally:
        h.remove()
    return torch.cat(out, 0)


# ---------------------------------------------------------------- RAMS reweight (ported from gslr_rams_e2.py)

def compute_unique(I, colnorm0, eps=1e-3, chunk=8192):
    """Ported verbatim (math unchanged) from gslr_rams_e2.py's
    compute_unique (topic refit-aware-mask-selection, auto/rams-e1, commit
    dee0414): unique_j = residual variance of neuron j's gauge-scaled
    activation xi_j = i_.,j * colnorm0_j after linearly regressing on
    every other neuron's xi (Schur-complement identity on the regularized
    precision matrix of the calibration Gram Ghat = sum_t xi_t xi_t^T)."""
    N, d = I.shape
    V = I * colnorm0[None, :]
    Ghat = glt.chunked_matmul(V, V, chunk)
    A = Ghat / N
    meandiag = A.diagonal().mean()
    A = A + eps * meandiag * torch.eye(d, device=A.device, dtype=A.dtype)
    L = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(L)
    unique = 1.0 / Ainv.diagonal().clamp(min=1e-12)
    return unique


def rams_weight(unique, exponent=0.25):
    """w_j = (unique_j / median_j unique_j) ^ exponent. exponent=0 -> w==1
    identically (no-op, boundary sanity)."""
    med = unique.median().clamp(min=1e-12)
    return (unique / med).clamp(min=1e-12).pow(exponent)


def compute_group_mask_weighted(I, colnorm0, K, g, weight=None):
    """Same score rule as glt.compute_group_mask, with an optional static
    multiplicative per-neuron weight on the group score before top-K.
    weight=None is an EXACT no-op, identical to glt.compute_group_mask
    (verified in selftest)."""
    N, d = I.shape
    score = I.abs() * colnorm0[None, :]
    gscore = score.view(-1, g, d).sum(1)
    if weight is not None:
        gscore = gscore * weight[None, :]
    idx = gscore.topk(K, dim=-1).indices
    group_mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
    token_mask = group_mask.repeat_interleave(g, 0)
    return group_mask, token_mask


# ---------------------------------------------------------------- per-layer driver

def process_layer(model, layer_idx, X_teacher_l, train_ids, g, K, args, out_dir):
    outpath = os.path.join(out_dir, f"layer_{layer_idx}.pt")
    metapath = os.path.join(out_dir, f"layer_{layer_idx}.meta.json")
    dev = args.device
    layer = model.model.layers[layer_idx]
    mlp0 = layer.mlp
    Wg0 = mlp0.gate_proj.weight.detach().float()
    Wu0 = mlp0.up_proj.weight.detach().float()
    Wd0 = mlp0.down_proj.weight.detach().float()
    act_name = model.config.hidden_act
    f = glt.act_fn(act_name)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()

    if os.path.exists(outpath) and os.path.exists(metapath):
        glt.log(f"layer {layer_idx}: output exists, skipping (resume)")
        sd = torch.load(outpath, map_location=dev)
        layer.mlp = ggp.MaskedMLP(mlp0, Wg0, Wu0, sd["wd"].to(dev), colnorm0, g, K, apply_mask=True)
        return

    X_teacher = X_teacher_l.to(dev).float()

    Lambda = g25l.compute_layer_lambda(model, layer_idx).to(dev)

    if args.alpha > 0:
        X_student = collect_student_at_layer(model, layer_idx, train_ids, dev).float()
        X_hat = (1.0 - args.alpha) * X_teacher + args.alpha * X_student
    else:
        X_student = X_teacher
        X_hat = X_teacher

    I_teacher = glt.intermediate(X_teacher, f(glt.linear_up(X_teacher, Wg0)), Wu0)
    Y_teacher = glt.chunked_mm(I_teacher, Wd0.T)

    I_hat = I_teacher if args.alpha == 0 else glt.intermediate(X_hat, f(glt.linear_up(X_hat, Wg0)), Wu0)

    mask_input = I_teacher if args.mask_source == "teacher" else I_hat

    weight = None
    unique_stats = None
    if args.rams_reweight:
        unique = compute_unique(I_teacher, colnorm0, eps=args.rams_eps)
        weight = rams_weight(unique, exponent=args.rams_alpha)
        unique_stats = {"min": unique.min().item(), "max": unique.max().item(),
                         "median": unique.median().item(), "mean": unique.mean().item()}

    group_mask, token_mask = compute_group_mask_weighted(mask_input, colnorm0, K, g, weight)
    z = token_mask * I_hat

    Wd = glt.refit_wdown_weighted(z, Y_teacher, Wd0, Lambda, args.lambda_down,
                                   args.cg_iters, args.cg_tol)

    # diagnostics
    ref_group_mask, _ = compute_group_mask_weighted(I_teacher, colnorm0, K, g, weight=None)
    flip_rate = (group_mask != ref_group_mask).float().mean().item()
    recon = glt.chunked_mm(z, Wd.T)
    ynorm = Y_teacher.pow(2).sum().sqrt().clamp(min=1e-12)
    relerr = ((recon - Y_teacher).pow(2).sum().sqrt() / ynorm).item()
    wdown_drift = ((Wd - Wd0).norm() / Wd0.norm().clamp(min=1e-12)).item()

    torch.save({"wg": Wg0.cpu(), "wu": Wu0.cpu(), "wd": Wd.cpu()}, outpath)
    meta = {
        "layer": layer_idx, "g": g, "K": K, "d": Wu0.shape[0], "sparsity": args.sparsity,
        "alpha": args.alpha, "mask_source": args.mask_source,
        "rams_reweight": bool(args.rams_reweight), "rams_alpha": args.rams_alpha if args.rams_reweight else None,
        "rams_eps": args.rams_eps if args.rams_reweight else None, "unique_stats": unique_stats,
        "lambda_down": args.lambda_down, "cg_iters": args.cg_iters, "cg_tol": args.cg_tol,
        "git_commit": glt.git_hash(), "seed": args.seed, "model": args.model,
        "train_seqs": args.train_seqs, "seqlen": args.seqlen,
        "lambda_lookahead_stats": {"mean": Lambda.mean().item(), "min": Lambda.min().item(),
                                    "max": Lambda.max().item()},
        "wdown_drift": wdown_drift,
        "masked_recon_relerr_train": relerr,
        "teacher_vs_student_mask_flip_rate": flip_rate,
    }
    with open(metapath, "w") as fh:
        json.dump(meta, fh, indent=2)
    glt.log(f"layer {layer_idx} alpha={args.alpha} g={g} mask_source={args.mask_source} "
            f"rams={args.rams_reweight}: wdown_drift={wdown_drift:.4f} "
            f"masked_recon_relerr(train)={relerr:.4f} flip_rate={flip_rate:.4f}")

    layer.mlp = ggp.MaskedMLP(mlp0, Wg0, Wu0, Wd, colnorm0, g, K, apply_mask=True)

    del X_teacher, X_student, X_hat, I_teacher, I_hat, Y_teacher, z, recon, group_mask, token_mask
    glt.empty_cache()


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)

    # ---- test 1: alpha=0 sanity -- must reproduce B1D's own computation
    # (glt.compute_group_mask + refit_wdown_weighted) bit-for-bit.
    N, h, d, g = 512, 12, 32, 8
    X = torch.randn(N, h)
    Wg0 = torch.randn(d, h) / math.sqrt(h)
    Wu0 = torch.randn(d, h) / math.sqrt(h)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    fsilu = glt.act_fn("silu")
    I0 = glt.intermediate(X, fsilu(X @ Wg0.T), Wu0)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    K = d // 2
    Y = I0 @ Wd0.T + 0.02 * torch.randn(N, h)
    Lambda = 1.0 + torch.rand(h)
    lam_down = 0.7

    _, tm_b1d = glt.compute_group_mask(I0, colnorm0, K, g)
    Wd_b1d = glt.refit_wdown_weighted(I0 * tm_b1d, Y, Wd0, Lambda, lam_down, cg_iters=200, cg_tol=1e-10)

    # E1 alpha=0 path (mirrors process_layer with args.alpha==0): X_hat ==
    # X_teacher identically -> I_hat == I_teacher regardless of mask_source,
    # weight=None (no rams).
    I_teacher = I0
    Y_teacher = Y
    mask_input = I_teacher
    gmask_e1, tmask_e1 = compute_group_mask_weighted(mask_input, colnorm0, K, g, weight=None)
    z = tmask_e1 * I_teacher
    Wd_e1 = glt.refit_wdown_weighted(z, Y_teacher, Wd0, Lambda, lam_down, cg_iters=200, cg_tol=1e-10)

    relerr = (Wd_b1d - Wd_e1).norm() / Wd_b1d.norm().clamp(min=1e-12)
    assert relerr < 1e-6, f"alpha=0 must reproduce B1D bit-for-bit: relerr={relerr:.2e}"
    print(f"selftest 1/6 OK (alpha=0 reproduces B1D bit-for-bit, relerr={relerr:.2e})")

    # ---- test 2: weight=None boundary == glt.compute_group_mask exactly
    gmask_ref, tmask_ref = glt.compute_group_mask(I0, colnorm0, K, g)
    assert torch.equal(gmask_e1, gmask_ref) and torch.equal(tmask_e1, tmask_ref)
    print("selftest 2/6 OK (compute_group_mask_weighted(weight=None) == compute_group_mask exactly)")

    # ---- test 3: weight=ones also reduces to the unweighted mask
    gmask_w1, _ = compute_group_mask_weighted(I0, colnorm0, K, g, weight=torch.ones(d))
    assert torch.equal(gmask_w1, gmask_ref)
    print("selftest 3/6 OK (weight=ones reduces to weight=None exactly)")

    # ---- test 4: a dominant weight DOES change the mask (wiring sanity)
    strong_weight = torch.ones(d)
    strong_weight[0] = 1e6
    gmask_strong, _ = compute_group_mask_weighted(I0, colnorm0, K, g, weight=strong_weight)
    assert not torch.equal(gmask_strong, gmask_ref), "a dominant weight should change the mask"
    assert gmask_strong[:, 0].all(), "dominant-weight neuron must be selected in every group"
    print("selftest 4/6 OK (nonzero weight changes selection; dominant neuron forced into every group)")

    # ---- test 5: RAMS unique_j / weight sanity (ported math)
    torch.manual_seed(7)
    Nu, du = 4000, 6
    base = torch.randn(Nu, 1)
    Iu = torch.cat([base + 0.001 * torch.randn(Nu, 1),
                    base + 0.001 * torch.randn(Nu, 1),
                    torch.randn(Nu, du - 2)], dim=1)
    colnorm_u = torch.ones(du)
    unique = compute_unique(Iu, colnorm_u, eps=1e-3)
    assert unique[0] < 0.3 * unique[2:].mean(), "redundant duplicated pair should have small unique_j"
    w0 = rams_weight(unique, exponent=0.0)
    assert torch.allclose(w0, torch.ones(du), atol=1e-5), "exponent=0 must give weight==1 exactly"
    print(f"selftest 5/6 OK (RAMS unique_j: dup-pair unique={unique[0]:.4f} vs indep mean="
          f"{unique[2:].mean():.4f}; exponent=0 -> weight==1)")

    # ---- test 6: StopForward pre-hook actually skips downstream compute
    class Marker(torch.nn.Module):
        def __init__(self, tag, ran):
            super().__init__()
            self.tag = tag
            self.ran = ran

        def forward(self, x):
            self.ran.append(self.tag)
            return x + 1

    class Chain(torch.nn.Module):
        def __init__(self, mods):
            super().__init__()
            self.mods = torch.nn.ModuleList(mods)

        def forward(self, x):
            for m in self.mods:
                x = m(x)
            return x

    ran = []
    chain = Chain([Marker("a", ran), Marker("b", ran), Marker("c", ran)])
    captured = []

    def hook(mod, inp):
        captured.append(inp[0].detach().clone())
        raise StopForward()

    hh = chain.mods[1].register_forward_pre_hook(hook)
    x0 = torch.zeros(1, 1)
    try:
        chain(x0)
    except StopForward:
        pass
    hh.remove()
    assert ran == ["a"], f"module b/c should never run after StopForward at b's pre-hook: ran={ran}"
    assert torch.allclose(captured[0], torch.ones(1, 1)), "captured input to b should be a's output"
    print("selftest 6/6 OK (StopForward pre-hook: captures the right tensor, downstream modules never run)")

    print("selftest ALL OK")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--layers", default="0-30", help="inclusive range 'lo-hi'")
    ap.add_argument("--g", type=int, required=False)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--train_seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.0,
                     help="damping: x_hat = (1-alpha)*x_teacher + alpha*x_student; "
                          "alpha=0 is the mandatory B1D-reproduction sanity arm")
    ap.add_argument("--mask_source", choices=["student", "teacher"], default="student",
                     help="'teacher' = mask ablation arm (masks always from teacher x, "
                          "regressor z_t still from x_hat)")
    ap.add_argument("--lambda_down", type=float, default=1.0, help="B1D's anchor strength (b1d default)")
    ap.add_argument("--cg_iters", type=int, default=25)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--rams_reweight", action="store_true",
                     help="apply the RAMS E2 static decorrelation reweight w_j=(unique_j/median)^rams_alpha")
    ap.add_argument("--rams_alpha", type=float, default=0.25)
    ap.add_argument("--rams_eps", type=float, default=1e-3)
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
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    lo, hi = (int(v) for v in args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    train_ids = token_stream(tok, "train", args.train_seqs, args.seqlen)
    torch.save({"train_ids": train_ids}, os.path.join(args.out, "calib_tokens.pt"))

    glt.log(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.to(args.device).eval()

    d = model.model.layers[0].mlp.down_proj.weight.shape[1]
    K = int(math.floor((1 - args.sparsity) * d))
    glt.log(f"d={d} K={K} layers={layer_ids}")

    glt.log("caching teacher activations (single dense pass, all target layers)...")
    t0 = time.time()
    X_teacher = collect_teacher_all_layers(model, train_ids, layer_ids, args.device)
    glt.log(f"teacher cache done ({len(layer_ids)} layers, {time.time() - t0:.1f}s).")

    for li in layer_ids:
        t0 = time.time()
        process_layer(model, li, X_teacher[li], train_ids, args.g, K, args, args.out)
        glt.log(f"=== layer {li} done in {time.time() - t0:.1f}s ===")

    glt.log("all layers done.")


if __name__ == "__main__":
    main()
