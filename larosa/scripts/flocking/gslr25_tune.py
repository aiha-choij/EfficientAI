"""GSLR stage-2.5 multi-layer retuning driver (topic: groupwise-flocking-
tuning, follow-up to gslr-stage2 / 5da111a).

Design principle (stage-2.5 request): put the group-mask STRUCTURE entirely
in the (Delta w = 0) mask rule, keep weight changes at refit-level drift.
Two arms per layer, both independent (layer 0..30, original dense
activations only -- never sequential propagation, same constraint as
gslr_multilayer_tune.py):

  B1D -- W_down ONLY, anchored ridge (gslr_layer_tune.run_arm arm="B1D"),
         Lambda-weighted. Wu/Wg stay exactly at their original values
         (Delta wu = Delta wg = 0). This arm's measured D_ell (synthesis
         safety budget, gslr_layer_tune.measure_D) becomes epsilon*_ell,
         the budget every other arm at this layer must respect.
  B1  -- all 3 matrices (arm="A2" with lambda_rel forced to 0 -- no l2,1
         penalty, group structure comes only from the mask rule), Lambda-
         weighted, strong anchor (--mu_up/--mu_gate/--lambda_down, default
         same value as stage-2's "v2" fix: 3.0/3.0/1.0). After fitting, D_ell
         is measured and compared to epsilon*_ell; if it exceeds budget, the
         anchor is escalated (x--anchor_escalation, default 3x) and refit,
         up to --max_budget_retries times (request section 1: "예산 초과 시
         앵커를 키워 재해").

Lambda_ell is computed once per layer from gslr25_lambda.py (depends only on
layer ell+1's ORIGINAL weights, not on g or arm) and cached to
<out>/lambda/layer_{i}.pt so B1D/B1/future-B2/B3 runs at different g all
reuse it without recomputing.

Resumable per layer (both B1D and B1 outputs must exist to skip, matching
gslr_multilayer_tune.py's convention) and per matrix (Wu/Wg untouched by
policy design, not by resume-detection -- B1D genuinely never touches them).
Same GPU-memory discipline as gslr_multilayer_tune.py: live model moved to
CPU before the fp32 N x d retuning buffers are allocated.
"""

import argparse
import copy
import json
import math
import os
import time

import torch

import gslr_layer_tune as glt
import gslr_multilayer_tune as gmt
import gslr25_lambda as g25l


def should_retry(D_ell, epsilon_star, attempt, max_retries):
    """Pure predicate (CPU-selftestable): True iff the budget is exceeded
    AND another attempt is allowed. epsilon_star<=0 (degenerate/未측정) never
    blocks -- treated as "no budget info", not "zero tolerance"."""
    if epsilon_star is None or epsilon_star <= 0:
        return False
    return D_ell > epsilon_star and attempt < max_retries


def make_b1_args(base_args, mu_up, mu_gate, lambda_down):
    a = copy.copy(base_args)
    a.mu_up = mu_up
    a.mu_gate = mu_gate
    a.lambda_down = lambda_down
    a.outer = base_args.b1_outer
    a.irls = base_args.irls
    return a


def fit_with_budget_retry(X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test,
                           K, g, args, Lambda, eps_star, I0_test, mask_a0_test,
                           mu_up0, mu_gate0, lambda_down0, mask_bonus=None, log_prefix=""):
    """Shared B1/B2 fitting loop: arm A2 (lambda_rel=0, Lambda-weighted),
    optionally mask_bonus-adjusted (B2 only -- None for B1), with the
    stage-2.5 safety-budget retry (escalate the anchor while D_ell exceeds
    eps_star and attempts remain -- gslr_layer_tune.measure_D /
    should_retry). Returns (m, Wu, Wg_, Wd, D, attempt, mu_up_final,
    mu_gate_final, lambda_down_final, budget_ok)."""
    mu_up, mu_gate, lambda_down = mu_up0, mu_gate0, lambda_down0
    attempt = 0
    while True:
        attempt += 1
        fit_args = make_b1_args(args, mu_up, mu_gate, lambda_down)
        fit_args.group_sizes = str(g)
        m, Wu, Wg_, Wd = glt.run_arm(
            "A2", X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test,
            K, g, 0.0, fit_args, args.train_seqs, return_weights=True, Lambda=Lambda,
            mask_bonus=mask_bonus)
        I_test = glt.intermediate(X_test, f(glt.linear_up(X_test, Wg_)), Wu)
        _, mask_test = glt.compute_group_mask(I_test, colnorm0, K, g, bonus=mask_bonus)
        D = glt.measure_D(I_test, mask_test, Wd, I0_test, mask_a0_test, Wd0d, Lambda)
        over_budget = should_retry(D, eps_star, attempt, args.max_budget_retries)
        glt.log(f"{log_prefix} attempt={attempt}: D_ell={D:.4f} budget(eps*)={eps_star:.4f} "
                f"mu_up={mu_up:.3f} mu_gate={mu_gate:.3f} lambda_down={lambda_down:.3f}"
                f"{' -> escalating anchor' if over_budget else ''}")
        if not over_budget:
            break
        mu_up *= args.anchor_escalation
        mu_gate *= args.anchor_escalation
        lambda_down *= args.anchor_escalation
        # explicit cleanup between attempts -- see gslr25-tune-b1d-b1-g16's
        # layer-0 OOM postmortem (commit 9340b16): back-to-back A2 (GN
        # W_gate step) calls in one process hit caching-allocator
        # fragmentation in practice even though refcounting alone should
        # free the previous attempt's tensors.
        del m, Wu, Wg_, Wd, I_test, mask_test
        glt.empty_cache()
    budget_ok = not should_retry(D, eps_star, attempt, args.max_budget_retries + 1)
    return m, Wu, Wg_, Wd, D, attempt, mu_up, mu_gate, lambda_down, budget_ok


def process_layer(model, layer_idx, train_ids, test_ids, g, args, out_root):
    b1d_dir = os.path.join(out_root, "b1d", str(g))
    b1_dir = os.path.join(out_root, "b1", str(g))
    lam_dir = os.path.join(out_root, "lambda")
    for d in (b1d_dir, b1_dir, lam_dir):
        os.makedirs(d, exist_ok=True)

    b1d_out = os.path.join(b1d_dir, f"layer_{layer_idx}.pt")
    b1d_meta = os.path.join(b1d_dir, f"layer_{layer_idx}.meta.json")
    b1_out = os.path.join(b1_dir, f"layer_{layer_idx}.pt")
    b1_meta = os.path.join(b1_dir, f"layer_{layer_idx}.meta.json")
    b1d_b1_done = all(os.path.exists(p) for p in (b1d_out, b1d_meta, b1_out, b1_meta))
    betas_done = all(
        os.path.exists(os.path.join(out_root, "b2", str(g), str(beta), f"layer_{layer_idx}.pt"))
        and os.path.exists(os.path.join(out_root, "b2", str(g), str(beta), f"layer_{layer_idx}.meta.json"))
        for beta in args.betas) if args.betas else True
    if b1d_b1_done and betas_done:
        glt.log(f"layer {layer_idx}: b1d+b1{'+ b2 betas' if args.betas else ''} outputs exist, skipping (resume)")
        return

    dev = args.device
    lam_path = os.path.join(lam_dir, f"layer_{layer_idx}.pt")
    if os.path.exists(lam_path):
        Lambda = torch.load(lam_path, map_location="cpu")["lambda"].to(dev)
    else:
        model.to(dev)
        Lambda = g25l.compute_layer_lambda(model, layer_idx).to(dev)
        torch.save({"lambda": Lambda.cpu()}, lam_path)
    lam_stats = {"mean": Lambda.mean().item(), "min": Lambda.min().item(), "max": Lambda.max().item()}

    model.to(dev)
    X_train, X_test, Wg0, Wu0, Wd0, act_name = gmt.collect_layer_activations(
        model, layer_idx, train_ids, test_ids, dev)
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
    d = Wu0d.shape[0]
    K = int(math.floor((1 - args.sparsity) * d))
    _, mask_a0_test = glt.compute_group_mask(I0_test, colnorm0, K, g)

    # ---------------------------------------------------------------- B1D
    if os.path.exists(b1d_out) and os.path.exists(b1d_meta):
        glt.log(f"layer {layer_idx}: b1d exists, skipping")
        eps_star = json.load(open(b1d_meta))["D_ell"]
    else:
        b1d_args = copy.copy(args)
        b1d_args.lambda_down = args.b1d_lambda_down
        b1d_args.group_sizes = str(g)
        m_b1d, Wu_b1d, Wg_b1d, Wd_b1d = glt.run_arm(
            "B1D", X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test,
            K, g, 0.0, b1d_args, args.train_seqs, return_weights=True, Lambda=Lambda)
        # B1D never touches Wu/Wg -> I_test is I0_test, mask is mask_a0_test
        eps_star = glt.measure_D(I0_test, mask_a0_test, Wd_b1d, I0_test, mask_a0_test, Wd0d, Lambda)
        torch.save({"wg": Wg_b1d.cpu(), "wu": Wu_b1d.cpu(), "wd": Wd_b1d.cpu()}, b1d_out)
        meta_b1d = {
            "arm": "B1D", "layer": layer_idx, "g": g, "K": K, "d": d, "sparsity": args.sparsity,
            "git_commit": glt.git_hash(), "seed": args.seed, "model": args.model,
            "train_seqs": args.train_seqs, "test_seqs": args.test_seqs, "seqlen": args.seqlen,
            "lambda_down": args.b1d_lambda_down, "cg_iters": args.cg_iters, "cg_tol": args.cg_tol,
            "lambda_lookahead_stats": lam_stats, "D_ell": eps_star, "metrics_a0": m_b1d,
        }
        with open(b1d_meta, "w") as fh:
            json.dump(meta_b1d, fh, indent=2)
        glt.log(f"layer {layer_idx} B1D g={g}: D_ell(=epsilon*)={eps_star:.4f} "
                f"maskrecon={m_b1d[f'group_mask_recon_relerr_g{g}']:.4f} "
                f"wdown_drift={m_b1d['wdown_drift']:.4f}")

    # ----------------------------------------------------------------- B1
    if os.path.exists(b1_out) and os.path.exists(b1_meta):
        glt.log(f"layer {layer_idx}: b1 exists, skipping")
    else:
        m_b1, Wu_b1, Wg_b1, Wd_b1, D_b1, attempt, mu_up, mu_gate, lambda_down, budget_ok = \
            fit_with_budget_retry(
                X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test, K, g, args,
                Lambda, eps_star, I0_test, mask_a0_test, args.mu_up, args.mu_gate, args.lambda_down,
                mask_bonus=None, log_prefix=f"layer {layer_idx} B1 g={g}")

        torch.save({"wg": Wg_b1.cpu(), "wu": Wu_b1.cpu(), "wd": Wd_b1.cpu()}, b1_out)
        meta_b1 = {
            "arm": "B1", "layer": layer_idx, "g": g, "K": K, "d": d, "sparsity": args.sparsity,
            "git_commit": glt.git_hash(), "seed": args.seed, "model": args.model,
            "train_seqs": args.train_seqs, "test_seqs": args.test_seqs, "seqlen": args.seqlen,
            "outer": args.b1_outer, "irls": args.irls, "cg_iters": args.cg_iters, "cg_tol": args.cg_tol,
            "mu_up_final": mu_up, "mu_gate_final": mu_gate, "lambda_down_final": lambda_down,
            "mu_up_requested": args.mu_up, "mu_gate_requested": args.mu_gate,
            "lambda_down_requested": args.lambda_down, "anchor_escalation": args.anchor_escalation,
            "attempts": attempt, "budget_ok": budget_ok,
            "epsilon_star": eps_star, "D_ell": D_b1, "lambda_lookahead_stats": lam_stats,
            "metrics_a2": m_b1,
        }
        with open(b1_meta, "w") as fh:
            json.dump(meta_b1, fh, indent=2)
        glt.log(f"layer {layer_idx} B1 g={g} FINAL attempt={attempt}: D_ell={D_b1:.4f} "
                f"budget_ok={budget_ok} "
                f"maskrecon={m_b1[f'group_mask_recon_relerr_g{g}']:.4f} "
                f"drift(u/g/d)={m_b1['wup_drift']:.4f}/{m_b1['wgate_drift']:.4f}/{m_b1['wdown_drift']:.4f}")
        del m_b1, Wu_b1, Wg_b1, Wd_b1
        glt.empty_cache()

    # ----------------------------------------------------------------- B2
    # Opt-in (args.betas non-empty): "reuse bonus" mask, S~_j(T) = S_j(T) +
    # beta*s0*rho_j. rho_j (calibration group-mask selection frequency) is
    # measured on the ORIGINAL activations/mask (anti-circularity, same
    # convention as colnorm0/Lambda always being frozen-original) -- NOT
    # recomputed per beta or per outer iteration. Reuses the same
    # budget-retry machinery as B1 against the SAME eps_star.
    if args.betas:
        rho = glt.compute_rho(I0_train, colnorm0, K, g)
        s0 = glt.measure_s0(I0_train, colnorm0, X_train.shape[0] // g, g)
        for beta in args.betas:
            b2_dir = os.path.join(out_root, "b2", str(g), str(beta))
            os.makedirs(b2_dir, exist_ok=True)
            b2_out = os.path.join(b2_dir, f"layer_{layer_idx}.pt")
            b2_meta = os.path.join(b2_dir, f"layer_{layer_idx}.meta.json")
            if os.path.exists(b2_out) and os.path.exists(b2_meta):
                glt.log(f"layer {layer_idx}: b2 beta={beta} exists, skipping")
                continue
            bonus = beta * s0 * rho
            m_b2, Wu_b2, Wg_b2, Wd_b2, D_b2, attempt, mu_up, mu_gate, lambda_down, budget_ok = \
                fit_with_budget_retry(
                    X_train, X_test, f, Wg, Wu0d, Wd0d, colnorm0, Y_train, Y_test, K, g, args,
                    Lambda, eps_star, I0_test, mask_a0_test, args.mu_up, args.mu_gate, args.lambda_down,
                    mask_bonus=bonus, log_prefix=f"layer {layer_idx} B2 g={g} beta={beta}")
            torch.save({"wg": Wg_b2.cpu(), "wu": Wu_b2.cpu(), "wd": Wd_b2.cpu(),
                        "bonus": bonus.cpu()}, b2_out)
            meta_b2 = {
                "arm": "B2", "layer": layer_idx, "g": g, "beta": beta, "K": K, "d": d,
                "sparsity": args.sparsity, "git_commit": glt.git_hash(), "seed": args.seed,
                "model": args.model, "train_seqs": args.train_seqs, "test_seqs": args.test_seqs,
                "seqlen": args.seqlen, "outer": args.b1_outer, "irls": args.irls,
                "cg_iters": args.cg_iters, "cg_tol": args.cg_tol,
                "mu_up_final": mu_up, "mu_gate_final": mu_gate, "lambda_down_final": lambda_down,
                "anchor_escalation": args.anchor_escalation, "attempts": attempt, "budget_ok": budget_ok,
                "epsilon_star": eps_star, "D_ell": D_b2, "lambda_lookahead_stats": lam_stats,
                "s0": s0, "rho_stats": {"mean": rho.mean().item(), "min": rho.min().item(),
                                         "max": rho.max().item()},
                "metrics_a2": m_b2,
            }
            with open(b2_meta, "w") as fh:
                json.dump(meta_b2, fh, indent=2)
            glt.log(f"layer {layer_idx} B2 g={g} beta={beta} FINAL attempt={attempt}: "
                    f"D_ell={D_b2:.4f} budget_ok={budget_ok} "
                    f"maskrecon={m_b2[f'group_mask_recon_relerr_g{g}']:.4f}")
            del m_b2, Wu_b2, Wg_b2, Wd_b2
            glt.empty_cache()

    del X_train, X_test, Y_train, Y_test, Wg, Wu0d, Wd0d, colnorm0, I0_train, I0_test
    glt.empty_cache()


def selftest():
    """CPU-only: budget-retry predicate + resume skip (no GPU/model needed)."""
    assert should_retry(0.5, 0.3, 1, 3) is True, "over budget, attempts left -> retry"
    assert should_retry(0.2, 0.3, 1, 3) is False, "under budget -> no retry"
    assert should_retry(0.5, 0.3, 3, 3) is False, "over budget but attempts exhausted -> stop"
    assert should_retry(0.5, None, 1, 3) is False, "no budget info -> never blocks"
    assert should_retry(0.5, 0.0, 1, 3) is False, "epsilon_star<=0 -> never blocks (degenerate, not zero-tolerance)"
    print("selftest 1/2 OK (should_retry: escalate while over-budget and attempts remain, else stop)")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for arm_dir in ("b1d/16", "b1/16"):
            full = os.path.join(td, arm_dir)
            os.makedirs(full, exist_ok=True)
            open(os.path.join(full, "layer_5.pt"), "w").close()
            open(os.path.join(full, "layer_5.meta.json"), "w").close()

        class Boom:
            def to(self, *a, **k):
                raise AssertionError("process_layer should have returned before touching the model")

        class NS:
            pass
        a = NS()
        a.device = "cpu"
        a.betas = []
        process_layer(Boom(), 5, None, None, 16, a, td)
        # betas requested but not yet computed -> must NOT resume-skip (would
        # silently drop the B2 stage the caller asked for)
        a2 = NS()
        a2.device = "cpu"
        a2.betas = [0.1]
        try:
            process_layer(Boom(), 5, None, None, 16, a2, td)
            raise AssertionError("expected Boom.to() to fire (process_layer should NOT resume-skip "
                                  "when requested betas are missing)")
        except AssertionError as e:
            assert "should have returned" in str(e), e
    print("selftest 2/2 OK (resume: b1d+b1 skip when done; missing requested B2 betas force re-entry)")

    print("selftest ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--layers", default="0-30", help="inclusive range 'lo-hi'")
    ap.add_argument("--g", type=int, required=False)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--train_seqs", type=int, default=32)
    ap.add_argument("--test_seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    # B1 anchor: "at least as strong as stage-2's v2 values" (3.0/3.0/1.0) --
    # these ARE the v2 values, CLI-overridable for the budget-escalation
    # retry loop's caller-provided starting point (not the escalation itself,
    # which multiplies from here by --anchor_escalation).
    ap.add_argument("--mu_up", type=float, default=3.0)
    ap.add_argument("--mu_gate", type=float, default=3.0)
    ap.add_argument("--lambda_down", type=float, default=1.0)
    ap.add_argument("--b1d_lambda_down", type=float, default=1.0)
    ap.add_argument("--b1_outer", type=int, default=1, help="B1's outer-loop count (1-2 per request)")
    ap.add_argument("--irls", type=int, default=1, help="moot at lambda_rel=0 (solve_row_irls forces 1 iter)")
    ap.add_argument("--cg_iters", type=int, default=25)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--anchor_escalation", type=float, default=3.0)
    ap.add_argument("--max_budget_retries", type=int, default=3)
    ap.add_argument("--betas", default="",
                     help="comma-separated B2 beta values (e.g. '0.1,0.3,1.0'); "
                          "empty (default) skips B2 entirely -- opt-in, run only after "
                          "B1D/B1 clear the go/no-go gate on real PPL")
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
    args.betas = [float(v) for v in args.betas.split(",") if v.strip()]
    torch.manual_seed(args.seed)
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
    os.makedirs(os.path.join(args.out, "b1d", str(args.g)), exist_ok=True)
    with open(os.path.join(args.out, "b1d", str(args.g), "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    glt.log(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.eval()

    for li in layer_ids:
        t0 = time.time()
        process_layer(model, li, train_ids, test_ids, args.g, args, args.out)
        glt.log(f"=== layer {li} done in {time.time() - t0:.1f}s ===")

    glt.log("all layers done.")


if __name__ == "__main__":
    main()
