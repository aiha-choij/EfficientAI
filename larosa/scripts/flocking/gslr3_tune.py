"""GSLR stage-3 driver: joint anchored-ridge compensation on top of B1D
(topic: groupwise-flocking-tuning, follow-up to gslr-stage2.5 / 69f4f63).

Stage-2.5 found B1D (W_down-only, Lambda-weighted anchored group refit,
W_up/W_gate frozen -- Dead End: never touch them, stage-2.5's B1 lost
55-75x by jointly refitting all three) is the only arm so far that beats
the untuned group-masked baseline (A0) on full-model PPL. Stage-3's
question: does jointly fitting a compensation branch for the DROPPED
neurons' contribution -- on top of B1D's same z = mask*intermediate
regressor, solved as ONE joint anchored ridge, not a separate step --
close more of the remaining gap to the per-token anchor (8.11, oracle-
residual-sparsity precedent) or even the dense reference (5.4735)?

  ML model, per layer (independent, dense teacher y*, fp32):
    yhat_t = Wd_tilde @ z_t + Theta @ phi_t,   z_t = mask_t * i_t
  solved as ONE joint anchored ridge over [z ; phi] (gslr_layer_tune.
  refit_joint_weighted), anchor Wd_tilde -> Wd0, Theta -> 0. Mask/score
  identical to every prior GSLR stage (gauge score |i|*colnorm0, group
  top-K, colnorm0 always the ORIGINAL W_down -- no change here).

Arm ladder (see gslr3_sketch.py for phi's definition):
  C0 -- phi absent. Must reproduce B1D bit-for-bit (refit_wdown_weighted is
        now a thin wrapper around refit_joint_weighted -- see
        gslr_layer_tune.py's stage-3 docstring note on that function).
  C1 -- phi_t = P_r (x_t - mu), top-r PCA of calibration x.
  C2 -- phi_t = A_d[(1-mask_t)*(ghat_t*uhat_t)], ghat/uhat = rank-r_sk SVD
        sketches of the ORIGINAL W_gate/W_up (never refit, matches the
        Dead End rule -- the sketch is a fixed READ of the frozen matrices,
        not a weight update), A_d = rank-r_sk factor of the ORIGINAL W_down.
  C2a -- C2 with uhat replaced by the EXACT up-projection (diagnostic only,
        not deployable -- isolates the up-sketch's own quality ceiling).

No safety-budget retry loop here (stage-2.5's B1D/B1 contrast): every arm
in this ladder only ever fits (Wd, Theta) -- W_up/W_gate stay at their
untouched originals exactly like B1D, which was the one stage-2.5 arm that
never needed budget escalation to begin with (it IS the budget's own
source, eps_star). Structurally this ladder is "B1D plus an extra
zero-anchored regressor block", not a new joint-multi-matrix fit, so the
stage-2.5 instability finding (jointly refitting 3 matrices breaks the
local-Hessian approximation) does not apply here by construction.

Resumable per (arm, g, r) directory / per layer, same GPU-memory discipline
as gslr25_tune.py (live model moved to CPU before the fp32 N x d retuning
buffers are allocated). Lambda_ell cached once per layer under
<out_root>/lambda/ (shared across every arm/g/r combination -- it depends
only on layer ell+1's ORIGINAL weights).
"""

import argparse
import json
import math
import os
import time

import torch

import gslr_layer_tune as glt
import gslr_multilayer_tune as gmt
import gslr25_lambda as g25l
import gslr3_sketch as g3s


# ---------------------------------------------------------------- core fit

def fit_comp_arm(I_masked, phi_train, Y_train, Wd0, Lambda, lam, cg_iters, cg_tol):
    """One joint anchored-ridge solve. phi_train (N,r) or None (C0).
    Returns (Wd (h,d), Theta (h,r) or None)."""
    d = Wd0.shape[1]
    h = Wd0.shape[0]
    if phi_train is None:
        Z = I_masked
        Theta0 = Wd0
    else:
        Z = torch.cat([I_masked, phi_train], dim=1)
        r = phi_train.shape[1]
        Theta0 = torch.cat([Wd0, torch.zeros(h, r, device=Wd0.device, dtype=Wd0.dtype)], dim=1)
    W = glt.refit_joint_weighted(Z, Y_train, Theta0, Lambda, lam, d, cg_iters, cg_tol)
    Wd = W[:, :d].contiguous()
    Theta = W[:, d:].contiguous() if phi_train is not None else None
    return Wd, Theta


def measure_D_comp(I_arm, mask_arm, Wd_arm, phi_arm, Theta_arm, I_a0, mask_a0, Wd0, Lambda, chunk=8192):
    """Same synthesis safety budget as gslr_layer_tune.measure_D, extended
    to optionally include the compensation branch's contribution to this
    arm's own group-masked forward (phi_arm/Theta_arm both None reproduces
    measure_D exactly -- C0's own D_ell must match a plain B1D D_ell)."""
    y_arm = glt.chunked_mm(I_arm * mask_arm, Wd_arm.T, chunk)
    if phi_arm is not None:
        y_arm = y_arm + glt.chunked_mm(phi_arm, Theta_arm.T, chunk)
    y_a0 = glt.chunked_mm(I_a0 * mask_a0, Wd0.T, chunk)
    Lsqrt = Lambda.to(y_arm.device, torch.float32).clamp(min=0).sqrt()
    diff_norm = ((y_arm - y_a0) * Lsqrt[None, :]).norm(dim=1)
    base_norm = (y_a0 * Lsqrt[None, :]).norm(dim=1)
    return (diff_norm.mean() / base_norm.mean().clamp(min=1e-12)).item()


# ---------------------------------------------------------------- driver

def arm_dir_name(arm, g, r):
    return f"{arm}_{g}_{r}"


def process_layer(model, layer_idx, train_ids, test_ids, arm, g, r, args, out_root):
    adir = os.path.join(out_root, arm_dir_name(arm, g, r))
    lam_dir = os.path.join(out_root, "lambda")
    os.makedirs(adir, exist_ok=True)
    os.makedirs(lam_dir, exist_ok=True)

    out_pt = os.path.join(adir, f"layer_{layer_idx}.pt")
    out_meta = os.path.join(adir, f"layer_{layer_idx}.meta.json")
    if os.path.exists(out_pt) and os.path.exists(out_meta):
        glt.log(f"layer {layer_idx} arm={arm} g={g} r={r}: output exists, skipping (resume)")
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
    Wg, Wu, Wd0 = Wg0.to(dev), Wu0.to(dev), Wd0.to(dev)
    f = glt.act_fn(act_name)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()

    I_train = glt.intermediate(X_train, f(glt.linear_up(X_train, Wg)), Wu)
    I_test = glt.intermediate(X_test, f(glt.linear_up(X_test, Wg)), Wu)
    Y_train = glt.chunked_mm(I_train, Wd0.T)
    Y_test = glt.chunked_mm(I_test, Wd0.T)
    d = Wu.shape[0]
    K = int(math.floor((1 - args.sparsity) * d))

    _, mask_train = glt.compute_group_mask(I_train, colnorm0, K, g)
    _, mask_test = glt.compute_group_mask(I_test, colnorm0, K, g)
    I_masked_train = I_train * mask_train

    payload = {"wg": Wg.cpu(), "wu": Wu.cpu()}
    meta = {
        "arm": arm, "layer": layer_idx, "g": g, "r": r, "K": K, "d": d,
        "sparsity": args.sparsity, "git_commit": glt.git_hash(), "seed": args.seed,
        "model": args.model, "train_seqs": args.train_seqs, "test_seqs": args.test_seqs,
        "seqlen": args.seqlen, "lambda_down": args.lambda_down, "cg_iters": args.cg_iters,
        "cg_tol": args.cg_tol, "lambda_lookahead_stats": lam_stats,
    }

    if arm in ("c0", "b1dprime"):
        # b1dprime = the request's REQUIRED frontier control: B1D re-run at
        # a larger K' = K + compFLOPs/(3h) (same total per-token compute as
        # whichever compensation arm r is matched to -- see the CLI's
        # --sparsity override; this branch's fit logic is IDENTICAL to c0,
        # only --sparsity/--r differ between invocations, so a "does
        # b1dprime==c0 at the same sparsity" regression check is exact by
        # construction, not approximate).
        Wd, _ = fit_comp_arm(I_masked_train, None, Y_train, Wd0, Lambda, args.lambda_down,
                              args.cg_iters, args.cg_tol)
        phi_test = None
        Theta = None
    elif arm == "c1":
        mu, P = g3s.compute_pca_basis(X_train, r)
        phi_train = g3s.apply_pca(X_train, mu, P)
        phi_test = g3s.apply_pca(X_test, mu, P)
        Wd, Theta = fit_comp_arm(I_masked_train, phi_train, Y_train, Wd0, Lambda,
                                  args.lambda_down, args.cg_iters, args.cg_tol)
        payload["pca_mu"] = mu.cpu()
        payload["pca_P"] = P.cpu()
        meta["comp_kind"] = "c1"
        meta["r_eff"] = P.shape[0]
    elif arm in ("c2", "c2a"):
        Ag, Bg, Au, Bu, Ad, Bd = g3s.build_c2_sketch(Wg, Wu, Wd0, r)
        exact_u_train = glt.linear_up(X_train, Wu) if arm == "c2a" else None
        exact_u_test = glt.linear_up(X_test, Wu) if arm == "c2a" else None
        phi_train = g3s.compute_c2_phi(X_train, mask_train, f, Ag, Bg, Au, Bu, Ad,
                                        exact_u=exact_u_train)
        phi_test = g3s.compute_c2_phi(X_test, mask_test, f, Ag, Bg, Au, Bu, Ad,
                                       exact_u=exact_u_test)
        Wd, Theta = fit_comp_arm(I_masked_train, phi_train, Y_train, Wd0, Lambda,
                                  args.lambda_down, args.cg_iters, args.cg_tol)
        payload["comp_kind"] = arm
        meta["comp_kind"] = arm
        meta["r_sk"] = int(Ad.shape[0])
    else:
        raise ValueError(f"unknown arm {arm!r}")

    payload["wd"] = Wd.cpu()
    if Theta is not None:
        payload["theta"] = Theta.cpu()

    D_ell = measure_D_comp(I_test, mask_test, Wd, phi_test, Theta, I_test, mask_test, Wd0, Lambda)
    wdown_drift = ((Wd - Wd0).norm() / Wd0.norm()).item()
    meta["D_ell"] = D_ell
    meta["wdown_drift"] = wdown_drift
    if Theta is not None:
        meta["theta_norm"] = Theta.norm().item()

    torch.save(payload, out_pt)
    with open(out_meta, "w") as fh:
        json.dump(meta, fh, indent=2)
    glt.log(f"layer {layer_idx} arm={arm} g={g} r={r}: D_ell={D_ell:.4f} "
            f"wdown_drift={wdown_drift:.4f}"
            + (f" theta_norm={meta['theta_norm']:.4f}" if Theta is not None else ""))

    del X_train, X_test, Y_train, Y_test, I_train, I_test, mask_train, mask_test, I_masked_train
    del Wg, Wu, Wd0, colnorm0
    glt.empty_cache()


# ---------------------------------------------------------------- selftest

def selftest_gslr3():
    """CPU-only, the stage-3 request's 5 required unit tests (1/2/3/4 fully
    synthetic here; 5 is the "diagnostic ceiling must not be violated"
    sanity -- see its own docstring below for what it does and doesn't
    prove). Unit test 1's REAL confirmation (C0 PPL == B1D PPL on actual
    LLaMA2-7B activations) is a GPU regression job, not reproducible here;
    what CAN be checked on CPU is the stronger, structural claim this
    module's implementation relies on: fit_comp_arm(arm="c0") calls
    refit_joint_weighted with an EMPTY phi block and IS, by construction
    (gslr_layer_tune.refit_wdown_weighted's stage-3 delegation), the exact
    same code path refit_wdown_weighted uses -- so if this synthetic check
    passes, C0's real-data PPL is mathematically guaranteed to reproduce
    B1D's, not merely "expected to"."""
    torch.manual_seed(20)
    N, h, d, g = 400, 14, 24, 8
    X = torch.randn(N, h)
    Wg0 = torch.randn(d, h) / math.sqrt(h)
    Wu0 = torch.randn(d, h) / math.sqrt(h)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    f = glt.act_fn("silu")
    I0 = glt.intermediate(X, f(X @ Wg0.T), Wu0)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    K = d // 2
    _, mask = glt.compute_group_mask(I0, colnorm0, K, g)
    Y = I0 @ Wd0.T + 0.02 * torch.randn(N, h)
    Lambda = 1.0 + torch.rand(h) * 2.0
    lam = 0.3

    # ---- unit test 1 (structural half, see docstring above): C0 (phi
    # absent) reproduces refit_wdown_weighted bit-for-bit.
    Wd_c0, Theta_c0 = fit_comp_arm(I0 * mask, None, Y, Wd0, Lambda, lam, 200, 1e-10)
    Wd_direct = glt.refit_wdown_weighted(I0 * mask, Y, Wd0, Lambda, lam, 200, 1e-10)
    assert Theta_c0 is None
    err1 = (Wd_c0 - Wd_direct).norm() / Wd_direct.norm().clamp(min=1e-12)
    assert err1 < 1e-6, f"C0 (empty phi) must match refit_wdown_weighted exactly: relerr={err1:.2e}"
    print(f"selftest_gslr3 1/5 OK (C0 == refit_wdown_weighted exactly, relerr={err1:.2e})")

    # ---- unit test 2: phi FORCED to zero (present, r>0, but literally all
    # zeros) -> joint solve's z-block matches the standalone refit AND the
    # phi-block's fitted Theta stays at its zero anchor.
    r = 5
    phi_zero = torch.zeros(N, r)
    Wd_z0, Theta_z0 = fit_comp_arm(I0 * mask, phi_zero, Y, Wd0, Lambda, lam, 200, 1e-10)
    err2a = (Wd_z0 - Wd_direct).norm() / Wd_direct.norm().clamp(min=1e-12)
    err2b = Theta_z0.norm().item()
    assert err2a < 1e-6, f"phi=0: Wd block must match standalone refit: relerr={err2a:.2e}"
    assert err2b < 1e-6, f"phi=0: Theta must stay at its zero anchor: norm={err2b:.2e}"
    print(f"selftest_gslr3 2/5 OK (phi forced to 0: Wd matches standalone refit "
          f"relerr={err2a:.2e}, Theta stays 0 norm={err2b:.2e})")

    # ---- unit test 3: r_sk = full rank + lam -> 0 => C2's compensated
    # forward recovers the EXACT dense (unmasked) reconstruction, i.e. the
    # tail contribution Theta@phi exactly cancels what the mask dropped.
    r_sk_full = max(d, h)  # svd_lr clamps internally to min(d,h)
    Ag, Bg, Au, Bu, Ad, Bd = g3s.build_c2_sketch(Wg0, Wu0, Wd0, r_sk_full)
    phi_full = g3s.compute_c2_phi(X, mask, f, Ag, Bg, Au, Bu, Ad)
    Y_dense = I0 @ Wd0.T  # noiseless dense target (no calibration noise this time)
    Wd_c2, Theta_c2 = fit_comp_arm(I0 * mask, phi_full, Y_dense, Wd0, Lambda, 1e-6, 500, 1e-12)
    yhat = (I0 * mask) @ Wd_c2.T + phi_full @ Theta_c2.T
    relerr3 = (yhat - Y_dense).norm() / Y_dense.norm()
    assert relerr3 < 1e-2, (
        f"C2 at full-rank sketch + lam->0 must recover the exact dense reconstruction: "
        f"relerr={relerr3:.2e}")
    print(f"selftest_gslr3 3/5 OK (r_sk=full, lam->0: C2 tail recovers exact dense recon, "
          f"relerr={relerr3:.2e})")

    # ---- unit test 4: joint Gram is symmetric and PSD.
    r4 = 6
    phi4 = torch.randn(N, r4)
    Z = torch.cat([I0 * mask, phi4], dim=1)
    G = glt.chunked_matmul(Z, Z)
    sym_err = (G - G.T).abs().max().item()
    assert sym_err < 1e-3, f"joint Gram must be symmetric: max|G-G.T|={sym_err:.2e}"
    min_eig = torch.linalg.eigvalsh(G).min().item()
    assert min_eig > -1e-2, f"joint Gram must be PSD: min eigenvalue={min_eig:.4f}"
    print(f"selftest_gslr3 4/5 OK (joint Gram symmetric, max|G-G.T|={sym_err:.2e}; "
          f"PSD, min eig={min_eig:.4f})")

    # ---- unit test 5: C2a (exact u) must not lose to C2 (sketched u) --
    # guards against WIRING bugs (e.g. accidentally feeding the sketch into
    # both arms, or a transposed A/B) rather than a general theorem: the
    # data here is generated so the TRUE tail signal is a function of the
    # EXACT u, which only C2a's phi can represent losslessly (C2's uhat is
    # deliberately truncated to a low rank, r_sk_small < true rank(Wu0), so
    # it cannot). If this check fails on data this clean, the almost
    # certain explanation is a bug (e.g. C2a silently still using the
    # sketch), which is the failure mode the stage-3 request calls out.
    torch.manual_seed(21)
    d5, h5, r_sk_small = 40, 10, 3
    Wg5 = torch.randn(d5, h5) / math.sqrt(h5)
    Wu5 = torch.randn(d5, h5) / math.sqrt(h5)
    Wd5 = torch.randn(h5, d5) / math.sqrt(d5)
    X5 = torch.randn(600, h5)
    I5 = glt.intermediate(X5, f(X5 @ Wg5.T), Wu5)
    colnorm5 = Wd5.pow(2).sum(0).sqrt()
    _, mask5 = glt.compute_group_mask(I5, colnorm5, d5 // 2, 4)
    Y5 = I5 @ Wd5.T  # noiseless: the true tail is exactly (1-mask)*(g*u) @ Wd5.T
    Lambda5 = torch.ones(h5)
    Ag5, Bg5, Au5, Bu5, Ad5, Bd5 = g3s.build_c2_sketch(Wg5, Wu5, Wd5, r_sk_small)
    u5_exact = X5 @ Wu5.T
    phi_c2 = g3s.compute_c2_phi(X5, mask5, f, Ag5, Bg5, Au5, Bu5, Ad5)
    phi_c2a = g3s.compute_c2_phi(X5, mask5, f, Ag5, Bg5, Au5, Bu5, Ad5, exact_u=u5_exact)
    lam5 = 1e-4
    Wd_c2_5, Th_c2_5 = fit_comp_arm(I5 * mask5, phi_c2, Y5, Wd5, Lambda5, lam5, 500, 1e-12)
    Wd_c2a_5, Th_c2a_5 = fit_comp_arm(I5 * mask5, phi_c2a, Y5, Wd5, Lambda5, lam5, 500, 1e-12)
    mse_c2 = ((I5 * mask5) @ Wd_c2_5.T + phi_c2 @ Th_c2_5.T - Y5).pow(2).mean().item()
    mse_c2a = ((I5 * mask5) @ Wd_c2a_5.T + phi_c2a @ Th_c2a_5.T - Y5).pow(2).mean().item()
    assert mse_c2a <= mse_c2 + 1e-6 * abs(mse_c2), (
        f"C2a (exact u) must not lose to C2 (sketched u) -- diagnostic ceiling violated "
        f"(bug): mse_c2a={mse_c2a:.3e} > mse_c2={mse_c2:.3e}")
    print(f"selftest_gslr3 5/5 OK (C2a in-sample MSE {mse_c2a:.3e} <= C2's {mse_c2:.3e}, "
          f"ceiling holds)")

    print("selftest_gslr3 ALL OK")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--layers", default="0-30", help="inclusive range 'lo-hi'")
    ap.add_argument("--arm", required=False, choices=["c0", "c1", "c2", "c2a", "b1dprime"])
    ap.add_argument("--g", type=int, required=False)
    ap.add_argument("--r", type=int, default=0,
                     help="C1: PCA rank. C2/C2a: r_sk (gate/up/down sketch rank). "
                          "C0: unused (kept 0, folded into the output dir name). "
                          "b1dprime (frontier control): the target K' or the compensation "
                          "arm's own r/r_sk, purely a label for the output dir name -- pass "
                          "--sparsity separately to actually set K' (see the request's "
                          "K' = K + compFLOPs/(3h) formula).")
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--train_seqs", type=int, default=32)
    ap.add_argument("--test_seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--lambda_down", type=float, default=1.0,
                     help="same default as stage-2.5's B1D (--b1d_lambda_down)")
    ap.add_argument("--cg_iters", type=int, default=25)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest_gslr3()
        return

    assert args.out_root, "--out_root is required"
    assert args.arm is not None, "--arm is required"
    assert args.g is not None, "--g is required"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_root, exist_ok=True)

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

    adir = os.path.join(args.out_root, arm_dir_name(args.arm, args.g, args.r))
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    glt.log(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.eval()

    for li in layer_ids:
        t0 = time.time()
        process_layer(model, li, train_ids, test_ids, args.arm, args.g, args.r, args, args.out_root)
        glt.log(f"=== layer {li} done in {time.time() - t0:.1f}s ===")

    glt.log("all layers done.")


if __name__ == "__main__":
    main()
