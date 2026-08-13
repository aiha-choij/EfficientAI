"""GSLR (Group-Sparse Local Retuning) single-layer stage-1 PoC
(topic: groupwise-flocking-tuning, follow-up to flocking-poc-l16 / ad8c1b3).

flocking PoC (design A: dense recon + raw-i l2,1 penalty, W_up+W_down only,
W_gate frozen) FAILED on LLaMA2-7B layer 16: within-group overlap 22->24% at
max lambda, output error 15%. Diagnosis: (i) token-dependence of selection
lives in the frozen W_gate; (ii) dense-recon protection deadlocks with the
penalty; (iii) a penalty on raw i leaks through the W_up/W_down scale gauge.

GSLR fixes all three:
- design B: reconstruction target is the GROUP-MASKED output (not dense) ->
  neurons outside the current mask feel only the anchor, not a recon pull
  back to their dense role. Solved as an exact per-row block-coordinate
  target (hold all other neurons' masked contribution fixed; the row's
  optimal contribution is the projection of the residual onto W_down's
  column direction).
- gauge-invariant penalty: xi_tj = i_tj * ||W_down0[:,j]|| (ORIGINAL W_down
  column norm, frozen across the whole run -- anti-circularity with the
  Top-K score gauge). R = lam * sum_T sum_j ||xi_j(T)||_2, IRLS-majorized
  (quadratic per iteration, same per-row CG machinery as W_up).
- W_gate is unfrozen: Gauss-Newton linearization of the activation function
  sigma(a) ~= sigma(a0) + sigma'(a0)*(a-a0) around the current gate
  pre-activation (sigma' from autograd on the actual act_fn, never
  hardcoded) turns the W_gate subproblem into the SAME affine-quadratic
  form as W_up (coef, base per token-neuron) and reuses solve_row_irls.

Arm ladder (cumulative, isolates which fix matters):
  A0 = original weights, no tuning (baseline, reproduces flocking PoC's
       baseline numbers exactly since the eval pipeline is unchanged).
  A1 = design B + gauge penalty, W_up + W_down only (W_gate frozen).
  A2 = A1 + W_gate GN step (all three matrices).

Outer loop (per (arm, group_size, lambda)):
  1. recompute group mask from current I and FROZEN colnorm0 gauge
  2. W_down: closed-form anchored ridge on the MASKED intermediate
     (reuses refit_wdown verbatim, I -> I*mask)
  3. W_up: solve_row_irls (masked recon target + gauge penalty)
  4. W_gate (A2 only): GN-linearize sigma around current W_gate, then
     solve_row_irls on the resulting affine problem
  5. lambda ramp: lambda_eff(k) = lambda * (k+1)/outer

All CPU selftests are synthetic (small random tensors) and check the math,
not the GPU pipeline.
"""

import argparse
import json
import math
import os
import subprocess
import time

import torch


# ---------------------------------------------------------------- utilities

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True).strip()
    except Exception:
        return "unknown"


def chunked_matmul(A, B, chunk=8192):
    """A.T @ B (p x q), accumulated in fp32 over row chunks of A/B (N x p, N x q)."""
    out = torch.zeros(A.shape[1], B.shape[1], dtype=torch.float32, device=A.device)
    for s in range(0, A.shape[0], chunk):
        out += A[s:s + chunk].T @ B[s:s + chunk]
    return out


def chunked_mm(A, B, chunk=8192):
    """A @ B (N x q), A (N x p), B (p x q), chunked over N for memory."""
    out = torch.empty(A.shape[0], B.shape[1], dtype=torch.float32, device=A.device)
    for s in range(0, A.shape[0], chunk):
        out[s:s + chunk] = A[s:s + chunk] @ B
    return out


# ---------------------------------------------------------------- data

def collect_activations(args):
    """Frozen-model forward pass; capture target layer's MLP input on
    calibration (wikitext-2 train) and held-out (wikitext-2 test) streams.
    Returns X_train, X_test (N x h fp32 cpu), Wg, Wu, Wd (fp32 cpu), act name."""
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.to(args.device).eval()

    layer = model.model.layers[args.layer]
    Wg = layer.mlp.gate_proj.weight.detach().float().cpu()
    Wu = layer.mlp.up_proj.weight.detach().float().cpu()
    Wd = layer.mlp.down_proj.weight.detach().float().cpu()
    act_name = model.config.hidden_act

    captured = []
    hook = layer.mlp.register_forward_pre_hook(
        lambda mod, inp: captured.append(inp[0].detach().squeeze(0).to(torch.float16).cpu()))

    def run(ids):
        captured.clear()
        with torch.no_grad():
            for i in range(ids.shape[0]):
                model(ids[i:i + 1].to(args.device))
        return torch.cat(captured, 0)

    X_train = run(train_ids)
    X_test = run(test_ids)
    hook.remove()
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return X_train, X_test, Wg, Wu, Wd, act_name


# ---------------------------------------------------------------- core math

def act_fn(name):
    import torch.nn.functional as F
    return {"silu": F.silu, "gelu": F.gelu}[name]


def act_grad(f, a, chunk=8192):
    """sigma'(a), elementwise, via autograd -- never hardcode the derivative.
    Chunked over rows: building one autograd graph over the full (N x d)
    pre-activation OOMs on a 40GB card at LLaMA2-7B's d=11008 (the forward
    + backward buffers for the whole tensor roughly double its footprint on
    top of everything else already resident)."""
    out = torch.empty_like(a)
    for s in range(0, a.shape[0], chunk):
        seg = a[s:s + chunk].detach().clone().requires_grad_(True)
        val = f(seg)
        g, = torch.autograd.grad(val.sum(), seg)
        out[s:s + chunk] = g.detach()
    return out


def intermediate(X, Gate, Wu, chunk=8192):
    """i = Gate * (X @ Wu.T), fp32 chunked. X (N x h), Gate (N x d)."""
    out = torch.empty(X.shape[0], Wu.shape[0], dtype=torch.float32, device=X.device)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = Gate[s:s + chunk] * (X[s:s + chunk] @ Wu.T)
    return out


def linear_up(X, Wu, chunk=8192):
    """u = X @ Wu.T, fp32 chunked (no gating)."""
    out = torch.empty(X.shape[0], Wu.shape[0], dtype=torch.float32, device=X.device)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = X[s:s + chunk] @ Wu.T
    return out


def compute_group_mask(I, colnorm0, K, g):
    """Group-summed gauge-fixed score, Top-K per group.
    Returns group_mask (n_groups x d bool), token_mask (N x d bool, repeat_interleave)."""
    N, d = I.shape
    score = I.abs() * colnorm0[None, :]
    gscore = score.view(-1, g, d).sum(1)                     # G x d
    idx = gscore.topk(K, dim=-1).indices
    group_mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
    token_mask = group_mask.repeat_interleave(g, 0)
    return group_mask, token_mask


def refit_wdown(I, Y, Wd0, lam, chunk=8192):
    """Anchored ridge: min ||I Wd^T - Y||^2 + lam*D*||Wd - Wd0||^2,
    D = mean(diag(I^T I)). I (N x d) is the (possibly masked) regressor,
    Y (N x h) the frozen dense target. Verbatim from the flocking PoC --
    reused unchanged for design B by passing I*token_mask as I."""
    G = chunked_matmul(I, I, chunk)
    C = chunked_matmul(I, Y, chunk)
    dscale = lam * G.diagonal().mean()
    A = G + dscale * torch.eye(G.shape[0], device=G.device)
    WdT = torch.linalg.solve(A, C + dscale * Wd0.T)
    return WdT.T.contiguous()


def refit_wdown_weighted(I, Y, Wd0, Lambda, lam, cg_iters=50, cg_tol=1e-6, chunk=8192):
    """Lambda-weighted anchored ridge (stage-2.5 GSLR):
      min_Wd sum_h Lambda_h * ||I @ Wd[h,:] - Y[:,h]||^2 + lam*D*||Wd - Wd0||^2
    (D = mean(diag(I^T I)), SAME scalar ridge strength for every h -- the
    anchor is deliberately NOT scaled by Lambda, per the stage-2.5 request's
    "Lambda weights the reconstruction objective and D_ell only" design.

    Wd's rows (indexed by output/hidden dim h) each regress against the SAME
    design matrix I, so the exact per-h solution would need a different
    (Lambda_h*G + lam*D*I) system per h -- infeasible at LLaMA scale (d up to
    ~11008, thousands of h). Instead this solves the identical weighted
    normal equations via batched CG shared across all h columns at once
    (single G=I^T I formed once, like the unweighted closed form; Lambda_h
    only enters as a per-column elementwise scale inside the CG matvec) --
    exact in the same sense refit_wdown is exact (up to cg_tol), not an
    approximation of the objective. Lambda=ones reduces to refit_wdown's
    objective exactly (see selftest_25)."""
    G = chunked_matmul(I, I, chunk)                      # d x d, shared Gram
    C = chunked_matmul(I, Y, chunk)                      # d x h
    dscale = lam * G.diagonal().mean()
    Lam = Lambda.to(G.device, torch.float32)
    W0T = Wd0.T.contiguous()                             # d x h

    rhs = Lam[None, :] * C + dscale * W0T

    def matvec(V):
        return Lam[None, :] * (G @ V) + dscale * V

    W = W0T.clone()
    r = rhs - matvec(W)
    p = r.clone()
    rs = (r * r).sum(0)
    b2 = (rhs * rhs).sum(0).clamp(min=1e-30)
    for _ in range(cg_iters):
        Ap = matvec(p)
        alpha = rs / (p * Ap).sum(0).clamp(min=1e-30)
        W = W + alpha[None, :] * p
        r = r - alpha[None, :] * Ap
        rs_new = (r * r).sum(0)
        if (rs_new / b2).max() < cg_tol ** 2:
            break
        p = r + (rs_new / rs.clamp(min=1e-30))[None, :] * p
        rs = rs_new
    return W.T.contiguous()


def measure_D(I_arm, mask_arm, Wd_arm, I_a0, mask_a0, Wd_a0, Lambda, chunk=8192):
    """Synthesis safety budget D_ell (stage-2.5 request section 1):
      D = E_t||Lambda^0.5 (yhat_masked,theta - yhat_masked,theta0)|| /
          E_t||Lambda^0.5 yhat_masked,theta0||
    yhat_masked,theta  = (I_arm * mask_arm) @ Wd_arm.T   (this arm's own
                          group-masked forward, its own mask)
    yhat_masked,theta0 = (I_a0 * mask_a0) @ Wd_a0.T      (A0's group-masked
                          forward -- untuned weights, untuned mask)
    theta=theta0 (same I/mask/Wd passed for both sides) -> D=0 exactly."""
    y_arm = chunked_mm(I_arm * mask_arm, Wd_arm.T, chunk)
    y_a0 = chunked_mm(I_a0 * mask_a0, Wd_a0.T, chunk)
    Lsqrt = Lambda.to(y_arm.device, torch.float32).clamp(min=0).sqrt()
    diff_norm = ((y_arm - y_a0) * Lsqrt[None, :]).norm(dim=1)
    base_norm = (y_a0 * Lsqrt[None, :]).norm(dim=1)
    return (diff_norm.mean() / base_norm.mean().clamp(min=1e-12)).item()


def solve_row_irls(X, coef, base, I_cur, token_mask, resid_global, Wd, colnorm0,
                    w0, lam_rel, n_groups, group_size, mu_rel, irls_iters,
                    cg_iters, cg_tol, s0, row_chunk=1024, Lambda=None):
    """Generalized per-row IRLS + batched CG solver, shared by the W_up and
    (GN-linearized) W_gate subproblems.

    Model: i_tj(w_j) = base_tj + coef_tj * (x_t . w_j)   (affine in w_j;
    W_up has base=0 exactly, recovering the flocking-PoC row model).

    Objective per row j:
      masked recon: sum_t token_mask_tj * (r_tj - i_tj(w_j))^2
        r_tj = [W_d[j,:] . resid_global_t] / ||W_d[j,:]||^2 + I_cur_tj*mask_tj
        (exact block-coordinate optimum holding all other neurons' current
        masked contribution fixed; resid_global = Y - (I_cur*mask) @ Wd.T)
      gauge penalty (IRLS-majorized ell_2,1 on xi_tj = colnorm0_j * i_tj(w_j)):
        lam_rel * s0 * sum_T omega_j(T) * sum_{t in T} xi_tj^2,
        omega_j(T) = 1 / (2 * max(||xi_j(T)||, eps))
      anchor: mu_j ||w_j - w0_j||^2

    Lambda (h,), optional (stage-2.5 GSLR lookahead metric): weights the
    masked-recon target's block-coordinate projection in OUTPUT (h) space,
    i.e. r_tj's numerator/denominator both pick up a Lambda_h factor before
    projecting the h-space residual onto W_down's column j direction (see
    module docstring derivation in refit_wdown_weighted's Lambda case).
    Lambda=None (default) is EXACTLY the unweighted stage-1/2 formula --
    zero behavior change for existing A0/A1/A2 callers.

    X (N x h), coef/base/I_cur/token_mask (N x d) on device. w0, Wd, colnorm0
    (d x h / d) on device. Returns new W (d x h fp32)."""
    N, h = X.shape
    d = coef.shape[1]
    assert N == n_groups * group_size
    Wnew = w0.clone()
    xsq = (X * X).sum(1)
    eps = 1e-6 + 1e-3 * s0

    for start in range(0, d, row_chunk):
        J = slice(start, min(start + row_chunk, d))
        R = J.stop - J.start
        coefJ = coef[:, J]
        baseJ = base[:, J] if base is not None else torch.zeros(N, R, device=X.device)
        mJ = token_mask[:, J].float()
        colnormJ = colnorm0[J]
        W0J = w0[J].T.contiguous()                            # h x R

        # exact block-coordinate masked-recon target r_tj (fixed for this call)
        WdJ = Wd[:, J].T.contiguous()                         # R x h (columns of W_down)
        if Lambda is None:
            rg_dot = chunked_mm(resid_global, WdJ.T)           # N x R
            normsqJ = WdJ.pow(2).sum(1).clamp(min=1e-12)
        else:
            LamW = WdJ * Lambda[None, :]                       # R x h, Lambda-weighted columns
            rg_dot = chunked_mm(resid_global, LamW.T)           # N x R (Lambda folded into the projection)
            normsqJ = (WdJ * LamW).sum(1).clamp(min=1e-12)     # sum_h Lambda_h * WdJ_h^2
        rJ = rg_dot / normsqJ[None, :] + I_cur[:, J] * mJ
        targ = rJ - baseJ                                     # recon target'

        muJ = mu_rel * (coefJ.pow(2) * xsq[:, None]).sum(0) / h
        WJ = W0J.clone()

        for it in range(irls_iters if lam_rel > 0 else 1):
            z = X @ WJ                                        # N x R
            iJ = baseJ + coefJ * z
            xi = iJ * colnormJ[None, :]
            gnorm = xi.view(n_groups, group_size, R).pow(2).sum(1).sqrt()   # G x R
            omega = 1.0 / (2.0 * gnorm.clamp(min=eps))
            penw = lam_rel * s0 * omega.repeat_interleave(group_size, 0)     # N x R (0 if lam_rel==0)
            if lam_rel <= 0:
                penw = penw * 0.0

            c = mJ * coefJ.pow(2) + penw * colnormJ[None, :].pow(2) * coefJ.pow(2)
            rhs_target = mJ * targ - penw * colnormJ[None, :].pow(2) * baseJ
            b = X.T @ (coefJ * rhs_target) + muJ[None, :] * W0J

            def matvec(V):
                return X.T @ (c * (X @ V)) + muJ[None, :] * V

            r = b - matvec(WJ)
            p = r.clone()
            rs = (r * r).sum(0)
            b2 = (b * b).sum(0).clamp(min=1e-30)
            for _ in range(cg_iters):
                Ap = matvec(p)
                alpha = rs / (p * Ap).sum(0).clamp(min=1e-30)
                WJ = WJ + alpha[None, :] * p
                r = r - alpha[None, :] * Ap
                rs_new = (r * r).sum(0)
                if (rs_new / b2).max() < cg_tol ** 2:
                    break
                p = r + (rs_new / rs.clamp(min=1e-30))[None, :] * p
                rs = rs_new
        Wnew[J] = WJ.T
    return Wnew


def measure_s0(I, colnorm0, n_groups, group_size, sample=2048):
    """Median group-column gauge-scaled norm, used to keep lambda_rel a
    dimensionless relative knob (same convention as the flocking PoC's s0)."""
    xi = (I[:, :sample] * colnorm0[None, :sample]).view(n_groups, group_size, -1)
    gn = xi.pow(2).sum(1).sqrt()
    return gn.median().item()


# ---------------------------------------------------------------- metrics

def topk_sets(score, K):
    idx = score.topk(K, dim=-1).indices
    S = torch.zeros_like(score, dtype=torch.bool)
    S.scatter_(1, idx, True)
    return S


def metrics(I, colnorm0, K, n_seqs, seqlen, group_sizes, Wd_eval, Y, deltas):
    """Flocking + reconstruction + degeneracy metrics for activation matrix I (N x d)."""
    score = I.abs() * colnorm0[None, :]
    S = topk_sets(score, K)
    Sv = S.view(n_seqs, seqlen, -1)
    out = {}

    out["C_delta"] = {
        str(dl): (Sv[:, :-dl] & Sv[:, dl:]).sum(-1).float().mean().item() / K
        for dl in deltas if dl < seqlen}

    ynorm = Y.pow(2).sum().sqrt()
    out["dense_recon_relerr"] = ((I @ Wd_eval.T - Y).pow(2).sum().sqrt() / ynorm).item()

    for g in group_sizes:
        B = Sv.view(n_seqs, seqlen // g, g, -1)               # S x P x g x d
        pair = torch.einsum("sgtd,sgud->sgtu", B.float(), B.float())
        off = (pair.sum((-1, -2)) - pair.diagonal(dim1=-2, dim2=-1).sum(-1))
        out[f"within_group_overlap_g{g}"] = (off / (g * (g - 1)) / K).mean().item()
        out[f"union_tax_g{g}"] = (B.any(2).sum(-1).float() / K).mean().item()

        gscore = score.view(-1, g, score.shape[-1]).sum(1)    # G x d  (n_seqs*P groups)
        gidx = gscore.topk(K, dim=-1).indices
        m = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, gidx, True)
        m_tok = m.repeat_interleave(g, 0)
        out[f"group_mask_recon_relerr_g{g}"] = (
            ((I * m_tok) @ Wd_eval.T - Y).pow(2).sum().sqrt() / ynorm).item()

        # degeneracy check: static-mask fraction (neurons on in >=90% of groups)
        freq = m.float().mean(0)                              # d
        out[f"static_mask_frac_g{g}"] = ((freq >= 0.9).sum().float() / K).item()

        # cross-sequence vs within-sequence mask overlap ratio
        Pn = seqlen // g
        mv = m.view(n_seqs, Pn, -1)                            # S x P x d
        if n_seqs >= 2 and Pn >= 1:
            cross = (mv[0:1].float() * mv[1:].float()).sum(-1) / K            # (S-1) x P
            cross_overlap = cross.mean().item()
        else:
            cross_overlap = float("nan")
        within = out[f"within_group_overlap_g{g}"]
        out[f"cross_seq_overlap_g{g}"] = cross_overlap
        out[f"cross_within_ratio_g{g}"] = (
            cross_overlap / within if within > 1e-9 else float("nan"))
    return out


# ---------------------------------------------------------------- outer loop

def run_arm(arm, X_train, X_test, Gate_fn, Wg0, Wu0, Wd0, colnorm0, Y_train, Y_test,
            K, g, lam, args, n_seqs_train, return_weights=False, Lambda=None):
    """return_weights=True additionally returns the arm's final (Wu, Wg, Wd)
    tensors (Wg==Wg0/Wu==Wu0 for A0/A1 where that matrix isn't refit) --
    used by the stage-2 multi-layer driver to persist retuned weights.
    Metrics/logging behavior is unchanged when False (default).

    Lambda (h,), optional (stage-2.5 GSLR lookahead metric, added without
    disturbing stage-1/2 behavior): Lambda=None reproduces the exact
    stage-1/2 unweighted formulas (refit_wdown's direct solve, solve_row_irls
    with Lambda=None). Passing Lambda threads it into the W_down ridge (via
    refit_wdown_weighted's CG solve instead of refit_wdown's direct solve)
    and into solve_row_irls's masked-recon target for W_up/W_gate. arm="B1D"
    (stage-2.5 only) refits ONLY W_down (mask + Y from the ORIGINAL, never
    retuned, Wu0/Wg0 -- Delta w = 0 for the other two matrices)."""
    n_groups = X_train.shape[0] // g

    if arm == "A0":
        I_test = intermediate(X_test, Gate_fn(linear_up(X_test, Wg0)), Wu0)
        m = metrics(I_test, colnorm0, K, args.test_seqs, args.seqlen,
                    [int(v) for v in args.group_sizes.split(",")], Wd0, Y_test,
                    [1, 2, 4, 8, 16, 32, 64])
        m.update({"wup_drift": 0.0, "wdown_drift": 0.0, "wgate_drift": 0.0,
                  "mask_flip_rates": []})
        if return_weights:
            return m, Wu0, Wg0, Wd0
        return m

    if arm == "B1D":
        I_train = intermediate(X_train, Gate_fn(linear_up(X_train, Wg0)), Wu0)
        _, token_mask_train = compute_group_mask(I_train, colnorm0, K, g)
        I_masked = I_train * token_mask_train
        if Lambda is None:
            Wd = refit_wdown(I_masked, Y_train, Wd0, args.lambda_down)
        else:
            Wd = refit_wdown_weighted(I_masked, Y_train, Wd0, Lambda, args.lambda_down,
                                       args.cg_iters, args.cg_tol)
        I_test = intermediate(X_test, Gate_fn(linear_up(X_test, Wg0)), Wu0)
        m = metrics(I_test, colnorm0, K, args.test_seqs, args.seqlen,
                    [int(v) for v in args.group_sizes.split(",")], Wd, Y_test,
                    [1, 2, 4, 8, 16, 32, 64])
        m.update({"wup_drift": 0.0, "wgate_drift": 0.0,
                  "wdown_drift": ((Wd - Wd0).norm() / Wd0.norm()).item(),
                  "mask_flip_rates": []})
        if return_weights:
            return m, Wu0, Wg0, Wd
        return m

    Wu, Wg_, Wd = Wu0.clone(), Wg0.clone(), Wd0.clone()
    s0 = measure_s0(intermediate(X_train, Gate_fn(linear_up(X_train, Wg0)), Wu0), colnorm0,
                     n_groups, g)
    mask_prev = None
    flip_rates = []

    for k in range(args.outer):
        lam_eff = lam * (k + 1) / args.outer

        I_cur = intermediate(X_train, Gate_fn(linear_up(X_train, Wg_)), Wu)
        group_mask, token_mask = compute_group_mask(I_cur, colnorm0, K, g)
        if mask_prev is not None:
            flip_rates.append((group_mask != mask_prev).float().mean().item())
        mask_prev = group_mask

        # W_down: masked anchored ridge (design B, closed-form unless Lambda-weighted)
        if Lambda is None:
            Wd = refit_wdown(I_cur * token_mask, Y_train, Wd0, args.lambda_down)
        else:
            Wd = refit_wdown_weighted(I_cur * token_mask, Y_train, Wd0, Lambda,
                                       args.lambda_down, args.cg_iters, args.cg_tol)
        empty_cache()

        # W_up: masked-recon + gauge penalty, IRLS+CG
        resid_global = Y_train - chunked_mm(I_cur * token_mask, Wd.T)
        Gate = Gate_fn(linear_up(X_train, Wg_))
        Wu = solve_row_irls(X_train, Gate, None, I_cur, token_mask, resid_global, Wd,
                             colnorm0, Wu0, lam_eff, n_groups, g, args.mu_up,
                             args.irls, args.cg_iters, args.cg_tol, s0, Lambda=Lambda)
        del I_cur, resid_global
        empty_cache()

        if arm == "A2":
            U = linear_up(X_train, Wu)
            I_cur2 = Gate * U
            resid_global2 = Y_train - chunked_mm(I_cur2 * token_mask, Wd.T)
            a0 = linear_up(X_train, Wg_)
            g0 = Gate_fn(a0)
            gprime0 = act_grad(Gate_fn, a0)
            base = U * (g0 - gprime0 * a0)
            coef = gprime0 * U
            del U, a0, g0, gprime0, Gate
            empty_cache()
            Wg_ = solve_row_irls(X_train, coef, base, I_cur2, token_mask, resid_global2,
                                  Wd, colnorm0, Wg0, lam_eff, n_groups, g, args.mu_gate,
                                  args.irls, args.cg_iters, args.cg_tol, s0, Lambda=Lambda)
            del I_cur2, resid_global2, base, coef
            empty_cache()

    I_test = intermediate(X_test, Gate_fn(linear_up(X_test, Wg_)), Wu)
    group_sizes = [int(v) for v in args.group_sizes.split(",")]
    m = metrics(I_test, colnorm0, K, args.test_seqs, args.seqlen, group_sizes, Wd, Y_test,
                [1, 2, 4, 8, 16, 32, 64])
    m.update({
        "wup_drift": ((Wu - Wu0).norm() / Wu0.norm()).item(),
        "wdown_drift": ((Wd - Wd0).norm() / Wd0.norm()).item(),
        "wgate_drift": ((Wg_ - Wg0).norm() / Wg0.norm()).item(),
        "mask_flip_rates": flip_rates,
    })
    if return_weights:
        return m, Wu, Wg_, Wd
    return m


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)

    # ---- test 1: lambda=0, full mask (K=d, sparsity=0) -> all three drifts ~0
    N, h, d, g = 512, 12, 32, 8
    X = torch.randn(N, h)
    Wg0 = torch.randn(d, h) / math.sqrt(h)
    Wu0 = torch.randn(d, h) / math.sqrt(h)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    f = act_fn("silu")
    Gate0 = f(X @ Wg0.T)
    I0 = intermediate(X, Gate0, Wu0)
    Y = I0 @ Wd0.T
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    K = d  # dense: every neuron always selected -> design B degenerates to design A

    class NS:
        pass
    a = NS()
    a.group_sizes = str(g)
    a.test_seqs = 1
    a.seqlen = N
    a.train_seqs = 1
    a.outer = 2
    a.irls = 3
    a.cg_iters = 100
    a.cg_tol = 1e-8
    a.mu_up = 0.03
    a.mu_gate = 0.03
    a.lambda_down = 0.01

    m = run_arm("A2", X, X, f, Wg0, Wu0, Wd0, colnorm0, Y, Y, K, g, 0.0, a, 1)
    assert m["wup_drift"] < 1e-2, f"wup_drift too large at lam=0: {m['wup_drift']}"
    assert m["wdown_drift"] < 1e-2, f"wdown_drift too large at lam=0: {m['wdown_drift']}"
    assert m["wgate_drift"] < 1e-2, f"wgate_drift too large at lam=0: {m['wgate_drift']}"
    print("selftest 1/4 OK (lam=0 drift ~0, all 3 matrices)")

    # ---- test 2: GN linearization first-order accuracy
    torch.manual_seed(1)
    Nb, hb, db = 64, 10, 6
    Xb = torch.randn(Nb, hb)
    Wg_bar = torch.randn(db, hb) / math.sqrt(hb)
    Wu_b = torch.randn(db, hb) / math.sqrt(hb)
    a0 = Xb @ Wg_bar.T
    g0 = f(a0)
    gprime0 = act_grad(f, a0)
    U = Xb @ Wu_b.T
    torch.manual_seed(2)
    direction = torch.randn(db, hb)
    errs = []
    for eps_ in (0.02, 0.01):
        Delta = eps_ * direction
        a_true = Xb @ (Wg_bar + Delta).T
        i_true = f(a_true) * U
        i_approx = (g0 + gprime0 * (Xb @ Delta.T)) * U
        errs.append((i_true - i_approx).norm().item())
    ratio = errs[0] / max(errs[1], 1e-12)
    assert 3.0 < ratio < 5.0, f"GN linearization not first-order accurate: ratio={ratio:.2f} (want ~4)"
    print(f"selftest 2/4 OK (GN linearization error halves-squared: ratio={ratio:.2f} ~ 4)")

    # ---- test 3: W_down masked step satisfies anchor-ridge definition
    torch.manual_seed(3)
    Nc, hc, dc = 256, 10, 20
    Xc = torch.randn(Nc, hc)
    Wg_c = torch.randn(dc, hc) / math.sqrt(hc)
    Wu_c = torch.randn(dc, hc) / math.sqrt(hc)
    Wd_c = torch.randn(hc, dc) / math.sqrt(dc)
    Gc = f(Xc @ Wg_c.T)
    Ic = intermediate(Xc, Gc, Wu_c)
    colnorm_c = Wd_c.pow(2).sum(0).sqrt()
    Yc = Ic @ Wd_c.T + 0.05 * torch.randn(Nc, hc)   # noisy target so refit != identity
    group_mask_c, token_mask_c = compute_group_mask(Ic, colnorm_c, dc // 2, 8)
    Wd_new = refit_wdown(Ic * token_mask_c, Yc, Wd_c, 0.05)
    mse_new = ((Ic * token_mask_c) @ Wd_new.T - Yc).pow(2).mean().item()
    mse_orig = ((Ic * token_mask_c) @ Wd_c.T - Yc).pow(2).mean().item()
    assert mse_new <= mse_orig + 1e-6, (
        f"masked W_down refit did not improve/match anchor point: new={mse_new} orig={mse_orig}")
    print(f"selftest 3/4 OK (masked calib MSE {mse_new:.5f} <= anchor {mse_orig:.5f})")

    # ---- test 4: gauge invariance of xi and the penalty it drives
    torch.manual_seed(4)
    Nd_, Rd_ = 40, 5
    I_blk = torch.randn(Nd_, Rd_).abs()
    colnorm_blk = torch.rand(Rd_) + 0.1
    xi_a = I_blk * colnorm_blk[None, :]
    c = torch.tensor([2.0, 0.5, 3.0, 1.0, 4.0])
    I_blk_scaled = I_blk * (1.0 / c)[None, :]      # e.g. i absorbs 1/c (up-scale gauge)
    colnorm_scaled = colnorm_blk * c               # W_down column absorbs c
    xi_b = I_blk_scaled * colnorm_scaled[None, :]
    assert torch.allclose(xi_a, xi_b, atol=1e-5), "xi not gauge invariant under i<->colnorm rescale"
    gnorm_a = xi_a.pow(2).sum(0).sqrt()
    gnorm_b = xi_b.pow(2).sum(0).sqrt()
    assert torch.allclose(gnorm_a, gnorm_b, atol=1e-5), "group xi-norm (penalty driver) not gauge invariant"
    print("selftest 4/4 OK (xi and its group norm invariant under i<->W_down column rescale)")

    print("selftest ALL OK")


# ------------------------------------------------------- stage-2.5 selftest

def selftest_25():
    """CPU-only checks for the stage-2.5 additions (Lambda-weighted GSLR).
    Covers 3 of the 5 unit tests in the stage-2.5 request; the other 2 (beta=0
    B2==B1 mask identity, B3's D_ell<=epsilon* assertion) are deferred until
    B2/B3 are implemented -- per the request's pre-registered order, B2/B3
    are not attempted unless B1 clears the go/no-go gate on real PPL."""
    torch.manual_seed(10)

    # ---- unit test 1: B1D's Lambda-weighted ridge with Lambda=ones must
    # match the existing (stage-1/2) refit_wdown direct-solve formula --
    # this IS the "existing refit" B1D reduces to at uniform Lambda.
    N, h, d, g = 400, 14, 24, 8
    X = torch.randn(N, h)
    Wg0 = torch.randn(d, h) / math.sqrt(h)
    Wu0 = torch.randn(d, h) / math.sqrt(h)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    f = act_fn("silu")
    I0 = intermediate(X, f(X @ Wg0.T), Wu0)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    K = d // 2
    _, token_mask = compute_group_mask(I0, colnorm0, K, g)
    Y = I0 @ Wd0.T + 0.03 * torch.randn(N, h)
    lam = 0.4
    Wd_direct = refit_wdown(I0 * token_mask, Y, Wd0, lam)
    Wd_weighted_ones = refit_wdown_weighted(I0 * token_mask, Y, Wd0, torch.ones(h), lam,
                                             cg_iters=200, cg_tol=1e-10)
    err = (Wd_direct - Wd_weighted_ones).norm() / Wd_direct.norm().clamp(min=1e-12)
    assert err < 1e-3, f"Lambda=ones weighted ridge should match refit_wdown direct solve: relerr={err:.2e}"
    print(f"selftest25 1/3 OK (B1D Lambda=ones matches refit_wdown direct solve, relerr={err:.2e})")

    # ---- unit test 3: Lambda=I (all-ones vector, explicitly passed, not
    # None) must reproduce the stage-2 lambda=0 A2 path (Lambda=None).
    class NS:
        pass
    a = NS()
    a.group_sizes = str(g)
    a.test_seqs = 1
    a.seqlen = N
    a.train_seqs = 1
    a.outer = 1
    a.irls = 1
    a.cg_iters = 200
    a.cg_tol = 1e-10
    a.mu_up = 3.0
    a.mu_gate = 3.0
    a.lambda_down = 1.0

    torch.manual_seed(11)
    m_none, Wu_none, Wg_none, Wd_none = run_arm(
        "A2", X, X, f, Wg0, Wu0, Wd0, colnorm0, Y, Y, K, g, 0.0, a, 1, return_weights=True)
    torch.manual_seed(11)
    m_lam, Wu_lam, Wg_lam, Wd_lam = run_arm(
        "A2", X, X, f, Wg0, Wu0, Wd0, colnorm0, Y, Y, K, g, 0.0, a, 1,
        return_weights=True, Lambda=torch.ones(h))
    for name, A, B in (("Wu", Wu_none, Wu_lam), ("Wg", Wg_none, Wg_lam), ("Wd", Wd_none, Wd_lam)):
        relerr = (A - B).norm() / A.norm().clamp(min=1e-12)
        assert relerr < 1e-3, f"Lambda=ones A2/lam=0 path diverged from Lambda=None path on {name}: {relerr:.2e}"
    print("selftest25 2/3 OK (Lambda=ones(explicit) B1 path matches stage-2 Lambda=None lam=0 path)")

    # ---- unit test 4: theta=theta0 (same weights/mask on both sides of the
    # comparison) -> D_ell = 0 exactly.
    torch.manual_seed(12)
    Lambda_rand = 1.0 + torch.rand(h) * 3.0
    _, mask_a0 = compute_group_mask(I0, colnorm0, K, g)
    D_same = measure_D(I0, mask_a0, Wd0, I0, mask_a0, Wd0, Lambda_rand)
    assert D_same < 1e-9, f"D_ell should be exactly 0 when theta=theta0: {D_same:.2e}"
    # sanity: a genuinely different Wd must give D_ell > 0
    D_diff = measure_D(I0, mask_a0, Wd0 + 0.1 * torch.randn_like(Wd0), I0, mask_a0, Wd0, Lambda_rand)
    assert D_diff > 1e-6, "D_ell should be >0 for a perturbed arm"
    print(f"selftest25 3/3 OK (D_ell(theta=theta0)={D_same:.2e}~0, D_ell(perturbed)={D_diff:.4f}>0)")

    print("selftest25 ALL OK")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--group_sizes", default="16,64")
    ap.add_argument("--lambdas", default="0,1e-3,1e-2,1e-1,1")
    ap.add_argument("--arms", default="A0,A1,A2")
    ap.add_argument("--train_seqs", type=int, default=32)
    ap.add_argument("--test_seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--outer", type=int, default=2)
    ap.add_argument("--irls", type=int, default=3)
    ap.add_argument("--cg_iters", type=int, default=25)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    # NOTE: these are much stronger than the flocking PoC's design-A defaults
    # (0.03 / 0.03 / 0.01). Design B's masked-recon target is far more
    # data/mask-specific than dense recon; at the PoC's anchor strength the
    # real-run held-out masked-recon error came out WORSE than the untuned
    # baseline (overfits the calibration mask pattern -- see stage1 report).
    # A synthetic held-out sweep put the crossover around mu~1-3, ld~0.3-1;
    # these are a conservative pick from that range, not independently tuned
    # on real LLaMA data. Still CLI-overridable for future sweeps.
    ap.add_argument("--mu_up", type=float, default=3.0)
    ap.add_argument("--mu_gate", type=float, default=3.0)
    ap.add_argument("--lambda_down", type=float, default=1.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, required=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest25", action="store_true")
    args = ap.parse_args()

    if args.selftest25:
        selftest_25()
        return
    if args.selftest:
        selftest()
        return
    assert args.out, "--out is required"
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    group_sizes = [int(v) for v in args.group_sizes.split(",")]
    lambdas = [float(v) for v in args.lambdas.split(",")]
    arms = args.arms.split(",")

    log("collecting activations...")
    X_train, X_test, Wg, Wu0, Wd0, act_name = collect_activations(args)
    d = Wu0.shape[0]
    K = int(math.floor((1 - args.sparsity) * d))
    log(f"d={d} K={K} act={act_name} train {tuple(X_train.shape)} test {tuple(X_test.shape)}")

    X_train = X_train.to(dev).float()
    X_test = X_test.to(dev).float()
    Wg, Wu0, Wd0 = Wg.to(dev), Wu0.to(dev), Wd0.to(dev)
    f = act_fn(act_name)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()

    I0_train = intermediate(X_train, f(linear_up(X_train, Wg)), Wu0)
    I0_test = intermediate(X_test, f(linear_up(X_test, Wg)), Wu0)
    Y_train = chunked_mm(I0_train, Wd0.T)
    Y_test = chunked_mm(I0_test, Wd0.T)

    results = {"args": vars(args), "git_commit": git_hash(), "K": K, "d": d,
               "runs": []}

    for arm in arms:
        for g in group_sizes:
            lam_list = [0.0] if arm == "A0" else lambdas
            for lam in lam_list:
                t0 = time.time()
                m = run_arm(arm, X_train, X_test, f, Wg, Wu0, Wd0, colnorm0, Y_train,
                            Y_test, K, g, lam, args, args.train_seqs)
                m.update({"arm": arm, "group_size_trained": g, "lambda_rel": lam,
                          "seconds": round(time.time() - t0, 1)})
                results["runs"].append(m)
                log(f"arm={arm} g={g} lam={lam}: W(g)={m[f'within_group_overlap_g{g}']:.4f} "
                    f"union={m[f'union_tax_g{g}']:.3f} "
                    f"maskrecon={m[f'group_mask_recon_relerr_g{g}']:.4f} "
                    f"dense_err={m['dense_recon_relerr']:.4f} "
                    f"static_frac={m[f'static_mask_frac_g{g}']:.3f} "
                    f"drift(u/g/d)={m['wup_drift']:.4f}/{m['wgate_drift']:.4f}/{m['wdown_drift']:.4f} "
                    f"({m['seconds']}s)")
                with open(os.path.join(args.out, "gslr_results.json"), "w") as fh:
                    json.dump(results, fh, indent=2)
    log("done.")


if __name__ == "__main__":
    main()
