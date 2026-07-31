"""Single-layer flocking PoC (topic: groupwise-flocking-tuning).

Tunes ONE FFN layer of a frozen LLM with a group x neuron l2,1 (group-lasso)
penalty added to a backprop-free local reconstruction objective, then measures
whether token-group neuron-selection overlap ("flocking") increases.

Method (design resolution 2026-07-31, gist Open Questions 1-2):
- Parameters: W_up (per-neuron-row IRLS, diagonal output-coupling approx)
  alternated with W_down (anchored closed-form ridge refit). W_gate FROZEN;
  Top-K score gauge-fixed to the ORIGINAL W_down column norms (anti-circularity).
- l2,1 handled by reweighted-l2 (IRLS): each iteration is a pure quadratic
  solved by batched per-row conjugate gradient (shared token matrix).
- Both weight solves are anchored to the ORIGINAL weights (W-anchored damping;
  lesson from local-loss-refit's 0-shrinkage-prior confound).
- PoC objective is DENSE reconstruction + penalty (no mask inside the
  objective; masks enter only at evaluation) — design decision A.

Per-row W_up subproblem (row j, fp32):
  min_w  sum_t g_jt^2 (x_t^T w - x_t^T w0)^2                       (recon)
       + P_j * sum_T omega_jT * sum_{t in T} g_jt^2 (x_t^T w)^2    (l2,1 via IRLS)
       + mu_j ||w - w0||^2                                          (anchor)
  P_j = lam_rel * a_med / max(a_j, 0.01*a_med)   (a_j = ||W_down[:,j]||^2;
        cheap-to-cut neurons are shrunk first), omega_jT = s0/(2*max(||i_jT||,eps))
  => (X^T diag(c_j) X + mu_j I) w = X^T(g_j^2 * (X w0)) + mu_j w0,
     c_j(t) = g_jt^2 * (1 + P_j * omega_j(t))

Outputs: JSON with baseline + per-(group_size, lambda) metrics:
  C(delta) containment overlap, within-group mean pairwise overlap, union tax,
  dense / group-masked reconstruction relative error, weight drift.
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


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True).strip()
    except Exception:
        return "unknown"


def chunked_matmul(A, B, chunk=8192):
    """A (N x p) @ B (p x q) accumulated over row chunks of A -> (N x q) is trivial;
    this helper computes A.T @ B (p x q) accumulated in fp32 over token chunks."""
    out = torch.zeros(A.shape[1], B.shape[1], dtype=torch.float32, device=A.device)
    for s in range(0, A.shape[0], chunk):
        out += A[s:s + chunk].T @ B[s:s + chunk]
    return out


# ---------------------------------------------------------------- data

def collect_activations(args):
    """Run the frozen model over train/test token streams; capture the target
    layer's MLP input. Returns X_train, X_test (N x h fp32 CPU), layer weights
    (fp32), act name."""
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
    Wg = layer.mlp.gate_proj.weight.detach().float().cpu()  # d x h
    Wu = layer.mlp.up_proj.weight.detach().float().cpu()    # d x h
    Wd = layer.mlp.down_proj.weight.detach().float().cpu()  # h x d
    act_name = model.config.hidden_act

    captured = []
    hook = layer.mlp.register_forward_pre_hook(
        lambda mod, inp: captured.append(inp[0].detach().squeeze(0).to(torch.float16).cpu()))

    def run(ids):
        captured.clear()
        with torch.no_grad():
            for i in range(ids.shape[0]):
                model(ids[i:i + 1].to(args.device))
        return torch.cat(captured, 0)  # N x h fp16 cpu

    X_train = run(train_ids)
    X_test = run(test_ids)
    hook.remove()
    del model
    torch.cuda.empty_cache() if args.device == "cuda" else None
    return X_train, X_test, Wg, Wu, Wd, act_name


# ---------------------------------------------------------------- core solver

def act_fn(name):
    import torch.nn.functional as F
    return {"silu": F.silu, "gelu": F.gelu}[name]


def intermediate(X, Gate, Wu, chunk=8192):
    """i = gate * (X @ Wu.T), fp32, chunked. X (N x h), Gate (N x d)."""
    out = torch.empty(X.shape[0], Wu.shape[0], dtype=torch.float32, device=X.device)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = Gate[s:s + chunk] * (X[s:s + chunk] @ Wu.T)
    return out


def solve_wup_irls(X, Gate, Wu0, a, lam_rel, n_groups, group_size, mu_rel,
                   irls_iters, cg_iters, cg_tol, row_chunk=1024):
    """Batched per-row IRLS + CG. X (N x h), Gate (N x d) fp32 on device.
    Token order must be group-contiguous (N = n_groups * group_size).
    Returns new W_up (d x h fp32)."""
    N, h = X.shape
    d = Gate.shape[1]
    assert N == n_groups * group_size
    Wnew = Wu0.clone()

    a_med = a.median()
    P = lam_rel * a_med / a.clamp(min=0.01 * a_med)          # (d,)
    xsq = (X * X).sum(1)                                     # (N,)

    # global scale s0: median group-column norm of the ORIGINAL activations
    with torch.no_grad():
        i0_sample = Gate[:, :2048] * (X @ Wu0[:2048].T)
        gn = i0_sample.view(n_groups, group_size, -1).pow(2).sum(1).sqrt()
        s0 = gn.median().item()
    eps = 1e-6 + 1e-3 * s0

    for start in range(0, d, row_chunk):
        J = slice(start, min(start + row_chunk, d))
        R = J.stop - J.start
        gsq = Gate[:, J].pow(2)                              # N x R
        W0J = Wu0[J].T.contiguous()                          # h x R
        muJ = mu_rel * (gsq * xsq[:, None]).sum(0) / h       # (R,)
        rhs0 = X.T @ (gsq * (X @ W0J))                       # h x R (recon pull)
        WJ = W0J.clone()

        # IRLS weights from current iterate's group column norms
        for it in range(irls_iters if lam_rel > 0 else 1):
            iJ = Gate[:, J] * (X @ WJ)                       # N x R
            gnorm = iJ.view(n_groups, group_size, R).pow(2).sum(1).sqrt()  # G x R
            omega = s0 / (2.0 * gnorm.clamp(min=eps))        # G x R
            wtok = omega.repeat_interleave(group_size, 0)    # N x R
            c = gsq * (1.0 + P[J][None, :] * wtok * float(lam_rel > 0))

            b = rhs0 + muJ[None, :] * W0J

            def matvec(V):
                return X.T @ (c * (X @ V)) + muJ[None, :] * V

            # CG (R systems in parallel, per-column scalars)
            r = b - matvec(WJ)
            p = r.clone()
            rs = (r * r).sum(0)
            b2 = (b * b).sum(0).clamp(min=1e-30)
            for _ in range(cg_iters):
                Ap = matvec(p)
                alpha = rs / (p * Ap).sum(0).clamp(min=1e-30)
                WJ += alpha[None, :] * p
                r -= alpha[None, :] * Ap
                rs_new = (r * r).sum(0)
                if (rs_new / b2).max() < cg_tol ** 2:
                    break
                p = r + (rs_new / rs.clamp(min=1e-30))[None, :] * p
                rs = rs_new
        Wnew[J] = WJ.T
    return Wnew


def refit_wdown(I, Y, Wd0, lam, chunk=8192):
    """Anchored ridge: min ||I Wd^T - Y||^2 + lam*D*||Wd - Wd0||^2,
    D = mean(diag(G)). I (N x d), Y (N x h) fp32 on device."""
    G = chunked_matmul(I, I, chunk)                          # d x d
    C = chunked_matmul(I, Y, chunk)                          # d x h
    dscale = lam * G.diagonal().mean()
    A = G + dscale * torch.eye(G.shape[0], device=G.device)
    WdT = torch.linalg.solve(A, C + dscale * Wd0.T)          # d x h
    return WdT.T.contiguous()


# ---------------------------------------------------------------- metrics

def topk_sets(score, K):
    """score (N x d) -> bool mask of per-token Top-K (N x d, uint8-as-bool)."""
    idx = score.topk(K, dim=-1).indices
    S = torch.zeros_like(score, dtype=torch.bool)
    S.scatter_(1, idx, True)
    return S


def metrics(I, colnorm0, K, n_seqs, seqlen, group_sizes, Wd_eval, Y, deltas):
    """All flocking + reconstruction metrics for activation matrix I (N x d)."""
    score = I.abs() * colnorm0[None, :]
    S = topk_sets(score, K)                                  # N x d bool
    Sv = S.view(n_seqs, seqlen, -1)
    out = {}

    out["C_delta"] = {
        str(dl): (Sv[:, :-dl] & Sv[:, dl:]).sum(-1).float().mean().item() / K
        for dl in deltas if dl < seqlen}

    ynorm = Y.pow(2).sum().sqrt()
    out["dense_recon_relerr"] = ((I @ Wd_eval.T - Y).pow(2).sum().sqrt() / ynorm).item()

    for g in group_sizes:
        B = Sv.view(n_seqs, seqlen // g, g, -1)
        pair = torch.einsum("sgtd,sgud->sgtu", B.float(), B.float())
        off = (pair.sum((-1, -2)) - pair.diagonal(dim1=-2, dim2=-1).sum(-1))
        out[f"within_group_overlap_g{g}"] = (off / (g * (g - 1)) / K).mean().item()
        out[f"union_tax_g{g}"] = (B.any(2).sum(-1).float() / K).mean().item()

        # group-shared mask from aggregated score (gauge-fixed colnorm0)
        gscore = score.view(-1, g, score.shape[-1]).sum(1)   # G x d
        gidx = gscore.topk(K, dim=-1).indices
        m = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, gidx, True)
        m_tok = m.repeat_interleave(g, 0)                    # N x d
        out[f"group_mask_recon_relerr_g{g}"] = (
            ((I * m_tok) @ Wd_eval.T - Y).pow(2).sum().sqrt() / ynorm).item()
    return out


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)
    N, h, d, g = 512, 16, 48, 8
    X = torch.randn(N, h)
    Wg0, Wu0 = torch.randn(d, h) / math.sqrt(h), torch.randn(d, h) / math.sqrt(h)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    Gate = torch.sigmoid(X @ Wg0.T)
    a = Wd0.pow(2).sum(0)

    # 1) CG vs direct solve, one row
    j = 3
    gsq = Gate[:, j:j + 1].pow(2)
    mu = 0.03 * (gsq[:, 0] * (X * X).sum(1)).sum() / h
    A = X.T @ (gsq * X) + mu * torch.eye(h)
    b = X.T @ (gsq[:, 0:1] * (X @ Wu0[j:j + 1].T)) + mu * Wu0[j:j + 1].T
    direct = torch.linalg.solve(A, b)
    W = solve_wup_irls(X, Gate, Wu0, a, 0.0, N // g, g, 0.03, 1, 200, 1e-8, row_chunk=d)
    assert (W[j:j + 1].T - direct).norm() / direct.norm() < 1e-3, "CG != direct"

    # 2) lam=0 recovers original weights (anchored, recon pulls to w0)
    assert (W - Wu0).norm() / Wu0.norm() < 1e-3, "lam=0 must recover W_up"

    # 3) large lam shrinks group column norms vs lam=0
    Wl = solve_wup_irls(X, Gate, Wu0, a, 3.0, N // g, g, 0.03, 4, 200, 1e-8, row_chunk=d)
    n0 = intermediate(X, Gate, W).view(N // g, g, d).pow(2).sum(1).sqrt().mean()
    nl = intermediate(X, Gate, Wl).view(N // g, g, d).pow(2).sum(1).sqrt().mean()
    assert nl < 0.8 * n0, f"l2,1 did not shrink group norms ({nl:.4f} vs {n0:.4f})"

    # 4) anchored refit with exact data returns ~Wd0
    I0 = intermediate(X, Gate, Wu0)
    Wd = refit_wdown(I0, I0 @ Wd0.T, Wd0, 0.01)
    assert (Wd - Wd0).norm() / Wd0.norm() < 1e-3, "refit must recover W_down"
    print("selftest OK")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--group_sizes", default="16,64")
    ap.add_argument("--lambdas", default="0,1e-3,1e-2,1e-1,1")
    ap.add_argument("--train_seqs", type=int, default=32)
    ap.add_argument("--test_seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--outer", type=int, default=2)
    ap.add_argument("--irls", type=int, default=3)
    ap.add_argument("--cg_iters", type=int, default=25)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.add_argument("--mu_rel", type=float, default=0.03)
    ap.add_argument("--lambda_down", type=float, default=0.01)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, required=False)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    assert args.out, "--out is required"
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    group_sizes = [int(v) for v in args.group_sizes.split(",")]
    lambdas = [float(v) for v in args.lambdas.split(",")]
    deltas = [1, 2, 4, 8, 16, 32, 64]

    log("collecting activations...")
    X_train, X_test, Wg, Wu0, Wd0, act_name = collect_activations(args)
    d = Wu0.shape[0]
    K = int(math.floor((1 - args.sparsity) * d))
    log(f"d={d} K={K} act={act_name} train {tuple(X_train.shape)} test {tuple(X_test.shape)}")

    X_train = X_train.to(dev).float()
    X_test = X_test.to(dev).float()
    Wg, Wu0, Wd0 = Wg.to(dev), Wu0.to(dev), Wd0.to(dev)
    f = act_fn(act_name)
    Gate_train = f(X_train @ Wg.T)
    Gate_test = f(X_test @ Wg.T)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()                      # gauge-fixed score norms

    I0_train = intermediate(X_train, Gate_train, Wu0)
    I0_test = intermediate(X_test, Gate_test, Wu0)
    Y_train = I0_train @ Wd0.T
    Y_test = I0_test @ Wd0.T

    results = {"args": vars(args), "git_commit": git_hash(), "K": K, "d": d,
               "baseline": metrics(I0_test, colnorm0, K, args.test_seqs, args.seqlen,
                                   group_sizes, Wd0, Y_test, deltas),
               "runs": []}
    log(f"baseline: {json.dumps(results['baseline']['C_delta'])}")

    for g in group_sizes:
        n_groups = X_train.shape[0] // g
        for lam in lambdas:
            t0 = time.time()
            Wu, Wd = Wu0, Wd0
            a = Wd0.pow(2).sum(0)
            for _ in range(args.outer):
                Wu = solve_wup_irls(X_train, Gate_train, Wu0, a, lam, n_groups, g,
                                    args.mu_rel, args.irls, args.cg_iters, args.cg_tol)
                It = intermediate(X_train, Gate_train, Wu)
                Wd = refit_wdown(It, Y_train, Wd0, args.lambda_down)
                a = Wd.pow(2).sum(0)
            I_test = intermediate(X_test, Gate_test, Wu)
            m = metrics(I_test, colnorm0, K, args.test_seqs, args.seqlen,
                        group_sizes, Wd, Y_test, deltas)
            m.update({"group_size_trained": g, "lambda_rel": lam,
                      "wup_drift": ((Wu - Wu0).norm() / Wu0.norm()).item(),
                      "wdown_drift": ((Wd - Wd0).norm() / Wd0.norm()).item(),
                      "seconds": round(time.time() - t0, 1)})
            results["runs"].append(m)
            log(f"g={g} lam={lam}: W(g)={m[f'within_group_overlap_g{g}']:.4f} "
                f"union={m[f'union_tax_g{g}']:.3f} "
                f"dense_err={m['dense_recon_relerr']:.4f} "
                f"drift={m['wup_drift']:.4f} ({m['seconds']}s)")
            with open(os.path.join(args.out, "poc_results.json"), "w") as fh:
                json.dump(results, fh, indent=2)
    log("done.")


if __name__ == "__main__":
    main()
