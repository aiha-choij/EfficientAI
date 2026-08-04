"""MGR x refit fusion (oracle-residual-sparsity reopen, 2026-08-04).

MGR deployable form (E1's best SLR arm, r256:k1536) rewritten as a linear
model:  y_hat = W_d (m*r) + B (A x) + h(x)
  r    = u * (g - g_bar)            (mean-gate residual)
  m    = top-K mask on |r| * col_norm (residual score, fixed-s select --
         matches E1's "achieved sparsity as targeted" protocol)
  B A  = rank-`rank` SVD of M = W_d diag(g_bar) W_u
  h(x) = Rres (m_x * x), Rres = M - B A, m_x = per-token top-`input_k` |x|
         (slr_input sparse correction; kept FIXED/original in all arms)

Refit fusion: re-solve the linear blocks against the dense teacher
y* = W_d i over calibration, with the W-anchored ridge (the refit C1 fix --
0-shrinkage priors are a known confound):
  R0 : original (W_d, B)                       [sanity vs E1 6.9417 @ s=0.9]
  R1 : W_d re-solved, B frozen                 [output-side refit only]
  R2 : (W_d, B) jointly re-solved              [master-doc section-5 form:
        theta = {W_tilde_down, U}, V0 = A frozen; phi = [m*r ; A x]]
One joint Gram per (layer, s) serves both R1 and R2 (R1 = Schur sub-solve).
Anti-circularity: mask score always uses ORIGINAL W_d col norms and the
calibration g_bar; refit weights never feed back into selection.

Self-contained: vanilla HF model + monkeypatched LlamaMLP.forward at eval;
g_bar computed in-script (pass 0). Only needs torch/transformers/datasets.
"""

import argparse
import json
import math
import os
import subprocess
import time

import torch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- data

def token_stream(tok, dataset, config, split, n_seqs, seqlen):
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    need = n_seqs * seqlen
    assert ids.numel() >= need, f"{split}: {ids.numel()} < {need}"
    return ids[:need].view(n_seqs, seqlen)


# ---------------------------------------------------------------- factors

@torch.no_grad()
def layer_tensors(mlp):
    Wg = mlp.gate_proj.weight.detach().float()   # d x h
    Wu = mlp.up_proj.weight.detach().float()     # d x h
    Wd = mlp.down_proj.weight.detach().float()   # h x d
    return Wg, Wu, Wd


@torch.no_grad()
def build_slr(Wd, Wu, g_bar, rank):
    """M = Wd diag(g_bar) Wu  [h,h]; returns A [rank,h], B [h,rank],
    Rres = M - B A [h,h] (all fp32, on Wd's device)."""
    M = (Wd * g_bar.unsqueeze(0)) @ Wu
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    B = U[:, :r] * sq.unsqueeze(0)
    A = sq.unsqueeze(1) * Vh[:r, :]
    return A, B, M - B @ A


def sparse_comp(x, Rres, input_k):
    """h(x) = Rres (m_x * x), m_x = per-token top-`input_k` of |x|.
    x [N,h] fp32. input_k=0 -> zeros; input_k>=h -> exact Rres @ x."""
    if input_k <= 0:
        return torch.zeros_like(x)
    if input_k >= x.shape[-1]:
        return x @ Rres.T
    idx = x.abs().topk(input_k, dim=-1).indices
    xs = torch.zeros_like(x).scatter_(-1, idx, x.gather(-1, idx))
    return xs @ Rres.T


def topk_mask(score, K):
    idx = score.topk(K, dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)


def solve_anchored(G, Ct, Theta0, lam):
    """Theta = (Ct + lam*D*Theta0)(G + lam*D*I)^-1, D = mean(diag(G)).
    G [p,p], Ct [h,p], Theta0 [h,p]. The refit C1 anchored ridge, verbatim."""
    p = G.shape[0]
    D = torch.diagonal(G).mean()
    reg = G + lam * D * torch.eye(p, dtype=G.dtype, device=G.device)
    rhs = Ct + lam * D * Theta0
    L = torch.linalg.cholesky(reg)
    return torch.cholesky_solve(rhs.T.contiguous(), L).T.contiguous()


def r1_from_joint(G, Ct, Wd0, B0, lam):
    """R1 (W_d only, B frozen at B0) from the JOINT Gram/corr:
    target residual t - B0(Ax); normal eqs use G blocks."""
    d = Wd0.shape[1]
    C1 = Ct[:, :d] - B0 @ G[d:, :d]
    return solve_anchored(G[:d, :d], C1, Wd0, lam)


# ---------------------------------------------------------------- phases

@torch.no_grad()
def pass0_gbar(model, ids, act, dev, bs=1):
    """Mean gate per layer over calibration."""
    layers = model.model.layers
    sums = [None] * len(layers)
    cnt = 0
    hooks = []

    def mk(li, mlp):
        def h(mod, inp):
            x = inp[0].detach().float().squeeze(0)
            g = act(x @ mod.gate_proj.weight.detach().float().T)
            sums[li] = g.sum(0) if sums[li] is None else sums[li] + g.sum(0)
        return h

    for li, layer in enumerate(layers):
        hooks.append(layer.mlp.register_forward_pre_hook(mk(li, layer.mlp)))
    for i in range(ids.shape[0]):
        model(ids[i:i + 1].to(dev))
        cnt += ids.shape[1]
    for h in hooks:
        h.remove()
    return [s / cnt for s in sums]


@torch.no_grad()
def build_grams(model, ids, act, dev, g_bars, factors, s_list, layer_slice):
    """One calibration sweep accumulating, for each layer in layer_slice and
    each s: G = sum(phi phi^T) [(d+r),(d+r)], Ct = sum(t phi^T) [h,(d+r)],
    phi = [m*r ; A x], t = y* - h(x). Returns {(li,s): (G,Ct)}."""
    layers = model.model.layers
    acc = {}
    hooks = []

    def mk(li):
        mlp = layers[li].mlp
        Wg, Wu, Wd = layer_tensors(mlp)
        gb = g_bars[li]
        A, B, Rres = [t.to(Wd.device) for t in factors[li]]  # factors live on CPU
        cn = Wd.pow(2).sum(0).sqrt()          # original col norms (gauge)
        d = Wd.shape[1]

        def h(mod, inp):
            x = inp[0].detach().float().squeeze(0)          # N x h
            g = act(x @ Wg.T)
            u = x @ Wu.T
            r = u * (g - gb)
            ystar = (u * g) @ Wd.T                          # dense teacher
            t = ystar - sparse_comp(x, Rres, ARGS.input_k)
            ax = x @ A.T                                    # N x rank
            score = r.abs() * cn
            del g, u, ystar
            for s in s_list:
                K = int(round((1 - s) * d))
                m = topk_mask(score, K)
                phi = torch.cat([r * m, ax], dim=-1)        # N x (d+rank)
                key = (li, s)
                if key not in acc:
                    p = phi.shape[-1]
                    acc[key] = (torch.zeros(p, p, dtype=torch.float32, device=phi.device),
                                torch.zeros(t.shape[-1], p, dtype=torch.float32, device=phi.device))
                G, Ct = acc[key]
                G.add_(phi.T @ phi)                          # in-place: no double buffer
                Ct.add_(t.T @ phi)
                del phi, m
        return h

    for li in layer_slice:
        hooks.append(layers[li].mlp.register_forward_pre_hook(
            lambda mod, inp, _h=mk(li): _h(mod, inp)))
    for i in range(ids.shape[0]):
        model(ids[i:i + 1].to(dev))
    for h in hooks:
        h.remove()
    return acc


def patched_forward_factory(mlp, gb, A, B, Rres, cn, K, Wd_use, B_use, input_k, act, sp_log):
    def fwd(x_in):
        x = x_in.detach().float().squeeze(0) if x_in.dim() == 3 else x_in.float()
        g = act(x @ mlp._fus_Wg.T)
        u = x @ mlp._fus_Wu.T
        r = u * (g - gb)
        m = topk_mask(r.abs() * cn, K)
        sp_log.append(1.0 - m.float().mean().item())
        y = (r * m) @ Wd_use.T + (x @ A.T) @ B_use.T + sparse_comp(x, Rres, input_k)
        return y.to(x_in.dtype).view_as(x_in) if x_in.dim() == 3 else y.to(x_in.dtype)
    return fwd


@torch.no_grad()
def eval_ppl(model, ids, dev):
    nll, ntok = 0.0, 0
    for i in range(ids.shape[0]):
        batch = ids[i:i + 1].to(dev)
        out = model(batch, labels=batch)
        nll += out.loss.float().item() * (batch.shape[1] - 1)
        ntok += batch.shape[1] - 1
    return math.exp(nll / ntok)


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)
    h, d, r, N = 12, 32, 4, 256
    Wg, Wu = torch.randn(d, h) / math.sqrt(h), torch.randn(d, h) / math.sqrt(h)
    Wd = torch.randn(h, d) / math.sqrt(d)
    X = torch.randn(N, h)
    act = torch.nn.functional.silu
    g = act(X @ Wg.T)
    gb = g.mean(0)
    A, B, Rres = build_slr(Wd, Wu, gb, r)
    # 1) full sparse correction => B A x + Rres x == M x exactly
    M = (Wd * gb.unsqueeze(0)) @ Wu
    comp = (X @ A.T) @ B.T + sparse_comp(X, Rres, h)
    assert (comp - X @ M.T).norm() / (X @ M.T).norm() < 1e-4, "identity M split"
    # 2) joint anchored solve at huge lam recovers originals (R2 ~ R0)
    u = X @ Wu.T
    rr = u * (g - gb)
    m = topk_mask(rr.abs() * Wd.pow(2).sum(0).sqrt(), d // 4)
    phi = torch.cat([rr * m, X @ A.T], -1)
    t = (u * g) @ Wd.T - sparse_comp(X, Rres, 2)
    G, Ct = phi.T @ phi, t.T @ phi
    Th0 = torch.cat([Wd, B], -1)
    Th = solve_anchored(G, Ct, Th0, 1e6)
    assert (Th - Th0).norm() / Th0.norm() < 1e-3, "anchor recovery"
    # 3) lam=0.01 joint solve strictly reduces in-sample error vs originals
    e0 = (t - phi @ Th0.T).norm()
    Th2 = solve_anchored(G, Ct, Th0, 0.01)
    e2 = (t - phi @ Th2.T).norm()
    assert e2 < e0, f"in-sample error must drop ({e2:.4f} vs {e0:.4f})"
    # 4) R1 sub-solve == direct W_d-only anchored solve
    W1 = r1_from_joint(G, Ct, Wd, B, 0.01)
    d_ = d
    C1 = (t - (X @ A.T) @ B.T).T @ phi[:, :d_]
    W1d = solve_anchored(G[:d_, :d_], C1, Wd, 0.01)
    assert (W1 - W1d).norm() / W1d.norm() < 1e-4, "R1 Schur sub-solve"
    print("selftest OK")


# ---------------------------------------------------------------- main

ARGS = None


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="/raid/LLM/llama2-7b")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--input_k", type=int, default=1536)
    ap.add_argument("--s_list", default="0.5,0.7,0.9")
    ap.add_argument("--lambdas", default="0.1")
    ap.add_argument("--calib_seqs", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--layers_per_pass", type=int, default=4)
    ap.add_argument("--arms", default="dense,r0,r1,r2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ARGS = args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert args.out
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    s_list = [float(v) for v in args.s_list.split(",")]
    lambdas = [float(v) for v in args.lambdas.split(",")]
    arms = args.arms.split(",")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    calib = token_stream(tok, "wikitext", "wikitext-103-raw-v1", "train",
                         args.calib_seqs, args.seqlen)
    test = token_stream(tok, "wikitext", "wikitext-2-raw-v1", "test",
                        166, args.seqlen)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(dev).eval()
    act = torch.nn.functional.silu
    layers = model.model.layers
    L = len(layers)

    log("pass 0: g_bar...")
    g_bars = pass0_gbar(model, calib, act, dev)
    log("factors (SVD per layer)...")
    factors = {}
    with torch.no_grad():
        for li, layer in enumerate(layers):
            Wg, Wu, Wd = layer_tensors(layer.mlp)
            factors[li] = tuple(t.cpu() for t in build_slr(Wd, Wu, g_bars[li], args.rank))
            torch.cuda.empty_cache()

    solved = {}   # (li, s, lam, arm) -> (Wd_use, B_use)
    log("build: joint Grams + anchored solves...")
    for start in range(0, L, args.layers_per_pass):
        sl = list(range(start, min(start + args.layers_per_pass, L)))
        acc = build_grams(model, calib, act, dev, g_bars, factors, s_list, sl)
        for (li, s), (G, Ct) in acc.items():
            _, _, Wd = layer_tensors(layers[li].mlp)
            A, B, Rres = [t.to(G.device) for t in factors[li]]
            Th0 = torch.cat([Wd, B], -1)
            d = Wd.shape[1]
            for lam in lambdas:
                Th = solve_anchored(G, Ct, Th0, lam)
                # solved weights live on CPU (32 layers x 3 s x 2 lam x ~184MB
                # would exceed any GPU); moved back per-layer at eval time
                solved[(li, s, lam, "r2")] = (Th[:, :d].cpu(), Th[:, d:].cpu())
                solved[(li, s, lam, "r1")] = (r1_from_joint(G, Ct, Wd, B, lam).cpu(), B.cpu())
        del acc
        torch.cuda.empty_cache()
        log(f"  layers {sl[0]}-{sl[-1]} solved")

    results = {"args": vars(args), "git_commit": git_hash(), "runs": []}
    orig_fwd = [layer.mlp.forward for layer in layers]

    def set_arm(arm, s, lam):
        torch.cuda.empty_cache()
        sp_log = []
        for li, layer in enumerate(layers):
            mlp = layer.mlp
            Wg, Wu, Wd = layer_tensors(mlp)
            mlp._fus_Wg, mlp._fus_Wu = Wg, Wu
            A, B, Rres = [t.to(Wd.device) for t in factors[li]]
            cn = Wd.pow(2).sum(0).sqrt()
            K = int(round((1 - s) * Wd.shape[1]))
            Wd_use, B_use = (Wd, B) if arm == "r0" else tuple(
                t.to(Wd.device) for t in solved[(li, s, lam, arm)])
            mlp.forward = patched_forward_factory(
                mlp, g_bars[li], A, B, Rres, cn, K, Wd_use, B_use,
                args.input_k, act, sp_log)
        return sp_log

    if "dense" in arms:
        ppl = eval_ppl(model, test, dev)
        results["runs"].append({"arm": "dense", "ppl": ppl})
        log(f"dense: PPL {ppl:.4f}")

    for s in s_list:
        for arm in [a for a in arms if a != "dense"]:
            for lam in (lambdas if arm in ("r1", "r2") else [None]):
                sp_log = set_arm(arm, s, lam)
                ppl = eval_ppl(model, test, dev)
                for layer, f in zip(layers, orig_fwd):
                    layer.mlp.forward = f
                rec = {"arm": arm, "s": s, "lam": lam, "ppl": ppl,
                       "achieved_sparsity": sum(sp_log) / max(len(sp_log), 1)}
                results["runs"].append(rec)
                log(f"{arm} s={s} lam={lam}: PPL {ppl:.4f} (sp {rec['achieved_sparsity']:.4f})")
                with open(os.path.join(args.out, "fusion_results.json"), "w") as f:
                    json.dump(results, f, indent=2)
    log("done.")


if __name__ == "__main__":
    main()
