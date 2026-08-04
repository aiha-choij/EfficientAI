"""MGR x refit fusion + amplification arms (oracle-residual-sparsity reopen).

Round 2 (2026-08-04) established: anchored refit of (W_d, B) on the
deployable SLR arm gives 6.780 -> 6.195 @ s=0.9 (LLaMA2-7B), passing all
three thresholds (SLR 6.94 / TIS@0.85 6.709 / exact C3 6.638) — but the
gain is dominated by the W_d block. Round 3 arms attack the three
structural limits identified in that round:

  r2      : [W_d, B] joint anchored refit, features [m*r ; A x],
            target y* - h(x). (Round-2 reference; sub-solve of famA.)
  r4      : + sketch-tail feature block (1-m)*(ghat*uhat) with its own
            learned output map anchored at W_d — imports C8's token-wise
            gate estimate INTO the closed form ("refit x C8 fusion").
            Attacks: refit cannot see token-idiosyncratic gate deviation.
  r3full  : regression-FIRST full linear compensation: solve [W_d, T],
            T [h,h], features [m*r ; x], target y* (T absorbs low-rank +
            sparse + anything linear). DIAGNOSTIC ceiling for any
            linear-in-x compensation. Attacks: A frozen at SVD-of-M basis.
  r3trunc : deployable projection of r3full: T -> SVD rank-`rank` (B3 A3)
            + sparse residual R3 = T - B3 A3 on top-`input_k` |x| channels
            (same runtime cost/structure as SLR).
  r5      : fair frontier CONTROL — plain-magnitude top-K mask (TIS
            protocol, score |i|, no compensation) + anchored W_d refit,
            at s in --r5_s. The E-W0 frontier verdict compared unrefit
            TIS to compensated arms; refit is free for both sides.

Shared discipline: masks from ORIGINAL weights (anti-circularity),
W-anchored ridge everywhere (refit C1 fix), all Grams accumulated in-place
on GPU, factors/solved weights CPU-resident outside their use site.
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
def svd_lr(W, rank):
    """rank-r truncated SVD of W: returns A [r, in], B [out, r]."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    return sq.unsqueeze(1) * Vh[:r, :], U[:, :r] * sq.unsqueeze(0)


@torch.no_grad()
def build_slr(Wd, Wu, g_bar, rank):
    """M = Wd diag(g_bar) Wu [h,h]; returns A, B, Rres = M - B A, M."""
    M = (Wd * g_bar.unsqueeze(0)) @ Wu
    A, B = svd_lr(M, rank)
    return A, B, M - B @ A, M


def sparse_comp(x, Rres, input_k):
    """h(x) = Rres (m_x * x), m_x = per-token top-`input_k` of |x|."""
    if input_k <= 0:
        return torch.zeros(x.shape[0], Rres.shape[0], dtype=x.dtype, device=x.device)
    if input_k >= x.shape[-1]:
        return x @ Rres.T
    idx = x.abs().topk(input_k, dim=-1).indices
    xs = torch.zeros_like(x).scatter_(-1, idx, x.gather(-1, idx))
    return xs @ Rres.T


def topk_mask(score, K):
    idx = score.topk(K, dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)


def block_topk_mask(score, K, g):
    """Token-block-shared top-K: sum the score over g consecutive tokens
    (one 2048-token sequence at a time upstream, so blocks never cross a
    sequence boundary), pick top-K once per block, broadcast to tokens.
    g=1 reduces to topk_mask exactly."""
    if g == 1:
        return topk_mask(score, K)
    N, D = score.shape
    assert N % g == 0, f"seqlen {N} not divisible by block size {g}"
    agg = score.view(N // g, g, D).sum(1)
    return topk_mask(agg, K).repeat_interleave(g, 0)


def solve_anchored(G, Ct, Theta0, lam):
    """Theta = (Ct + lam*D*Theta0)(G + lam*D*I)^-1, D = mean(diag(G))."""
    p = G.shape[0]
    D = torch.diagonal(G).mean()
    reg = G + lam * D * torch.eye(p, dtype=G.dtype, device=G.device)
    rhs = Ct + lam * D * Theta0
    L = torch.linalg.cholesky(reg)
    return torch.cholesky_solve(rhs.T.contiguous(), L).T.contiguous()


def sub_solve(G, Ct, p_sub, Theta0_sub, lam):
    """Anchored solve restricted to the first p_sub feature dims of a
    larger joint Gram (features are a prefix-block: exact, not approx)."""
    return solve_anchored(G[:p_sub, :p_sub], Ct[:, :p_sub], Theta0_sub, lam)


# ---------------------------------------------------------------- phases

@torch.no_grad()
def pass0_gbar(model, ids, act, dev):
    layers = model.model.layers
    sums = [None] * len(layers)
    cnt = 0
    hooks = []

    def mk(li):
        def h(mod, inp):
            x = inp[0].detach().float().squeeze(0)
            g = act(x @ mod.gate_proj.weight.detach().float().T)
            sums[li] = g.sum(0) if sums[li] is None else sums[li] + g.sum(0)
        return h

    for li, layer in enumerate(layers):
        hooks.append(layer.mlp.register_forward_pre_hook(
            lambda mod, inp, _h=mk(li): _h(mod, inp)))
    for i in range(ids.shape[0]):
        model(ids[i:i + 1].to(dev))
        cnt += ids.shape[1]
    for h in hooks:
        h.remove()
    return [s / cnt for s in sums]


def _alloc(acc, key, p, h, dev):
    if key not in acc:
        acc[key] = (torch.zeros(p, p, dtype=torch.float32, device=dev),
                    torch.zeros(h, p, dtype=torch.float32, device=dev))
    return acc[key]


@torch.no_grad()
def build_grams(model, ids, act, dev, ctx, s_list, r5_s, fams, layer_slice, input_k, rsks, gsz):
    """One calibration sweep over `layer_slice`, accumulating per (li, s):
    famA: phi = [m*r ; A x ; (1-m)*(ghat*uhat)], t = y* - h(x)
    famB: phi = [m*r ; x],                        t = y*
    famC: phi = m_i * i (plain |i| top-K, s in r5_s), t = y*
    In-place accumulation; Grams live on GPU only for this slice."""
    layers = model.model.layers
    acc = {}
    hooks = []

    def mk(li):
        mlp = layers[li].mlp
        Wg, Wu, Wd = layer_tensors(mlp)
        gb = ctx["g_bars"][li]
        A, B, Rres, _ = [t.to(Wd.device) for t in ctx["slr"][li]]
        cn = Wd.pow(2).sum(0).sqrt()
        d = Wd.shape[1]
        hdim = Wd.shape[0]
        sk_dev = {rsk: [t.to(Wd.device) for t in ctx["sk"][(li, rsk)]]
                  for rsk in (rsks if "A" in fams else [])}

        def h(mod, inp):
            x = inp[0].detach().float().squeeze(0)
            g = act(x @ Wg.T)
            u = x @ Wu.T
            i_full = u * g
            r = u * (g - gb)
            ystar = i_full @ Wd.T
            score = r.abs() * cn
            if "A" in fams or "B" in fams:
                for s in s_list:
                    m = block_topk_mask(score, int(round((1 - s) * d)), gsz)
                    mr = r * m
                    if "A" in fams:
                        t = ystar - sparse_comp(x, Rres, input_k)
                        for rsk in rsks:
                            Ag, Bg, Au, Bu = sk_dev[rsk]
                            ghat = act((x @ Ag.T) @ Bg.T)
                            uhat = (x @ Au.T) @ Bu.T
                            tail = (ghat * uhat).mul_((~m).float())
                            phi = torch.cat([mr, x @ A.T, tail], dim=-1)
                            G, Ct = _alloc(acc, ("A", li, s, rsk), phi.shape[-1], hdim, x.device)
                            G.add_(phi.T @ phi)
                            Ct.add_(t.T @ phi)
                            del phi, tail, ghat, uhat
                        del t
                    if "B" in fams:
                        phi = torch.cat([mr, x], dim=-1)
                        G, Ct = _alloc(acc, ("B", li, s, 0), phi.shape[-1], hdim, x.device)
                        G.add_(phi.T @ phi)
                        Ct.add_(ystar.T @ phi)
                        del phi
                    del m, mr
            if "C" in fams:
                score_i = i_full.abs()          # TIS protocol: plain |i|
                for s in r5_s:
                    m = block_topk_mask(score_i, int(round((1 - s) * d)), gsz)
                    phi = i_full * m
                    G, Ct = _alloc(acc, ("C", li, s, 0), d, hdim, x.device)
                    G.add_(phi.T @ phi)
                    Ct.add_(ystar.T @ phi)
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
    h, d, r, N = 12, 32, 4, 512
    Wg, Wu = torch.randn(d, h) / math.sqrt(h), torch.randn(d, h) / math.sqrt(h)
    Wd = torch.randn(h, d) / math.sqrt(d)
    X = torch.randn(N, h)
    act = torch.nn.functional.silu
    g = act(X @ Wg.T)
    gb = g.mean(0)
    A, B, Rres, M = build_slr(Wd, Wu, gb, r)
    # 1) full sparse correction => B A x + Rres x == M x exactly
    comp = (X @ A.T) @ B.T + sparse_comp(X, Rres, h)
    assert (comp - X @ M.T).norm() / (X @ M.T).norm() < 1e-4, "M split identity"
    u = X @ Wu.T
    rr = u * (g - gb)
    i_full = u * g
    m = topk_mask(rr.abs() * Wd.pow(2).sum(0).sqrt(), d // 4)
    ystar = i_full @ Wd.T
    # 2) famA sub-block == direct r2 solve
    Agk, Bgk = svd_lr(Wg, d)   # full-rank sketch
    Auk, Buk = svd_lr(Wu, d)
    ghat = act((X @ Agk.T) @ Bgk.T)
    uhat = (X @ Auk.T) @ Buk.T
    tail = (ghat * uhat) * (~m).float()
    phiA = torch.cat([rr * m, X @ A.T, tail], -1)
    tA = ystar - sparse_comp(X, Rres, 2)
    GA, CtA = phiA.T @ phiA, tA.T @ phiA
    Th0_2 = torch.cat([Wd, B], -1)
    direct2 = solve_anchored(torch.cat([rr * m, X @ A.T], -1).T @ torch.cat([rr * m, X @ A.T], -1),
                             tA.T @ torch.cat([rr * m, X @ A.T], -1), Th0_2, 0.01)
    sub2 = sub_solve(GA, CtA, d + r, Th0_2, 0.01)
    assert (sub2 - direct2).norm() / direct2.norm() < 1e-4, "famA sub-block == r2"
    # 3) full-rank sketch tail == exact dropped intermediate
    assert (tail - i_full * (~m).float()).norm() / i_full.norm() < 1e-4, "sketch tail exact at full rank"
    # 4) r3trunc at rank=h, input_k=h reproduces r3full outputs
    phiB = torch.cat([rr * m, X], -1)
    ThB = solve_anchored(phiB.T @ phiB, ystar.T @ phiB, torch.cat([Wd, M], -1), 0.01)
    Wd3, T = ThB[:, :d], ThB[:, d:]
    A3, B3 = svd_lr(T, h)
    R3 = T - B3 @ A3
    yfull = (rr * m) @ Wd3.T + X @ T.T
    ytrunc = (rr * m) @ Wd3.T + (X @ A3.T) @ B3.T + sparse_comp(X, R3, h)
    assert (yfull - ytrunc).norm() / yfull.norm() < 1e-4, "r3trunc==r3full at full rank"
    # 5) anchored solve reduces in-sample error; huge lam recovers anchors
    e0 = (tA - phiA @ torch.cat([Wd, B, Wd], -1).T).norm()
    ThA = solve_anchored(GA, CtA, torch.cat([Wd, B, Wd], -1), 0.01)
    assert (tA - phiA @ ThA.T).norm() < e0, "in-sample error must drop"
    ThInf = solve_anchored(GA, CtA, torch.cat([Wd, B, Wd], -1), 1e7)
    assert (ThInf - torch.cat([Wd, B, Wd], -1)).norm() / ThInf.norm() < 1e-3, "anchor recovery"
    # 6) block mask: g=1 identity; within-block uniformity; K per block
    sc = torch.rand(64, 16)
    assert (block_topk_mask(sc, 5, 1) == topk_mask(sc, 5)).all(), "g=1 reduction"
    mb = block_topk_mask(sc, 5, 8)
    assert (mb.view(8, 8, 16)[:, 0:1] == mb.view(8, 8, 16)).all(), "block uniformity"
    assert (mb.sum(-1) == 5).all(), "K kept per token under block sharing"
    print("selftest OK")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="/raid/LLM/llama2-7b")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--input_k", type=int, default=1536)
    ap.add_argument("--r_sk", default="0", help="comma list of sketch ranks for r4 (0 -> d//8)")
    ap.add_argument("--group_size", type=int, default=1,
                    help="token-block mask sharing g (1 = per-token)")
    ap.add_argument("--s_list", default="0.7,0.9")
    ap.add_argument("--r5_s", default="0.85,0.9")
    ap.add_argument("--lambdas", default="0.1")
    ap.add_argument("--calib_seqs", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--layers_per_pass", type=int, default=1)
    ap.add_argument("--arms", default="r3full,r3trunc,r4,r5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert args.out
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    s_list = [float(v) for v in args.s_list.split(",")]
    r5_s = [float(v) for v in args.r5_s.split(",")]
    lambdas = [float(v) for v in args.lambdas.split(",")]
    arms = args.arms.split(",")
    fams = set()
    if {"r2", "r4", "r4trunc"} & set(arms):
        fams.add("A")
    if {"r3full", "r3trunc"} & set(arms):
        fams.add("B")
    if "r5" in arms:
        fams.add("C")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    calib = token_stream(tok, "wikitext", "wikitext-103-raw-v1", "train",
                         args.calib_seqs, args.seqlen)
    test = token_stream(tok, "wikitext", "wikitext-2-raw-v1", "test", 166, args.seqlen)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(dev).eval()
    act = torch.nn.functional.silu
    layers = model.model.layers
    L = len(layers)
    d = layers[0].mlp.up_proj.weight.shape[0]
    rsks = [int(v) if int(v) > 0 else d // 8 for v in args.r_sk.split(",")]
    gsz = args.group_size

    log("pass 0: g_bar...")
    g_bars = pass0_gbar(model, calib, act, dev)
    log(f"factors (SLR rank={args.rank}, sketches r_sk={rsks}, g={gsz})...")
    ctx = {"g_bars": g_bars, "slr": {}, "sk": {}}
    with torch.no_grad():
        for li, layer in enumerate(layers):
            Wg, Wu, Wd = layer_tensors(layer.mlp)
            ctx["slr"][li] = tuple(t.cpu() for t in build_slr(Wd, Wu, g_bars[li], args.rank))
            if "A" in fams:
                for rsk in rsks:
                    Ag, Bg = svd_lr(Wg, rsk)
                    Au, Bu = svd_lr(Wu, rsk)
                    ctx["sk"][(li, rsk)] = tuple(t.cpu() for t in (Ag, Bg, Au, Bu))
            torch.cuda.empty_cache()

    solved = {}   # (arm, li, s, lam) -> CPU tensors tuple
    log(f"build: fams={sorted(fams)} lpp={args.layers_per_pass}...")
    for start in range(0, L, args.layers_per_pass):
        sl = list(range(start, min(start + args.layers_per_pass, L)))
        acc = build_grams(model, calib, act, dev, ctx, s_list, r5_s, fams, sl,
                          args.input_k, rsks, gsz)
        for (fam, li, s, rsk), (G, Ct) in acc.items():
            _, _, Wd = layer_tensors(layers[li].mlp)
            for lam in lambdas:
                if fam == "A":
                    A, B, _, _ = [t.to(dev) for t in ctx["slr"][li]]
                    Th0 = torch.cat([Wd, B, Wd], -1)
                    Th = solve_anchored(G, Ct, Th0, lam)
                    Wtail = Th[:, d + args.rank:]
                    solved[("r4", li, s, lam, rsk)] = tuple(
                        t.cpu() for t in (Th[:, :d], Th[:, d:d + args.rank], Wtail))
                    At, Bt = svd_lr(Wtail, rsk)   # deployable truncated tail output
                    solved[("r4trunc", li, s, lam, rsk)] = tuple(
                        t.cpu() for t in (Th[:, :d], Th[:, d:d + args.rank], At, Bt))
                    if rsk == rsks[0]:
                        sub = sub_solve(G, Ct, d + args.rank, torch.cat([Wd, B], -1), lam)
                        solved[("r2", li, s, lam, 0)] = (sub[:, :d].cpu(), sub[:, d:].cpu())
                elif fam == "B":
                    M = ctx["slr"][li][3].to(dev)
                    Th = solve_anchored(G, Ct, torch.cat([Wd, M], -1), lam)
                    Wd3, T = Th[:, :d], Th[:, d:]
                    A3, B3 = svd_lr(T, args.rank)
                    solved[("r3", li, s, lam, 0)] = (Wd3.cpu(), T.cpu(), A3.cpu(),
                                                     B3.cpu(), (T - B3 @ A3).cpu())
                elif fam == "C":
                    W5 = solve_anchored(G, Ct, Wd, lam)
                    solved[("r5", li, s, lam, 0)] = (W5.cpu(),)
        del acc
        torch.cuda.empty_cache()
        log(f"  layers {sl[0]}-{sl[-1]} solved")

    results = {"args": vars(args), "git_commit": git_hash(), "r_sk": rsks,
               "group_size": gsz, "runs": []}
    orig_fwd = [layer.mlp.forward for layer in layers]

    def wrap(mlp, body, sp_log):
        def fwd(x_in):
            x = x_in.detach().float().squeeze(0) if x_in.dim() == 3 else x_in.float()
            y, sp = body(x)
            sp_log.append(sp)
            return y.to(x_in.dtype).view_as(x_in) if x_in.dim() == 3 else y.to(x_in.dtype)
        return fwd

    def set_arm(arm, s, lam, rsk=0):
        torch.cuda.empty_cache()
        sp_log = []
        for li, layer in enumerate(layers):
            mlp = layer.mlp
            gb = g_bars[li]
            A, B, Rres, _ = [t.to(dev) for t in ctx["slr"][li]]
            # col norms without keeping an fp32 W_d copy resident
            cn = mlp.down_proj.weight.detach().float().pow(2).sum(0).sqrt()
            K = int(round((1 - s) * d))
            # gate/up projections reuse the module's own bf16 weights at
            # eval time (round-3 OOM fix: fp32 clones held by 32 closures
            # cost 11.5GB; bf16 module matmul matches deployed numerics)
            def proj_g(x, mlp=mlp):
                return act(mlp.gate_proj(x.to(mlp.gate_proj.weight.dtype)).float())
            def proj_u(x, mlp=mlp):
                return mlp.up_proj(x.to(mlp.up_proj.weight.dtype)).float()

            if arm == "m0":     # mask-only control (residual score, no comp, no refit)
                Wd_orig = mlp.down_proj.weight.detach().float()

                def body(x, pg=proj_g, pu=proj_u, gb=gb, cn=cn, K=K, Wd_orig=Wd_orig):
                    g_ = pg(x)
                    u_ = pu(x)
                    m = block_topk_mask((u_ * (g_ - gb)).abs() * cn, K, gsz)
                    return (u_ * g_ * m) @ Wd_orig.T, 1.0 - m.float().mean().item()
            elif arm == "r5":
                (W5,) = [t.to(dev) for t in solved[("r5", li, s, lam, 0)]]

                def body(x, pg=proj_g, pu=proj_u, W5=W5, K=K):
                    i_full = pu(x) * pg(x)
                    m = block_topk_mask(i_full.abs(), K, gsz)
                    return (i_full * m) @ W5.T, 1.0 - m.float().mean().item()
            elif arm in ("r3full", "r3trunc"):
                Wd3, T, A3, B3, R3 = [t.to(dev) for t in solved[("r3", li, s, lam, 0)]]

                def body(x, pg=proj_g, pu=proj_u, gb=gb, cn=cn, K=K, Wd3=Wd3, T=T,
                         A3=A3, B3=B3, R3=R3, full=(arm == "r3full")):
                    g_ = pg(x)
                    r_ = pu(x) * (g_ - gb)
                    m = block_topk_mask(r_.abs() * cn, K, gsz)
                    base = (r_ * m) @ Wd3.T
                    comp = x @ T.T if full else \
                        (x @ A3.T) @ B3.T + sparse_comp(x, R3, args.input_k)
                    return base + comp, 1.0 - m.float().mean().item()
            else:  # r0 / r2 / r4 / r4trunc
                At_ = Bt_ = None
                if arm == "r0":     # SLR with ORIGINAL weights (no refit)
                    Wdu = mlp.down_proj.weight.detach().float()
                    Bu_ = B
                    Wtail = Ag = Bg = Au = Buu = None
                elif arm == "r2":
                    Wdu, Bu_ = [t.to(dev) for t in solved[("r2", li, s, lam, 0)]]
                    Wtail = Ag = Bg = Au = Buu = None
                elif arm == "r4":
                    Wdu, Bu_, Wtail = [t.to(dev) for t in solved[("r4", li, s, lam, rsk)]]
                    Ag, Bg, Au, Buu = [t.to(dev) for t in ctx["sk"][(li, rsk)]]
                else:               # r4trunc: low-rank tail output (deploy cost)
                    Wdu, Bu_, At_, Bt_ = [t.to(dev) for t in solved[("r4trunc", li, s, lam, rsk)]]
                    Wtail = None
                    Ag, Bg, Au, Buu = [t.to(dev) for t in ctx["sk"][(li, rsk)]]

                def body(x, pg=proj_g, pu=proj_u, gb=gb, cn=cn, K=K, A=A, Rres=Rres,
                         Wdu=Wdu, Bu_=Bu_, Wtail=Wtail, Ag=Ag, Bg=Bg, Au=Au, Buu=Buu,
                         At_=At_, Bt_=Bt_):
                    g_ = pg(x)
                    r_ = pu(x) * (g_ - gb)
                    m = block_topk_mask(r_.abs() * cn, K, gsz)
                    y = (r_ * m) @ Wdu.T + (x @ A.T) @ Bu_.T + sparse_comp(x, Rres, args.input_k)
                    if Wtail is not None or At_ is not None:
                        ghat = act((x @ Ag.T) @ Bg.T)
                        uhat = (x @ Au.T) @ Buu.T
                        tail = (ghat * uhat) * (~m).float()
                        y = y + (tail @ Wtail.T if Wtail is not None
                                 else (tail @ At_.T) @ Bt_.T)
                    return y, 1.0 - m.float().mean().item()

            mlp.forward = wrap(mlp, body, sp_log)
        return sp_log

    for arm in arms:
        ss = r5_s if arm == "r5" else s_list
        arm_rsks = rsks if arm in ("r4", "r4trunc") else [0]
        for s in ss:
            for lam in lambdas:
                for rsk in arm_rsks:
                    sp_log = set_arm(arm, s, lam, rsk)
                    ppl = eval_ppl(model, test, dev)
                    for layer, f in zip(layers, orig_fwd):
                        layer.mlp.forward = f
                    rec = {"arm": arm, "s": s, "lam": lam, "rsk": rsk, "g": gsz,
                           "ppl": ppl,
                           "achieved_sparsity": sum(sp_log) / max(len(sp_log), 1)}
                    results["runs"].append(rec)
                    log(f"{arm} s={s} lam={lam} rsk={rsk} g={gsz}: PPL {ppl:.4f} "
                        f"(sp {rec['achieved_sparsity']:.4f})")
                with open(os.path.join(args.out, "fusion3_results.json"), "w") as f:
                    json.dump(results, f, indent=2)
    log("done.")


if __name__ == "__main__":
    main()
