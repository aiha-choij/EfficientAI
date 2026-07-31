# coding=utf-8
# Block-wise sharing-tax compensation conditions (C7a/C7/C8a/C8) for gated
# MLPs. Continues the oracle intermediate-sparsity C0-C6 numbering
# (oracle_mlp.py); full spec preserved verbatim at
# .labtool/topics/block-sparse-compensation/spec.md.
#
# Setting: prefill processes tokens in GEMM tiles of g consecutive
# same-sequence tokens, so the mask must be block-shared (m_T, one mask per
# block of g tokens) rather than per-token. This module asks how much of
# the resulting "sharing tax" (accuracy lost vs a per-token mask) can be
# recovered by an input-dependent compensation term.
#
#   c7a : y = W_d (m_T*i)                                        [control: block-shared mask, NO compensation]
#   c7  : y = W_d (m_T*i) + comp_lr(x) - W_d (m_T*(g_bar*u))      [block version of oracle C4]
#   c8a : y = W_d (m_T*i) + W_d ((1-m_T)*(ghat*u))                [diagnostic: gate-sketch only, u/W_down exact]
#   c8  : y = W_d (m_T*i) + tail_sketch(x)                        [deployable: gate+up+down all sketched]
#
# where i = u*g, u = up_proj(x), g = act_fn(gate_proj(x)), same as
# oracle_mlp.py. The block mask's SELECTION score is the same residual
# score as oracle C3/C4/C5 (|u*(g-g_bar)| * col_norm) -- NOT the C1/C2
# score |i|*col_norm -- so that c7 at g=1 reduces bit-exactly to oracle C4
# (required unit test 1; see block_p_mask + _resid_score). This resolves an
# ambiguity in the spec's generic S_j(T) notation (written with |i_tj| as
# a placeholder in the shared block-notation section) in favor of the
# literal claim right below the condition table: "C7의 유래: oracle spec
# C4와 동일한 수학... 차이는 m_T가 블록 공유라는 것뿐" -- i.e. c7/c7a/c8/c8a
# all share C4's residual-based selection criterion, generalized to blocks;
# only the mask-sharing granularity changes, not the score.
#
# c8's tail_sketch is computed from the ALREADY block-masked-out tail
# vector (1-m_T)*(ghat*uhat) -- gated to exactly zero wherever nothing is
# masked out -- so c8 (and c8a) satisfy p=1 identity at ANY sketch rank.
# c7 does not share that structure (comp_lr(x) approximates M over ALL
# neurons, then the exact kept-neuron contribution is subtracted out), so
# c7's p=1 identity only holds bit-exactly at full-rank compensation
# (same caveat oracle's own C4 has always had -- see test_oracle_units.py
# test_1, which tests c3/c5 p=1 identity but never c4). Unit test 2 in this
# module's test file therefore runs at rank=full for c7, matching that
# existing precedent (test_oracle_units.py test_2's "set rank=full to
# isolate wiring from approximation error" pattern) rather than inventing
# a new tolerance.
#
# Block aggregation (block_ids/aggregate_block_score) intentionally
# duplicates local-loss-refit's refit_mlp.py logic rather than importing
# it: that module lives only on the unmerged auto/local-loss-refit branch,
# and this topic's branch (auto/block-sparse-compensation) is off main, so
# the two PRs must not depend on each other's unmerged state. If both land,
# a follow-up could factor this out; not done here (out of scope).

import json
import os

import torch

from .oracle_mlp import compute_M, iter_mlps, top_p_mask

CONDITIONS = ("c7a", "c7", "c8a", "c8")


# ---------------------------------------------------------------------------
# block-shared top-p masking (same aggregation discipline as refit_mlp.py:
# fp32, sequence-boundary-respecting -- batch/leading axis never mixed
# across blocks --, padding excluded via seq_mask, ragged last block)
# ---------------------------------------------------------------------------

def block_ids(seqlen, g, device=None):
    """Block index per position, g consecutive positions per block, last
    block ragged if seqlen % g != 0. Shape [seqlen]."""
    return torch.arange(seqlen, device=device) // g


def aggregate_block_score(score, g, seq_mask=None):
    """score: [..., T, D] (fp32, non-negative). Sums score within each block
    of g consecutive positions along dim -2, independently per leading dim
    (batch/sequence axis is never mixed across blocks). seq_mask, if given:
    [..., T] bool, True = real token; padded positions contribute 0 to the
    block sum (excluded from aggregation, per spec).
    Returns (block_score [..., num_blocks, D], bid [T])."""
    *lead, T, D = score.shape
    flat = score.reshape(-1, T, D)
    if seq_mask is not None:
        flat = flat * seq_mask.reshape(-1, T, 1).to(flat.dtype)
    bid = block_ids(T, g, device=score.device)
    num_blocks = int(bid[-1].item()) + 1
    out = torch.zeros(flat.shape[0], num_blocks, D, dtype=flat.dtype, device=flat.device)
    out.index_add_(1, bid, flat)
    return out.reshape(*lead, num_blocks, D), bid


def block_p_mask(score, p, g, seq_mask=None):
    """Block-shared top-p mask (spec section 1: top-p over the block-
    aggregated score S(T)), broadcast back to per-token shape. g=1 reduces
    to top_p_mask(score, p) exactly bitwise (aggregation over a singleton
    block is the identity, so S(T) == score)."""
    block_score, bid = aggregate_block_score(score, g, seq_mask=seq_mask)
    m_block = top_p_mask(block_score, p)
    return m_block.index_select(-2, bid)


def _resid_score(mlp, u, g):
    """Same per-token score as oracle C3/C4/C5: |u*(g-g_bar)| * col_norm.
    Requires mlp.oracle_g_bar / mlp.oracle_col_norm already attached (via
    oracle_mlp.finalize_stats / attach_col_norms -- run the usual oracle
    dense-calibration pass first, same precondition as oracle C3/C4/C5)."""
    resid = u.float() * (g.float() - mlp.oracle_g_bar)
    return resid.abs() * mlp.oracle_col_norm


# ---------------------------------------------------------------------------
# weight sketches for C8/C8a (rank-r SVD of gate_proj/up_proj/down_proj)
# ---------------------------------------------------------------------------

def _svd_factors(W, rank):
    """W: [out, in] (nn.Linear weight layout). Returns A [r,in], B [out,r]
    (fp32, on W's original device) with B @ A ~ W (rank-r truncated SVD);
    exact at rank=full.

    Runs on W's own device (GPU in real usage). An earlier version of
    this function moved the SVD to CPU, misdiagnosing the root cause of
    Phase 3's first OOM (see block-sparse-compensation journal, round 1)
    as "GPU SVD of a big matrix is just too memory-hungry." The REAL
    cause: this function is called with the model's weight tensors,
    which have requires_grad=True by default (`.eval()` does not disable
    it) -- every layer's SVD therefore built and RETAINED an autograd
    graph that nothing ever freed (no backward ever runs), so GPU memory
    grew monotonically across a whole model's layers regardless of which
    device the SVD ran on. oracle_mlp.py's own use of the same style of
    SVD (build_M_factors) never hit this because every caller wraps it in
    `torch.no_grad()`; this module's caller (`attach_block_factors_inplace`,
    below) didn't. The CPU-SVD workaround "fixed" the OOM by trading it
    for CPU compute -- but CPU SVD of a full [d,h] matrix (e.g. 14336x4096
    at 8B) turned out to be slow enough, and multiple concurrent jobs
    contending for the same host's CPU cores made it slower still, that
    Phase 3's 8B sweep stalled for 40+ minutes producing no eval output
    (caught during a periodic check, not a crash). Fixed properly:
    `attach_block_factors_inplace` now runs its whole body under
    `torch.no_grad()`, so the SVD is both fast (GPU) and memory-safe
    (nothing retained)."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    B = U[:, :r] * sq.unsqueeze(0)
    A = sq.unsqueeze(1) * Vh[:r, :]
    return A, B


def build_gate_up_down_sketch(mlp, r_sk):
    """Single-knob r_sk = r_g = r_u = r_d (spec default). Returns
    (Ag,Bg,Au,Bu,Ad,Bd), all fp32."""
    Ag, Bg = _svd_factors(mlp.gate_proj.weight, r_sk)
    Au, Bu = _svd_factors(mlp.up_proj.weight, r_sk)
    Ad, Bd = _svd_factors(mlp.down_proj.weight, r_sk)
    return Ag, Bg, Au, Bu, Ad, Bd


def _build_comp_lr_factors(mlp, rank):
    """C7's comp_lr factors: same math as oracle_mlp.build_M_factors's
    plain (non-whitened) path -- M = W_down diag(g_bar) W_up, rank-r SVD.

    oracle_mlp.py itself is intentionally left untouched (shared with the
    paused oracle-residual-sparsity topic), so this is a local
    reimplementation rather than a call to oracle_mlp.build_M_factors.
    Runs on GPU (M's device) under the caller's `torch.no_grad()` --
    see _svd_factors's docstring for why that (not the device) is what
    actually matters for avoiding the memory-growth bug this module hit
    twice before finding the real cause."""
    M = compute_M(mlp)  # [h,h], computed on GPU
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    B = U[:, :r] * sq.unsqueeze(0)
    A = sq.unsqueeze(1) * Vh[:r, :]
    return A, B


@torch.no_grad()
def attach_block_factors_inplace(model, rank, r_sk, condition=None):
    """Build and attach C7's comp_lr factors (rank) and/or C8/C8a's
    gate/up/down sketches (r_sk) directly (used by tests; a save/load
    pair mirrors oracle_mlp.py's convention for the real build scripts).

    condition, if given, skips whichever half a condition never uses (c7
    only needs comp_lr; c8/c8a only need the sketch) -- real eval runs
    should always pass it, since building both unconditionally wastes a
    full gate/up/down SVD sweep every time (each condition only reads one
    half at forward time). Default None (tests) builds both, unchanged
    from the original behavior.

    @torch.no_grad() is load-bearing, not decoration: model weights have
    requires_grad=True by default (.eval() doesn't change that), so
    without it every layer's SVD retains an autograd graph that never
    gets freed (no backward ever runs) -- this is the actual mechanism
    behind the memory-growth bugs this module hit twice before (see
    _svd_factors's docstring), not the choice of GPU vs CPU."""
    need_comp = condition in (None, "c7")
    need_sketch = condition in (None, "c8", "c8a")
    for _, mlp in iter_mlps(model):
        dtype = mlp.down_proj.weight.dtype
        if need_comp:
            A, B = _build_comp_lr_factors(mlp, rank)
            mlp.blk_comp_A, mlp.blk_comp_B = A.to(dtype), B.to(dtype)
        if need_sketch:
            Ag, Bg, Au, Bu, Ad, Bd = build_gate_up_down_sketch(mlp, r_sk)
            mlp.blk_sketch_Ag, mlp.blk_sketch_Bg = Ag.to(dtype), Bg.to(dtype)
            mlp.blk_sketch_Au, mlp.blk_sketch_Bu = Au.to(dtype), Bu.to(dtype)
            mlp.blk_sketch_Ad, mlp.blk_sketch_Bd = Ad.to(dtype), Bd.to(dtype)


def save_block_factors(model, rank, r_sk, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, mlp in iter_mlps(model):
        payload = {
            "comp_A": mlp.blk_comp_A.cpu(), "comp_B": mlp.blk_comp_B.cpu(),
            "sketch_Ag": mlp.blk_sketch_Ag.cpu(), "sketch_Bg": mlp.blk_sketch_Bg.cpu(),
            "sketch_Au": mlp.blk_sketch_Au.cpu(), "sketch_Bu": mlp.blk_sketch_Bu.cpu(),
            "sketch_Ad": mlp.blk_sketch_Ad.cpu(), "sketch_Bd": mlp.blk_sketch_Bd.cpu(),
        }
        torch.save(payload, os.path.join(out_dir, f"layer_{layer_idx}.pt"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"rank": rank, "r_sk": r_sk}, f, indent=2)


def load_block_factors(model, factors_dir):
    for layer_idx, mlp in iter_mlps(model):
        dev = mlp.down_proj.weight.device
        dtype = mlp.down_proj.weight.dtype
        d = torch.load(os.path.join(factors_dir, f"layer_{layer_idx}.pt"), map_location=dev)
        mlp.blk_comp_A = d["comp_A"].to(dtype)
        mlp.blk_comp_B = d["comp_B"].to(dtype)
        mlp.blk_sketch_Ag = d["sketch_Ag"].to(dtype)
        mlp.blk_sketch_Bg = d["sketch_Bg"].to(dtype)
        mlp.blk_sketch_Au = d["sketch_Au"].to(dtype)
        mlp.blk_sketch_Bu = d["sketch_Bu"].to(dtype)
        mlp.blk_sketch_Ad = d["sketch_Ad"].to(dtype)
        mlp.blk_sketch_Bd = d["sketch_Bd"].to(dtype)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def block_comp_mlp_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i = u * g

    score = _resid_score(mlp, u, g)
    m_bool = block_p_mask(score, mlp.blk_p, mlp.blk_g, seq_mask=getattr(mlp, "blk_seq_mask", None))
    achieved = 1.0 - m_bool.float().mean().item()
    mlp.infer_sparsity_h1 = 0.0
    mlp.infer_sparsity_h2 = achieved
    mlp.blk_sp_sum = getattr(mlp, "blk_sp_sum", 0.0) + achieved
    mlp.blk_sp_cnt = getattr(mlp, "blk_sp_cnt", 0) + 1

    m = m_bool.to(i.dtype)
    cond = mlp.blk_condition

    if cond == "c7a":
        return mlp.down_proj(m * i)

    if cond == "c7":
        gu = mlp.oracle_g_bar.to(u.dtype) * u
        comp = (x @ mlp.blk_comp_A.T) @ mlp.blk_comp_B.T
        return mlp.down_proj(m * i) + comp - mlp.down_proj(m * gu)

    if cond == "c8a":
        ghat = mlp.act_fn((x @ mlp.blk_sketch_Ag.T) @ mlp.blk_sketch_Bg.T)
        tail = (1.0 - m) * (ghat.to(u.dtype) * u)
        return mlp.down_proj(m * i) + mlp.down_proj(tail)

    if cond == "c8":
        ghat = mlp.act_fn((x @ mlp.blk_sketch_Ag.T) @ mlp.blk_sketch_Bg.T)
        uhat = (x @ mlp.blk_sketch_Au.T) @ mlp.blk_sketch_Bu.T
        tail = (1.0 - m) * (ghat.to(uhat.dtype) * uhat)
        tail_sketch = (tail @ mlp.blk_sketch_Ad.T) @ mlp.blk_sketch_Bd.T
        return mlp.down_proj(m * i) + tail_sketch

    raise ValueError(f"unknown block condition {cond!r}")


def set_condition(model, condition, p=1.0, g=1, seq_mask=None):
    assert condition in CONDITIONS, condition
    for _, mlp in iter_mlps(model):
        mlp.blk_condition = condition
        mlp.blk_p = p
        mlp.blk_g = g
        mlp.blk_seq_mask = seq_mask
        mlp.blk_sp_sum = 0.0
        mlp.blk_sp_cnt = 0


def achieved_sparsity_per_layer(model):
    """s_block = 1 - mean_T(|m_T|)/d, per spec section 3 (block-unit
    reporting axis -- do not mix with the per-token definition)."""
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        cnt = getattr(mlp, "blk_sp_cnt", 0)
        out[layer_idx] = (mlp.blk_sp_sum / cnt) if cnt else 0.0
    return out
