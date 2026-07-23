# coding=utf-8
# Oracle intermediate-sparsity conditions (C0-C5) for gated MLPs.
#
# Implements the ablation ladder of .labtool/topics/oracle-residual-sparsity/spec.md:
#   dense : y = W_d (u*g)                       (no modification)
#   c1    : score |i|,            y = W_d (m*i)
#   c2    : score |i|*coln,       y = W_d (m*i)
#   c3    : score |r|*coln,       y = W_d (m*i + (1-m)*(g_bar*u))      [diagnostic]
#   c4    : score |r|*coln,       y = W_d (m*i) + B(Ax) - W_d (m*(g_bar*u))
#   c5    : c3 with g_bar -> g_bar_star (u^2-weighted mean)
# where i = u*g, r = u*(g - g_bar), coln[j] = ||W_d[:,j]||_2.
#
# Everything is simulated compute-then-mask (oracle setting): the mask is
# derived from the true activations of the current token, per layer, never
# cached. Attention and every linear stay dense; only the MLP combination
# changes. Scores/statistics are fp32 even when the model runs bf16.

import json
import os

import torch

CONDITIONS = ("dense", "c1", "c2", "c3", "c4", "c5")


def best_attn_impl():
    """flash_attention_2 where available, else sdpa (CUDA) / eager (CPU).
    PPL impact of the backend is far inside pipeline noise; cross-condition
    comparisons always run within a single backend."""
    if not torch.cuda.is_available():
        return "eager"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def iter_mlps(model):
    for layer_idx, layer in enumerate(model.model.layers):
        yield layer_idx, layer.mlp


# ---------------------------------------------------------------------------
# selection masks: top-p (spec default) and top-K (exact-s, matches the
# larosa topk_intermediate experiment's semantics)
# ---------------------------------------------------------------------------

def top_k_mask(score, s):
    """Per-token top-K mask over the last dim with K = int((1-s)*d).

    Same K formula and tie handling (>= kth value keeps ties) as top_k_new in
    modeling_llama_larosa.py, so C1 under top-K selection reproduces the
    topk_intermediate experiment exactly. Returns a bool mask.
    """
    d = score.shape[-1]
    k = int((1.0 - s) * d)
    if k >= d:
        return torch.ones_like(score, dtype=torch.bool)
    if k <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    kth = torch.topk(score.float(), k, dim=-1).values[..., -1:]
    return score.float() >= kth


def top_p_mask(score, p, chunk=8192):
    """Per-token top-p mask over the last dim.

    score: [..., d], non-negative. Keeps the smallest prefix of the
    descending-sorted scores whose cumulative sum reaches p * total.
    Returns a bool mask of score's shape. Computed in fp32.
    """
    orig_shape = score.shape
    d = orig_shape[-1]
    flat = score.reshape(-1, d)
    mask = torch.empty(flat.shape, dtype=torch.bool, device=flat.device)
    arange = torch.arange(d, device=flat.device)
    for s in range(0, flat.shape[0], chunk):
        block = flat[s:s + chunk].float()
        sorted_s, idx = torch.sort(block, dim=-1, descending=True)
        csum = torch.cumsum(sorted_s, dim=-1)
        target = p * csum[:, -1:]
        k = torch.searchsorted(csum.contiguous(), target).clamp(max=d - 1)
        keep_sorted = arange.unsqueeze(0) <= k
        mask[s:s + chunk] = torch.zeros_like(keep_sorted).scatter(1, idx, keep_sorted)
    return mask.reshape(orig_shape)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def oracle_mlp_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i = u * g

    if getattr(mlp, "oracle_stats_mode", False):
        _accumulate_stats(mlp, u, g)
        mlp.infer_sparsity_h1 = 0.0
        mlp.infer_sparsity_h2 = 0.0
        return mlp.down_proj(i)

    cond = mlp.oracle_condition
    if cond == "dense" or getattr(mlp, "oracle_layer_dense", False):
        mlp.infer_sparsity_h1 = 0.0
        mlp.infer_sparsity_h2 = 0.0
        return mlp.down_proj(i)

    if cond in ("c1", "c2"):
        score = i.abs().float()
        if cond == "c2":
            score = score * mlp.oracle_col_norm
    elif cond in ("c3", "c4", "c5"):
        g_bar = mlp.oracle_g_bar_star if cond == "c5" else mlp.oracle_g_bar
        resid = u.float() * (g.float() - g_bar)
        score = resid.abs() * mlp.oracle_col_norm
    else:
        raise ValueError(f"unknown oracle condition {cond!r}")

    if getattr(mlp, "oracle_select", "topp") == "topk":
        m_bool = top_k_mask(score, mlp.oracle_s)
    else:
        m_bool = top_p_mask(score, mlp.oracle_p)
    achieved = 1.0 - m_bool.float().mean().item()
    mlp.infer_sparsity_h1 = 0.0
    mlp.infer_sparsity_h2 = achieved
    mlp.oracle_sp_sum = getattr(mlp, "oracle_sp_sum", 0.0) + achieved
    mlp.oracle_sp_cnt = getattr(mlp, "oracle_sp_cnt", 0) + 1

    m = m_bool.to(i.dtype)
    if cond in ("c1", "c2"):
        return mlp.down_proj(m * i)
    if cond in ("c3", "c5"):
        g_bar = mlp.oracle_g_bar_star if cond == "c5" else mlp.oracle_g_bar
        gu = g_bar.to(u.dtype) * u
        return mlp.down_proj(m * i + (1.0 - m) * gu)
    # c4: keep the exact kept-neuron residual, replace the tail with the
    # rank-r linear branch comp_lr(x) = (x A^T) B^T ~= M x, M = W_d diag(g_bar) W_u
    gu = mlp.oracle_g_bar.to(u.dtype) * u
    comp = (x @ mlp.oracle_A.T) @ mlp.oracle_B.T
    return mlp.down_proj(m * i) + comp - mlp.down_proj(m * gu)


# ---------------------------------------------------------------------------
# calibration statistics (fp32 batch reduction, fp64 running totals)
# ---------------------------------------------------------------------------

def enable_stats_mode(model):
    for _, mlp in iter_mlps(model):
        d = mlp.intermediate_size
        dev = mlp.down_proj.weight.device
        mlp.oracle_stats_mode = True
        mlp.oracle_sum_g = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_sum_g2 = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_sum_u2 = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_sum_u2g = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_stat_count = 0


def _accumulate_stats(mlp, u, g):
    g32 = g.float().reshape(-1, g.shape[-1])
    u32 = u.float().reshape(-1, u.shape[-1])
    u2 = u32 * u32
    mlp.oracle_sum_g += g32.sum(0).double()
    mlp.oracle_sum_g2 += (g32 * g32).sum(0).double()
    mlp.oracle_sum_u2 += u2.sum(0).double()
    mlp.oracle_sum_u2g += (u2 * g32).sum(0).double()
    mlp.oracle_stat_count += g32.shape[0]


def finalize_stats(model):
    """Turn accumulators into per-layer stat dicts and attach g_bar buffers."""
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        n = mlp.oracle_stat_count
        assert n > 0, "finalize_stats called before any calibration tokens"
        g_bar = (mlp.oracle_sum_g / n).float()
        e_g2 = (mlp.oracle_sum_g2 / n).float()
        g_bar_star = (mlp.oracle_sum_u2g / mlp.oracle_sum_u2.clamp(min=1e-30)).float()
        out[layer_idx] = {
            "g_bar": g_bar.cpu(),
            "g_bar_star": g_bar_star.cpu(),
            "e_g2": e_g2.cpu(),
            "count": n,
        }
        mlp.oracle_stats_mode = False
        mlp.oracle_g_bar = g_bar
        mlp.oracle_g_bar_star = g_bar_star
    return out


def save_stats(stats, out_dir, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, d in stats.items():
        torch.save(d, os.path.join(out_dir, f"layer_{layer_idx}.pt"))
    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def load_stats(model, stats_dir):
    for layer_idx, mlp in iter_mlps(model):
        d = torch.load(os.path.join(stats_dir, f"layer_{layer_idx}.pt"),
                       map_location=mlp.down_proj.weight.device)
        mlp.oracle_g_bar = d["g_bar"].float()
        mlp.oracle_g_bar_star = d["g_bar_star"].float()


# ---------------------------------------------------------------------------
# weight-derived pieces: column norms and the compensation factors
# ---------------------------------------------------------------------------

def attach_col_norms(model):
    # down_proj.weight is [h, d]; neuron j is column j
    for _, mlp in iter_mlps(model):
        mlp.oracle_col_norm = mlp.down_proj.weight.float().norm(dim=0)


def build_M_factors(mlp, rank):
    """M = W_down diag(g_bar) W_up, SVD, split at sqrt(S). Returns (A, B, S)
    with A [r,h], B [h,r], all fp32 on the layer's device."""
    w_down = mlp.down_proj.weight.float()   # [h, d]
    w_up = mlp.up_proj.weight.float()       # [d, h]
    M = (w_down * mlp.oracle_g_bar.to(w_down.device).unsqueeze(0)) @ w_up  # [h, h]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    A = sq.unsqueeze(1) * Vh[:r, :]         # [r, h]
    B = U[:, :r] * sq.unsqueeze(0)          # [h, r]
    return A, B, S


def save_factors(model, rank, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, mlp in iter_mlps(model):
        A, B, S = build_M_factors(mlp, rank)
        torch.save({"A": A.cpu(), "B": B.cpu(), "S": S.cpu(), "rank": rank},
                   os.path.join(out_dir, f"layer_{layer_idx}.pt"))


def load_factors(model, factors_dir):
    for layer_idx, mlp in iter_mlps(model):
        dev = mlp.down_proj.weight.device
        dtype = mlp.down_proj.weight.dtype
        d = torch.load(os.path.join(factors_dir, f"layer_{layer_idx}.pt"), map_location=dev)
        mlp.oracle_A = d["A"].to(dtype)
        mlp.oracle_B = d["B"].to(dtype)


def attach_factors_inplace(model, rank):
    """Build and attach A/B directly (used by tests; 03_build_M.py saves to disk)."""
    for _, mlp in iter_mlps(model):
        A, B, _ = build_M_factors(mlp, rank)
        dtype = mlp.down_proj.weight.dtype
        mlp.oracle_A = A.to(dtype)
        mlp.oracle_B = B.to(dtype)


# ---------------------------------------------------------------------------
# run configuration
# ---------------------------------------------------------------------------

def set_condition(model, condition, p=1.0, select="topp", s=0.0, exclude_layers=()):
    """select='topp' uses the spec's cumulative-mass knob p; select='topk'
    enforces exact sparsity s per token (K = int((1-s)*d), larosa semantics)."""
    assert condition in CONDITIONS, condition
    assert select in ("topp", "topk"), select
    for layer_idx, mlp in iter_mlps(model):
        mlp.oracle_condition = condition
        mlp.oracle_select = select
        mlp.oracle_p = p
        mlp.oracle_s = s
        mlp.oracle_layer_dense = layer_idx in exclude_layers
        mlp.oracle_sp_sum = 0.0
        mlp.oracle_sp_cnt = 0


def achieved_sparsity_per_layer(model):
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        cnt = getattr(mlp, "oracle_sp_cnt", 0)
        out[layer_idx] = (mlp.oracle_sp_sum / cnt) if cnt else 0.0
    return out
