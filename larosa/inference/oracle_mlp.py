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


def top_count_mask(score, k):
    """Per-token mask keeping the k largest entries of the last dim (>= kth
    value keeps ties, same semantics as top_k_mask). k=0 -> all-False."""
    d = score.shape[-1]
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
        _accumulate_stats(mlp, x, u, g)
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
    # compensation branch comp(x) ~= M x, M = W_d diag(g_bar) W_up.
    #   lr         : comp = B(Ax), rank-r SVD of M (original C4)
    #   slr_neuron : comp = B(Ax) + exact hot rank-1 neuron terms,
    #                A,B from SVD of M_cold (H4/S1)
    #   slr_input  : comp = B(Ax) + R (m_x * x), R = M - BA, m_x = top-k
    #                input channels by |x| or |x|*||R[:,c]|| (H4/S2)
    gu = mlp.oracle_g_bar.to(u.dtype) * u
    comp = (x @ mlp.oracle_A.T) @ mlp.oracle_B.T
    mode = getattr(mlp, "oracle_comp_mode", "lr")
    if mode == "slr_neuron":
        hot = mlp.oracle_hot_mask.to(gu.dtype)
        return mlp.down_proj(m * i + (hot - m) * gu) + comp
    if mode == "slr_input":
        score = x.abs().float()
        if getattr(mlp, "oracle_x_score", "abs") == "wnorm":
            score = score * mlp.oracle_R_colnorm
        m_x = top_count_mask(score, mlp.oracle_sparse_k).to(x.dtype)
        comp = comp + (m_x * x) @ mlp.oracle_R.T
    return mlp.down_proj(m * i) + comp - mlp.down_proj(m * gu)


# ---------------------------------------------------------------------------
# calibration statistics (fp32 batch reduction, fp64 running totals)
# ---------------------------------------------------------------------------

def enable_stats_mode(model, xxt=False):
    for _, mlp in iter_mlps(model):
        d = mlp.intermediate_size
        dev = mlp.down_proj.weight.device
        mlp.oracle_stats_mode = True
        mlp.oracle_sum_g = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_sum_g2 = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_sum_u2 = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_sum_u2g = torch.zeros(d, dtype=torch.float64, device=dev)
        mlp.oracle_stat_count = 0
        if xxt:
            # input autocorrelation Sigma = E[x x^T] for whitened SVD (fp32)
            h = mlp.hidden_size
            mlp.oracle_sum_xxt = torch.zeros(h, h, dtype=torch.float32, device=dev)


def _accumulate_stats(mlp, x, u, g):
    g32 = g.float().reshape(-1, g.shape[-1])
    u32 = u.float().reshape(-1, u.shape[-1])
    u2 = u32 * u32
    mlp.oracle_sum_g += g32.sum(0).double()
    mlp.oracle_sum_g2 += (g32 * g32).sum(0).double()
    mlp.oracle_sum_u2 += u2.sum(0).double()
    mlp.oracle_sum_u2g += (u2 * g32).sum(0).double()
    mlp.oracle_stat_count += g32.shape[0]
    if hasattr(mlp, "oracle_sum_xxt"):
        x32 = x.float().reshape(-1, x.shape[-1])
        mlp.oracle_sum_xxt += x32.T @ x32


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
        if hasattr(mlp, "oracle_sum_xxt"):
            out[layer_idx]["sigma"] = (mlp.oracle_sum_xxt / n).cpu()
            del mlp.oracle_sum_xxt
        mlp.oracle_stats_mode = False
        mlp.oracle_g_bar = g_bar
        mlp.oracle_g_bar_star = g_bar_star
    return out


def save_stats(stats, out_dir, meta=None):
    # sigma matrices are large ([h,h] fp32) and live in their own files so the
    # small per-layer stat files (and any older baselines) stay untouched
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, d in stats.items():
        d = dict(d)
        sigma = d.pop("sigma", None)
        torch.save(d, os.path.join(out_dir, f"layer_{layer_idx}.pt"))
        if sigma is not None:
            torch.save({"sigma": sigma}, os.path.join(out_dir, f"sigma_layer_{layer_idx}.pt"))
    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def load_sigma(stats_dir, layer_idx, device="cpu"):
    return torch.load(os.path.join(stats_dir, f"sigma_layer_{layer_idx}.pt"),
                      map_location=device)["sigma"].float()


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


def compute_M(mlp):
    w_down = mlp.down_proj.weight.float()   # [h, d]
    w_up = mlp.up_proj.weight.float()       # [d, h]
    return (w_down * mlp.oracle_g_bar.to(w_down.device).unsqueeze(0)) @ w_up  # [h, h]


def build_M_factors(mlp, rank, sigma=None):
    """M = W_down diag(g_bar) W_up; SVD of M (plain) or M @ C (whitened,
    C = cholesky(Sigma + eps I), eps = 1e-4 mean(diag)). Returns (A, B, S):
    A [r,h], B [h,r] fp32 on the layer's device; B @ A approximates M in both
    modes (whitened: A = sqrt(S) V^T C^{-1} via triangular solve)."""
    M = compute_M(mlp)
    if sigma is not None:
        sig = sigma.to(M.device).float()
        eps = 1e-4 * sig.diagonal().mean()
        C = torch.linalg.cholesky(sig + eps * torch.eye(sig.shape[0], device=sig.device))
        target = M @ C
    else:
        C = None
        target = M
    U, S, Vh = torch.linalg.svd(target, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    B = U[:, :r] * sq.unsqueeze(0)          # [h, r]
    W = sq.unsqueeze(1) * Vh[:r, :]         # [r, h]
    if C is not None:
        # A = W C^{-1}  <=>  C^T A^T = W^T (C lower-triangular; no explicit inverse)
        A = torch.linalg.solve_triangular(C.mT, W.mT, upper=True).mT
    else:
        A = W
    return A, B, S


def hot_scores(mlp):
    """Static per-neuron importance of the rank-1 terms of M:
    c_j = g_bar_j * ||W_d[:,j]||_2 * ||W_u[j,:]||_2 (fp32, on device)."""
    w_down = mlp.down_proj.weight.float()
    w_up = mlp.up_proj.weight.float()
    g_bar = mlp.oracle_g_bar.to(w_down.device).float()
    return g_bar.abs() * w_down.norm(dim=0) * w_up.norm(dim=1)


def compute_M_cold(mlp, hot_idx):
    """M with the hot neurons' rank-1 terms removed (g_bar zeroed on hot)."""
    w_down = mlp.down_proj.weight.float()
    w_up = mlp.up_proj.weight.float()
    g_masked = mlp.oracle_g_bar.to(w_down.device).float().clone()
    g_masked[hot_idx] = 0.0
    return (w_down * g_masked.unsqueeze(0)) @ w_up


def build_slr_factors(mlp, rank, hot_n=0, mode="slr_neuron"):
    """Sparse + low-rank compensation factors (H4, R-Sparse template).

    slr_neuron: hot set H = top-hot_n neurons by hot_scores; A,B = rank-`rank`
      SVD of M_cold = M minus the hot rank-1 terms. The exact hot terms are
      applied at runtime through the existing gu path (2h MACs/neuron).
    slr_input: A,B = rank-`rank` SVD of the full M; the dense residual
      R = M - BA is applied to the top-k |x| input channels at runtime
      (h MACs/channel). R is rebuilt at load time, never stored.
    Returns (A [r,h], B [h,r], S, hot_idx) — hot_idx None for slr_input.
    rank=0 yields empty A/B (pure-sparse arm)."""
    assert mode in ("slr_neuron", "slr_input"), mode
    if mode == "slr_neuron":
        hot_idx = torch.topk(hot_scores(mlp), hot_n).indices.sort().values if hot_n else \
            torch.empty(0, dtype=torch.long, device=mlp.down_proj.weight.device)
        target = compute_M_cold(mlp, hot_idx)
    else:
        hot_idx = None
        target = compute_M(mlp)
    U, S, Vh = torch.linalg.svd(target, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    B = U[:, :r] * sq.unsqueeze(0)
    A = sq.unsqueeze(1) * Vh[:r, :]
    return A, B, S, hot_idx


def ranks_from_tau(spectra, tau):
    """Per-layer rank = smallest r whose cumulative squared-singular-value
    energy reaches tau. spectra: {layer_idx: 1-D tensor S}."""
    out = {}
    for l, S in spectra.items():
        e = S.double() ** 2
        c = torch.cumsum(e, 0) / e.sum()
        out[l] = int((c < tau).sum().item()) + 1
    return out


def ranks_for_budget(spectra, r_bar, iters=60):
    """Bisect tau so that mean(rank_l) <= r_bar (as close as possible)."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        mean = sum(ranks_from_tau(spectra, mid).values()) / len(spectra)
        if mean > r_bar:
            hi = mid
        else:
            lo = mid
    return ranks_from_tau(spectra, lo)


def save_factors(model, rank, out_dir, stats_dir=None, whiten=False, ranks=None,
                 comp_mode="lr", hot_n=0, sparse_k=0, x_score="abs"):
    """ranks: optional {layer_idx: r_l} overriding the uniform rank.
    comp_mode 'slr_neuron'/'slr_input' saves H4 hybrid factors instead of the
    plain SVD; sparse_k and x_score are runtime knobs recorded in the meta
    (slr_input only). whiten is incompatible with the slr modes."""
    assert comp_mode in ("lr", "slr_neuron", "slr_input"), comp_mode
    assert not (whiten and comp_mode != "lr"), "whiten only supports comp_mode='lr'"
    os.makedirs(out_dir, exist_ok=True)
    meta = {"whiten": whiten, "ranks": {}, "uniform_rank": rank,
            "comp_mode": comp_mode, "hot_n": hot_n, "sparse_k": sparse_k,
            "x_score": x_score}
    for layer_idx, mlp in iter_mlps(model):
        r_l = ranks[layer_idx] if ranks else rank
        if comp_mode == "lr":
            sigma = load_sigma(stats_dir, layer_idx, mlp.down_proj.weight.device) if whiten else None
            A, B, S = build_M_factors(mlp, r_l, sigma=sigma)
            hot_idx = None
        else:
            A, B, S, hot_idx = build_slr_factors(mlp, r_l, hot_n=hot_n, mode=comp_mode)
        payload = {"A": A.cpu(), "B": B.cpu(), "S": S.cpu(), "rank": r_l}
        if hot_idx is not None:
            payload["hot_idx"] = hot_idx.cpu()
        torch.save(payload, os.path.join(out_dir, f"layer_{layer_idx}.pt"))
        meta["ranks"][str(layer_idx)] = r_l
    meta["mean_rank"] = sum(meta["ranks"].values()) / len(meta["ranks"])
    with open(os.path.join(out_dir, "factors_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _attach_slr_runtime(mlp, comp_mode, hot_idx=None, sparse_k=0, x_score="abs"):
    """Attach the per-layer runtime pieces the c4 slr branches need. For
    slr_input, R = M - BA is rebuilt from the weights (never stored on disk)."""
    dev = mlp.down_proj.weight.device
    dtype = mlp.down_proj.weight.dtype
    mlp.oracle_comp_mode = comp_mode
    if comp_mode == "slr_neuron":
        hot_mask = torch.zeros(mlp.intermediate_size, dtype=torch.bool, device=dev)
        if hot_idx is not None and hot_idx.numel():
            hot_mask[hot_idx.to(dev)] = True
        mlp.oracle_hot_mask = hot_mask
    elif comp_mode == "slr_input":
        R = compute_M(mlp) - mlp.oracle_B.float() @ mlp.oracle_A.float()
        mlp.oracle_R_colnorm = R.norm(dim=0)
        mlp.oracle_R = R.to(dtype)
        mlp.oracle_sparse_k = sparse_k
        mlp.oracle_x_score = x_score


def load_factors(model, factors_dir):
    meta_path = os.path.join(factors_dir, "factors_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    comp_mode = meta.get("comp_mode", "lr")
    for layer_idx, mlp in iter_mlps(model):
        dev = mlp.down_proj.weight.device
        dtype = mlp.down_proj.weight.dtype
        d = torch.load(os.path.join(factors_dir, f"layer_{layer_idx}.pt"), map_location=dev)
        mlp.oracle_A = d["A"].to(dtype)
        mlp.oracle_B = d["B"].to(dtype)
        if comp_mode == "lr":
            mlp.oracle_comp_mode = "lr"  # clear any stale slr attachment
        else:
            _attach_slr_runtime(mlp, comp_mode, hot_idx=d.get("hot_idx"),
                                sparse_k=meta.get("sparse_k", 0),
                                x_score=meta.get("x_score", "abs"))


def attach_factors_inplace(model, rank):
    """Build and attach A/B directly (used by tests; 03_build_M.py saves to disk)."""
    for _, mlp in iter_mlps(model):
        A, B, _ = build_M_factors(mlp, rank)
        dtype = mlp.down_proj.weight.dtype
        mlp.oracle_A = A.to(dtype)
        mlp.oracle_B = B.to(dtype)
        mlp.oracle_comp_mode = "lr"


def attach_slr_factors_inplace(model, rank, hot_n=0, mode="slr_neuron",
                               sparse_k=0, x_score="abs"):
    """Build and attach slr factors + runtime pieces directly (tests)."""
    for _, mlp in iter_mlps(model):
        A, B, _, hot_idx = build_slr_factors(mlp, rank, hot_n=hot_n, mode=mode)
        dtype = mlp.down_proj.weight.dtype
        mlp.oracle_A = A.to(dtype)
        mlp.oracle_B = B.to(dtype)
        _attach_slr_runtime(mlp, mode, hot_idx=hot_idx,
                            sparse_k=sparse_k, x_score=x_score)


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
