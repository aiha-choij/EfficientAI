# coding=utf-8
# Group-Conditional Refit (GC-refit) E0 -- gate diagnostic only, no deployed
# method. Question: does the optimal anchor-ridge refit of down_proj
# actually depend on which neuron-blocks a token-block's P3' mask cut, or
# is the marginal (all-mask-patterns-pooled) refit already close to
# optimal for every mask pattern? If the latter, GC-refit (a full
# per-mask-cluster method) is not worth building.
#
# Reuses, unchanged:
#   - block_comp_mlp.py's P3' mask machinery (_resid_score, block_p3_mask's
#     two-step construction, build_neuron_partition_onehots) -- same
#     (g, B, s, score) setting as that topic's Phase 4.
#   - refit_mlp.py's solve_refit (anchor ridge, anchor = original W_down)
#     and its calibration-token save/reuse discipline (used by the driver
#     script, not this module).
#
# New here: local-loss-style (single dense pass, L1 flavor -- every layer
# sees the TRUE DENSE input, mask/z computed fresh from that input, target
# is that same layer's dense down_proj(i)) (G, C, Y2, n) accumulation that
# can be split by an externally supplied per-token-block cluster label,
# plus a held-out cross-evaluation pass and an oracle block-average
# compensation forward for Part 2.
#
# Four gc_role forward passes, dispatched by gc_refit_forward:
#   collect_a  : mask collection (stashes mu, the per-token-block B-length
#                keep pattern, for later clustering) + marginal (G,C,Y2,n)
#                accumulation (no cluster split -- this is the E0 "joint"
#                solution W_tilde^0 the request compares against).
#   collect_b  : same mask/z computation, (G,C,Y2,n) accumulation split by
#                a precomputed per-(layer,block) cluster label (0/1) --
#                requires clustering to have already run on collect_a's
#                stashed mu. Needs mlp.gc_seq_idx set before each forward
#                call (batch_size=1 discipline, see driver script) so the
#                current sequence's blocks can be sliced out of the full
#                per-layer cluster-id tensor.
#   eval       : held-out cross-evaluation -- for each candidate weight
#                (marginal / cluster-0 / cluster-1, per lambda), accumulate
#                SSE split by the held-out block's TRUE cluster label
#                (nearest-centroid assignment, computed by the driver
#                script from collect_a's fit-set centroids -- never
#                re-clustered on held-out data, which would leak).
#   oracle_avg : Part 2's oracle block-average compensation -- a genuine
#                end-to-end propagating forward (like block_comp_mlp's
#                c7a/c7/c8a/c8), not a local-loss diagnostic pass. Every
#                token in a token-block gets the SAME exact compensation:
#                the block-average of the masked-out contribution
#                W_down[(1-m_T)*i_t], not a per-token estimate.

import torch

from . import block_comp_mlp
from .oracle_mlp import iter_mlps


# ---------------------------------------------------------------------------
# shared P3' mask + mu (block-level neuron-block keep pattern) computation
# ---------------------------------------------------------------------------

def _block_p3_mask_mu(mlp, u, g):
    """Same math as block_comp_mlp.block_p3_mask / neuron_block_topm_mask,
    but also returns mu (the block-level keep pattern itself, [..,nb_token,
    B] bool) -- block_p3_mask only returns the per-neuron broadcast, which
    throws mu away. Deterministic given (u, g, mlp.blk_g, mlp.blk_neuron_M,
    mlp.blk_neuron_m), so recomputing it here rather than importing a
    mu-returning variant from block_comp_mlp.py keeps that ported file
    unmodified (bit-identical to its source branch)."""
    score = block_comp_mlp._resid_score(mlp, u, g)
    block_score, bid = block_comp_mlp.aggregate_block_score(score, mlp.blk_g, seq_mask=mlp.blk_seq_mask)
    bscore = block_score.float() @ mlp.blk_neuron_M
    top = torch.topk(bscore, mlp.blk_neuron_m, dim=-1).indices
    mu = torch.zeros_like(bscore).scatter_(-1, top, 1.0)
    per_neuron = (mu @ mlp.blk_neuron_M.T) > 0
    m_bool = per_neuron.index_select(-2, bid)
    return m_bool, mu.bool(), bid


def attach_partitions(model, partitions_path, B, m_keep, device, layers=None):
    """Loads the same P3 partition artifact block_comp_mlp.py's eval script
    uses and attaches per-layer M/m onto each mlp (mlp.blk_neuron_M,
    mlp.blk_neuron_m) -- required before any gc_role forward runs."""
    import os
    M = block_comp_mlp.build_neuron_partition_onehots(
        os.path.expanduser(partitions_path), B, device, layers=layers)
    for layer_idx, mlp in iter_mlps(model):
        if layer_idx in M:
            mlp.blk_neuron_M = M[layer_idx]
            mlp.blk_neuron_m = m_keep


def set_seq_idx(model, seq_idx):
    """Must be called before every forward pass in collect_b/eval roles
    (batch_size=1 discipline): tells each layer which global fit/held-out
    sequence index the upcoming single-sequence batch is, so collect_b/eval
    can slice the right rows out of their full per-layer cluster-id
    tensors."""
    for _, mlp in iter_mlps(model):
        mlp.gc_seq_idx = seq_idx


def set_role(model, role):
    for _, mlp in iter_mlps(model):
        mlp.gc_role = role


# ---------------------------------------------------------------------------
# collect_a: mask stash (for clustering) + marginal (G, C, Y2, n)
# ---------------------------------------------------------------------------

def enable_collect_a(model, layers=None):
    """layers: which layers actually get (G,C,Y2,n) accumulators this pass
    (None = all) -- mirrors refit_mlp.enable_l1_collect_mode's chunking
    discipline (a [d,d] fp32 G per layer is large; the driver script chunks
    layers across repeated calibration sweeps). Every layer still computes
    and stashes mu regardless of chunk membership -- mask collection is
    cheap (a [nb_block, B] bool per layer per sequence)."""
    target = set(layers) if layers is not None else None
    for layer_idx, mlp in iter_mlps(model):
        mlp.gc_role = "collect_a"
        if target is None or layer_idx in target:
            d, h = mlp.intermediate_size, mlp.hidden_size
            dev = mlp.down_proj.weight.device
            mlp.gc_G = torch.zeros(d, d, dtype=torch.float32, device=dev)
            mlp.gc_C = torch.zeros(h, d, dtype=torch.float32, device=dev)
            mlp.gc_Y2 = torch.zeros((), dtype=torch.float32, device=dev)
            mlp.gc_n = 0


def _collect_a_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i = u * g
    m_bool, mu, bid = _block_p3_mask_mu(mlp, u, g)
    y_dense = mlp.down_proj(i)
    mlp.gc_last_mu = mu.detach().to("cpu", torch.bool)
    if hasattr(mlp, "gc_G"):
        z = (m_bool.to(i.dtype) * i).float().reshape(-1, i.shape[-1])
        y = y_dense.float().reshape(-1, y_dense.shape[-1])
        mlp.gc_G += z.T @ z
        mlp.gc_C += y.T @ z
        mlp.gc_Y2 += (y * y).sum()
        mlp.gc_n += z.shape[0]
    return y_dense


def finalize_collect_a(model):
    """Returns {layer_idx: {'G','C','Y2','n'}} for whichever layers were in
    the active chunk; clears those accumulators (mlp.gc_last_mu, the most
    recent call's mask, is left in place -- collect_a is also used purely
    for mask collection on layers outside the current (G,C) chunk)."""
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        if hasattr(mlp, "gc_G"):
            assert mlp.gc_n > 0, "finalize_collect_a called before any calibration tokens"
            out[layer_idx] = {"G": mlp.gc_G.cpu(), "C": mlp.gc_C.cpu(),
                              "Y2": mlp.gc_Y2.cpu(), "n": mlp.gc_n}
            del mlp.gc_G, mlp.gc_C, mlp.gc_Y2
    return out


# ---------------------------------------------------------------------------
# balanced 2-way Hamming clustering of collected mu patterns
# ---------------------------------------------------------------------------

def balanced_hamming_kmeans(mu, seed=0, iters=25):
    """mu: [N, B] bool/float, one row per token-block (a single layer's
    collected mask patterns). Standard Lloyd 2-means under L1 (== Hamming
    for 0/1 vectors) distance, THEN a final assignment step that ranks all
    N points by their centroid-preference margin and splits at the exact
    median -- unconstrained k-means on skewed mask-cluster data can (and,
    checked during development, does on some layers) converge to a
    lopsided split, which defeats the point of a 2-cluster cross-eval.
    Deterministic given seed. Returns (labels [N] int64 in {0,1}, centroids
    [2, B] float32, computed from the FINAL balanced labels)."""
    x = mu.float()
    N, B = x.shape
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=gen)[:2]
    centroids = x[idx].clone()
    labels = None
    for _ in range(iters):
        d0 = (x - centroids[0]).abs().sum(-1)
        d1 = (x - centroids[1]).abs().sum(-1)
        new_labels = (d1 < d0).long()
        if labels is not None and torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in (0, 1):
            sel = labels == c
            if sel.any():
                centroids[c] = x[sel].mean(0)
    d0 = (x - centroids[0]).abs().sum(-1)
    d1 = (x - centroids[1]).abs().sum(-1)
    margin = d0 - d1  # more negative -> prefers cluster 0
    order = torch.argsort(margin)
    half = N // 2
    labels = torch.ones(N, dtype=torch.long)
    labels[order[:half]] = 0
    centroids = torch.stack([x[labels == 0].mean(0), x[labels == 1].mean(0)])
    return labels, centroids


def assign_nearest_centroid(mu, centroids):
    """mu: [N, B] bool/float. centroids: [2, B] float (from
    balanced_hamming_kmeans on the FIT set). Nearest-centroid assignment
    under L1 distance -- used for held-out blocks, which must never
    re-cluster (that would leak eval-slice structure into the cluster
    definition)."""
    x = mu.float()
    d0 = (x - centroids[0]).abs().sum(-1)
    d1 = (x - centroids[1]).abs().sum(-1)
    return (d1 < d0).long()


# ---------------------------------------------------------------------------
# collect_b: same mask/z computation, (G, C, Y2, n) split by a precomputed
# per-(layer, block) cluster label
# ---------------------------------------------------------------------------

def enable_collect_b(model, cluster_ids_by_layer, blocks_per_seq, layers=None):
    """cluster_ids_by_layer: {layer_idx: LongTensor[n_fit_blocks]} (global
    fit-set block index -> cluster label {0,1}), from balanced_hamming_kmeans
    on that layer's collect_a mu. blocks_per_seq: seqlen // g (constant
    across layers, batch_size=1 discipline)."""
    target = set(layers) if layers is not None else None
    for layer_idx, mlp in iter_mlps(model):
        mlp.gc_role = "collect_b"
        mlp.gc_blocks_per_seq = blocks_per_seq
        if target is None or layer_idx in target:
            d, h = mlp.intermediate_size, mlp.hidden_size
            dev = mlp.down_proj.weight.device
            mlp.gc_cluster_ids_full = cluster_ids_by_layer[layer_idx].to(dev)
            for suf in ("0", "1"):
                setattr(mlp, f"gc_G{suf}", torch.zeros(d, d, dtype=torch.float32, device=dev))
                setattr(mlp, f"gc_C{suf}", torch.zeros(h, d, dtype=torch.float32, device=dev))
                setattr(mlp, f"gc_Y2_{suf}", torch.zeros((), dtype=torch.float32, device=dev))
                setattr(mlp, f"gc_n{suf}", 0)


def _collect_b_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i = u * g
    m_bool, mu, bid = _block_p3_mask_mu(mlp, u, g)
    y_dense = mlp.down_proj(i)
    if hasattr(mlp, "gc_G0"):
        nb = mlp.gc_blocks_per_seq
        start = mlp.gc_seq_idx * nb
        block_labels = mlp.gc_cluster_ids_full[start:start + nb]
        tok_labels = block_labels.index_select(0, bid)
        z = (m_bool.to(i.dtype) * i).float().reshape(-1, i.shape[-1])
        y = y_dense.float().reshape(-1, y_dense.shape[-1])
        tok_labels = tok_labels.reshape(-1)
        for c in (0, 1):
            sel = (tok_labels == c).float().unsqueeze(-1)
            zc = z * sel
            yc = y * sel
            n_c = int(sel.sum().item())
            if n_c == 0:
                continue
            setattr(mlp, f"gc_G{c}", getattr(mlp, f"gc_G{c}") + zc.T @ zc)
            setattr(mlp, f"gc_C{c}", getattr(mlp, f"gc_C{c}") + yc.T @ zc)
            setattr(mlp, f"gc_Y2_{c}", getattr(mlp, f"gc_Y2_{c}") + (yc * yc).sum())
            setattr(mlp, f"gc_n{c}", getattr(mlp, f"gc_n{c}") + n_c)
    return y_dense


def finalize_collect_b(model):
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        if hasattr(mlp, "gc_G0"):
            assert mlp.gc_n0 > 0 and mlp.gc_n1 > 0, \
                "finalize_collect_b called before any calibration tokens (or a cluster got zero tokens)"
            out[layer_idx] = {
                0: {"G": mlp.gc_G0.cpu(), "C": mlp.gc_C0.cpu(), "Y2": mlp.gc_Y2_0.cpu(), "n": mlp.gc_n0},
                1: {"G": mlp.gc_G1.cpu(), "C": mlp.gc_C1.cpu(), "Y2": mlp.gc_Y2_1.cpu(), "n": mlp.gc_n1},
            }
            del mlp.gc_G0, mlp.gc_C0, mlp.gc_Y2_0, mlp.gc_G1, mlp.gc_C1, mlp.gc_Y2_1, mlp.gc_cluster_ids_full
    return out


# ---------------------------------------------------------------------------
# eval: held-out cross-evaluation -- SSE per candidate weight, split by the
# held-out block's TRUE (nearest-centroid) cluster label
# ---------------------------------------------------------------------------

def enable_eval(model, candidates_by_layer, held_cluster_ids_by_layer, blocks_per_seq, layers=None):
    """candidates_by_layer: {layer_idx: {name: W [h,d] fp32}}.
    held_cluster_ids_by_layer: {layer_idx: LongTensor[n_held_blocks]} (from
    assign_nearest_centroid against that layer's FIT centroids -- never
    re-clustered on held-out data)."""
    target = set(layers) if layers is not None else None
    for layer_idx, mlp in iter_mlps(model):
        mlp.gc_role = "eval"
        mlp.gc_blocks_per_seq = blocks_per_seq
        if target is None or layer_idx in target:
            dev = mlp.down_proj.weight.device
            mlp.gc_candidates = {name: W.to(device=dev, dtype=torch.float32)
                                 for name, W in candidates_by_layer[layer_idx].items()}
            mlp.gc_held_cluster_ids = held_cluster_ids_by_layer[layer_idx].to(dev)
            mlp.gc_sse = {name: {0: torch.zeros((), device=dev), 1: torch.zeros((), device=dev)}
                         for name in mlp.gc_candidates}
            mlp.gc_count = {0: 0, 1: 0}


def _eval_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i = u * g
    m_bool, mu, bid = _block_p3_mask_mu(mlp, u, g)
    y_dense = mlp.down_proj(i)
    if hasattr(mlp, "gc_candidates"):
        nb = mlp.gc_blocks_per_seq
        start = mlp.gc_seq_idx * nb
        block_labels = mlp.gc_held_cluster_ids[start:start + nb]
        tok_labels = block_labels.index_select(0, bid).reshape(-1)
        z = (m_bool.to(i.dtype) * i).float().reshape(-1, i.shape[-1])
        y = y_dense.float().reshape(-1, y_dense.shape[-1])
        for c in (0, 1):
            mlp.gc_count[c] += int((tok_labels == c).sum().item())
        for name, W in mlp.gc_candidates.items():
            y_hat = z @ W.T
            err2 = (y_hat - y).pow(2).sum(-1)
            for c in (0, 1):
                sel = tok_labels == c
                if sel.any():
                    mlp.gc_sse[name][c] += err2[sel].sum()
    return y_dense


def finalize_eval(model):
    """Returns {layer_idx: {'mse': {name: {0: mse0, 1: mse1}}, 'count': {0:n0,1:n1}}}.
    mse is per-output-element (SSE / (count * hidden_size)) -- the constant
    hidden_size scale factor cancels in every ratio this diagnostic reports,
    kept only so the numbers are on a legible MSE scale."""
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        if hasattr(mlp, "gc_candidates"):
            h = mlp.hidden_size
            res = {}
            for name in mlp.gc_candidates:
                res[name] = {c: (mlp.gc_sse[name][c].item() / max(mlp.gc_count[c], 1) / h) for c in (0, 1)}
            out[layer_idx] = {"mse": res, "count": dict(mlp.gc_count)}
            del mlp.gc_candidates, mlp.gc_sse, mlp.gc_held_cluster_ids
    return out


def closed_form_mse(W, G, C, Y2, n):
    """Exact fit-set (training) MSE of candidate weight W against the
    (G, C, Y2, n) accumulators, with no extra forward pass:
    SSE(W) = Y2 - 2*trace(W C^T) + trace(W G W^T)
    (Y2 = sum_t ||y*_t||^2, C = sum_t y*_t z_t^T, G = sum_t z_t z_t^T).
    Used for the required sanity check that the closed-form ridge solution
    never has worse fit-set MSE than the anchor (masking-only) weight --
    true by construction (W_tilde minimizes the ridge objective, which
    upper-bounds the anchor's objective value, which equals the anchor's
    plain SSE since the ridge penalty vanishes at W=W_anchor); a violation
    means a wiring bug, not a modeling surprise."""
    h = C.shape[0]
    sse = Y2 - 2 * (W * C).sum() + (W @ G * W).sum()
    return (sse / (n * h)).item()


# ---------------------------------------------------------------------------
# oracle_avg: Part 2's oracle block-average compensation (end-to-end
# propagating forward, like block_comp_mlp's c7a/c7/c8a/c8)
# ---------------------------------------------------------------------------

def enable_oracle_avg(model):
    for _, mlp in iter_mlps(model):
        mlp.gc_role = "oracle_avg"
        mlp.blk_sp_sum = 0.0
        mlp.blk_sp_cnt = 0


def _oracle_avg_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i = u * g
    m_bool, mu, bid = _block_p3_mask_mu(mlp, u, g)
    achieved = 1.0 - m_bool.float().mean().item()
    mlp.infer_sparsity_h1 = 0.0
    mlp.infer_sparsity_h2 = achieved
    mlp.blk_sp_sum = getattr(mlp, "blk_sp_sum", 0.0) + achieved
    mlp.blk_sp_cnt = getattr(mlp, "blk_sp_cnt", 0) + 1

    m = m_bool.to(i.dtype)
    kept = mlp.down_proj(m * i)
    tail_out = mlp.down_proj((1.0 - m) * i)  # exact per-token masked-out contribution W_d[(1-m_T)*i_t]
    block_sum, _ = block_comp_mlp.aggregate_block_score(tail_out.float(), mlp.blk_g, seq_mask=mlp.blk_seq_mask)
    block_avg = block_sum / float(mlp.blk_g)  # (1/g) * sum_{t' in T} W_d[(1-m_T)*i_t']
    comp = block_avg.index_select(-2, bid).to(tail_out.dtype)  # broadcast: every token in T uses the SAME comp
    return kept + comp


def gc_refit_forward(mlp, x):
    role = mlp.gc_role
    if role == "collect_a":
        return _collect_a_forward(mlp, x)
    if role == "collect_b":
        return _collect_b_forward(mlp, x)
    if role == "eval":
        return _eval_forward(mlp, x)
    if role == "oracle_avg":
        return _oracle_avg_forward(mlp, x)
    raise ValueError(f"unknown gc_role {role!r}")
