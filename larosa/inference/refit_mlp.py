# coding=utf-8
# Local Loss Refit: measure the isolated effect of refitting down_proj to a
# frozen mask, with no other repair mechanism (no mean-gate residual, no
# rank-r branch — those live in oracle_mlp.py / oracle-residual-sparsity).
#
# Mask: top-K per (token or token-block) on the C2-score |i|*col_norm(orig),
# i = u*g, u = up_proj(x), g = act_fn(gate_proj(x)). col_norm is ALWAYS taken
# from the ORIGINAL (never-refit) down_proj weight, in every condition —
# refit must never feed back into the score (anti-circularity).
#
# Ladder (see .labtool/topics/local-loss-refit/gist.md):
#   l0 : mask only, original W_down                         (dense input)
#   l1 : mask (frozen) + W_down refit per layer independently (dense input,
#        one sweep — every layer's regression uses the DENSE model's own
#        activations, not each other's refit output)
#   l2 : same mask rule, score recomputed against the SPARSE stream's own
#        activations; W_down refit layer-by-layer in sequence (teacher y*
#        is always the dense model; input to layer L is the output of
#        layers < L, already refit) — built by a separate sequential script;
#        this module's eval-time forward is identical to l1 once weights are
#        loaded (the streaming/sequential part is a BUILD-time concern only).
#
# Refit: closed-form ridge regression, fp32 accumulation, no gradients:
#   G = sum_t z_t z_t^T,  C = sum_t y*_t z_t^T   (z_t = m_t * i_t)
#   W_tilde_down = C (G + lambda * mean(diag(G)) * I)^-1

import json
import os

import torch

from .oracle_mlp import iter_mlps, top_k_mask  # reuse: no re-implementation

MODES = ("l0", "l1", "l2")


# ---------------------------------------------------------------------------
# block-grouped top-K masking
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


def block_mask(score, s, g, seq_mask=None):
    """Per-block top-K mask (K = int((1-s)*D)) broadcast back to per-token
    shape. g=1 reduces to top_k_mask(score, s) exactly (each block is a
    singleton, aggregation is the identity)."""
    block_score, bid = aggregate_block_score(score, g, seq_mask=seq_mask)
    m_block = top_k_mask(block_score, s)
    return m_block.index_select(-2, bid)


# ---------------------------------------------------------------------------
# weight-derived pieces
# ---------------------------------------------------------------------------

def attach_col_norms(model):
    # down_proj.weight is [h, d]; neuron j is column j. Call this BEFORE any
    # refit weight is loaded/overwritten -- the score must stay tied to the
    # original weight even after down_proj is replaced.
    for _, mlp in iter_mlps(model):
        mlp.refit_col_norm = mlp.down_proj.weight.float().norm(dim=0)


def score_c2(mlp, u, g):
    i = u * g
    return i, i.abs().float() * mlp.refit_col_norm


# ---------------------------------------------------------------------------
# eval-time forward (l0/l1/l2 are identical here -- they differ only in
# which down_proj weight is loaded and whether the mask was built against a
# dense or sparse stream at BUILD time; see scripts/refit/)
# ---------------------------------------------------------------------------

def refit_mlp_forward(mlp, x):
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i, score = score_c2(mlp, u, g)
    m_bool = block_mask(score, mlp.refit_s, mlp.refit_g)
    achieved = 1.0 - m_bool.float().mean().item()
    mlp.infer_sparsity_h1 = 0.0
    mlp.infer_sparsity_h2 = achieved
    z = m_bool.to(i.dtype) * i
    return mlp.down_proj(z)


def set_condition(model, mode, s=0.0, g=1):
    assert mode in MODES, mode
    for _, mlp in iter_mlps(model):
        mlp.refit_mode = mode
        mlp.refit_s = s
        mlp.refit_g = g


# ---------------------------------------------------------------------------
# L1 calibration: single dense forward pass, per-layer independent (G, C)
# accumulation. Mirrors oracle_mlp.py's stats-mode pattern: the forward
# returns the TRUE DENSE output (so every layer really does see dense input,
# per L1's definition) while accumulating regression statistics as a side
# effect.
# ---------------------------------------------------------------------------

def enable_l1_collect_mode(model, s, g, layers=None):
    """layers: iterable of layer indices to actually accumulate (G, C) for
    this pass (None = all). Every layer still runs l1_collect_forward (plain
    dense passthrough, so downstream layers always see dense input, per L1's
    definition) -- only the (G, C) allocation, and therefore the memory
    cost, is restricted to `layers`. Needed because a [d,d] fp32 G per layer
    is large for bigger models (e.g. 0.82GB/layer at d=14336 -> 26GB for all
    32 layers of Llama-3.1-8B at once, on top of the model weights); the
    build script chunks layers across multiple calibration sweeps to keep
    this bounded (see scripts/refit/01_build_l1.py --layers_per_pass)."""
    target = set(layers) if layers is not None else None
    for layer_idx, mlp in iter_mlps(model):
        mlp.refit_collect = True
        mlp.refit_s = s
        mlp.refit_g = g
        if target is None or layer_idx in target:
            d = mlp.intermediate_size
            h = mlp.hidden_size
            dev = mlp.down_proj.weight.device
            mlp.refit_G = torch.zeros(d, d, dtype=torch.float32, device=dev)
            mlp.refit_C = torch.zeros(h, d, dtype=torch.float32, device=dev)
            mlp.refit_n = 0


def _accumulate_l1(mlp, i, score, y_star):
    if not hasattr(mlp, "refit_G"):
        return  # this layer is not in the current collection chunk
    m_bool = block_mask(score, mlp.refit_s, mlp.refit_g)
    z = (m_bool.to(i.dtype) * i).float().reshape(-1, i.shape[-1])
    y = y_star.float().reshape(-1, y_star.shape[-1])
    mlp.refit_G += z.T @ z
    mlp.refit_C += y.T @ z
    mlp.refit_n += z.shape[0]


def l1_collect_forward(mlp, x):
    """Used only while refit_collect is set; returns the plain dense output
    (so downstream layers see dense input, matching L1's definition) and
    accumulates (G, C) for this layer's regression as a side effect. Also
    stashes the dense output as `refit_last_y_star` -- L2's paired
    teacher/student pass reads this back (see l2_collect_forward) so the
    student call's (G, C) accumulation can target the SAME chunk's teacher
    output without a second, separate dense forward."""
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i, score = score_c2(mlp, u, g)
    y_dense = mlp.down_proj(i)
    mlp.refit_last_y_star = y_dense.detach()
    _accumulate_l1(mlp, i, score, y_dense)
    return y_dense


def l2_collect_forward(mlp, x):
    """L2's per-layer accumulation pass on the SPARSE stream: x is the
    sparse stream's actual input at this layer, so mask/score/z are
    recomputed against it (L2's definition, vs L1's frozen-dense mask).
    The regression target is the teacher output the paired dense-stream
    call stashed on this same mlp via l1_collect_forward
    (`refit_last_y_star`) for the SAME calibration chunk. Returns a
    throwaway dense-ish value -- the L2 build script discards this pass's
    output and reruns the layer once more after solving (ordinary
    refit_mlp_forward, with the now-installed W_tilde_down)."""
    u = mlp.up_proj(x)
    g = mlp.act_fn(mlp.gate_proj(x))
    i, score = score_c2(mlp, u, g)
    _accumulate_l1(mlp, i, score, mlp.refit_last_y_star)
    return mlp.down_proj(i)


def finalize_l1(model):
    """Returns {layer_idx: {'G':..., 'C':..., 'n':...}} for whichever layers
    were in the active collection chunk (see enable_l1_collect_mode), and
    clears collect mode. Does NOT solve -- solving happens in the build
    script so lambda can be swept without recomputing G/C."""
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        if hasattr(mlp, "refit_G"):
            assert mlp.refit_n > 0, "finalize_l1 called before any calibration tokens"
            out[layer_idx] = {"G": mlp.refit_G.cpu(), "C": mlp.refit_C.cpu(), "n": mlp.refit_n}
            del mlp.refit_G, mlp.refit_C
        mlp.refit_collect = False
    return out


# ---------------------------------------------------------------------------
# L2 calibration: sequential (GPTQ-style) refit. Layer ell's calibration
# input is the SPARSE stream's output from layers < ell (already refit);
# its regression target is always the ORIGINAL dense model's output at
# layer ell (never the sparse stream's). Processed strictly in layer order,
# one layer at a time -- O(L) total layer-equivalent forward passes, not
# O(L^2): each layer needs only ONE extra pass over the calibration set
# (the dense "teacher" stream is advanced in that same pass) plus a cheap
# second pass to install the solved weight, not a full-model replay.
# ---------------------------------------------------------------------------

def single_layer_forward(model, layer_idx, hidden_states):
    """Run exactly one decoder layer's forward (attention + MLP + both
    residual adds) on `hidden_states`, reusing the model's own causal-mask
    and rotary-position-embedding construction (LlamaModel.forward) instead
    of re-deriving them by hand. Temporarily shrinks model.model.layers to
    just this one layer and replaces the final RMSNorm with the identity
    (that norm belongs only after the true last layer), then restores both.
    Attention is unaffected by sparse_mode='refit' (DENSE_ATTN_MODES), so
    this is valid for both the dense and sparse streams."""
    base = model.model
    layer = base.layers[layer_idx]
    orig_layers, orig_norm = base.layers, base.norm
    base.layers = torch.nn.ModuleList([layer])
    base.norm = torch.nn.Identity()
    try:
        out = base(inputs_embeds=hidden_states, use_cache=False,
                   output_hidden_states=False, return_dict=True)
        return out.last_hidden_state
    finally:
        base.layers, base.norm = orig_layers, orig_norm


def enable_l2_layer_collect(model, layer_idx, s, g):
    """(G, C) accumulators for exactly ONE layer -- L2 processes layers
    strictly in sequence, so memory never scales with model depth (unlike
    L1's need to chunk many simultaneous [d,d] accumulators)."""
    for idx, mlp in iter_mlps(model):
        mlp.refit_s = s
        mlp.refit_g = g
        mlp.refit_collect = False
        mlp.refit_l2_student = False
        if idx == layer_idx:
            d, h = mlp.intermediate_size, mlp.hidden_size
            dev = mlp.down_proj.weight.device
            mlp.refit_G = torch.zeros(d, d, dtype=torch.float32, device=dev)
            mlp.refit_C = torch.zeros(h, d, dtype=torch.float32, device=dev)
            mlp.refit_n = 0


def set_l2_role(model, layer_idx, role):
    """role: 'teacher' (dense passthrough via l1_collect_forward, stashes
    refit_last_y_star) or 'student' (l2_collect_forward: mask/z from the
    sparse input, accumulates against the stashed teacher output) or
    'apply' (ordinary refit_mlp_forward, masked eval with whatever
    down_proj weight is currently installed -- used for L2's pass 2, after
    solving). Only the named layer's mlp is touched."""
    assert role in ("teacher", "student", "apply"), role
    _, mlp = next((idx, m) for idx, m in iter_mlps(model) if idx == layer_idx)
    mlp.refit_collect = role == "teacher"
    mlp.refit_l2_student = role == "student"


def finalize_l2_layer(model, layer_idx):
    _, mlp = next((idx, m) for idx, m in iter_mlps(model) if idx == layer_idx)
    assert mlp.refit_n > 0, "finalize_l2_layer called before any calibration tokens"
    out = {"G": mlp.refit_G.cpu(), "C": mlp.refit_C.cpu(), "n": mlp.refit_n}
    del mlp.refit_G, mlp.refit_C
    mlp.refit_collect = False
    mlp.refit_l2_student = False
    return out


# ---------------------------------------------------------------------------
# closed-form solve + weight I/O
# ---------------------------------------------------------------------------

def solve_refit(G, C, W_anchor, lam=0.01):
    """W_tilde = (C + lambda*diag_mean*W_anchor) (G + lambda*diag_mean*I)^-1,
    via Cholesky. G: [d,d] fp32 (symmetric PSD), C: [h,d] fp32, W_anchor:
    [h,d] fp32 (the ORIGINAL, never-refit down_proj weight -- same anchor
    used for the C2 score, anti-circularity applies here too).

    C1 fix: the original ridge (anchor 0, i.e. W_anchor=0 recovers it
    exactly) is a 0-shrinkage prior -- when a column has little
    calibration evidence (G's corresponding diagonal small), it pulls
    W_tilde toward the all-zero row instead of toward "trust the original
    weight." At low sparsity (s=0.5/0.7) masking barely changes activations,
    so there is little true bias for refit to correct, and the 0-anchor
    ridge was adding pure shrinkage-toward-zero noise -- the master
    design's fix anchors to W_anchor instead, so "no evidence" means
    "keep the original weight," not "shrink to zero." Reduces to the
    original formula when W_anchor is the zero matrix."""
    d = G.shape[0]
    diag_mean = torch.diagonal(G).mean()
    reg = G + lam * diag_mean * torch.eye(d, dtype=G.dtype, device=G.device)
    rhs = C + lam * diag_mean * W_anchor.to(C.dtype)
    L = torch.linalg.cholesky(reg)
    X = torch.cholesky_solve(rhs.T.contiguous(), L)  # [d, h] = reg^-1 rhs^T
    return X.T.contiguous()  # [h, d]


def save_refit_weights(out_dir, weights, meta=None):
    """weights: {layer_idx: W_tilde [h,d] fp32 tensor}."""
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, w in weights.items():
        torch.save({"W_down": w.cpu()}, os.path.join(out_dir, f"layer_{layer_idx}.pt"))
    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def load_refit_weights(model, weights_dir):
    for layer_idx, mlp in iter_mlps(model):
        d = torch.load(os.path.join(weights_dir, f"layer_{layer_idx}.pt"),
                       map_location=mlp.down_proj.weight.device)
        mlp.down_proj.weight.data.copy_(d["W_down"].to(mlp.down_proj.weight.dtype))


# ---------------------------------------------------------------------------
# calibration token reproducibility: the corpus-streaming/tokenizer part
# lives in scripts/refit/ (needs `datasets` + a tokenizer); this is the pure,
# testable core -- deterministic reshape + save/reuse, same discipline as
# scripts/oracle/01_calibrate.py's calib_tokens.pt reuse.
# ---------------------------------------------------------------------------

def reshape_calib_tokens(id_buffer, nsamples, seqlen):
    need = nsamples * seqlen
    assert len(id_buffer) >= need, f"corpus exhausted: {len(id_buffer)} < {need} tokens"
    return torch.tensor(id_buffer[:need], dtype=torch.long).reshape(nsamples, seqlen)


def save_calib_tokens(tokens, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(tokens, path)


def load_calib_tokens(path, nsamples, seqlen):
    tokens = torch.load(path)
    assert tokens.shape == (nsamples, seqlen), \
        f"saved calib tokens {tuple(tokens.shape)} != requested ({nsamples},{seqlen})"
    return tokens


def achieved_sparsity_per_layer(model):
    out = {}
    for layer_idx, mlp in iter_mlps(model):
        out[layer_idx] = getattr(mlp, "infer_sparsity_h2", 0.0)
    return out
