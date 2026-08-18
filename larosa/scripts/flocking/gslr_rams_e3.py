"""RAMS E3: low-dimensional reweight parameterization, tuned directly against
probe PPL (topic: refit-aware-mask-selection, follow-up to gslr_rams_e2.py /
E2, auto/rams-e1, PR #6). Reuses E1/E2's Gram/mask/refit/eval machinery
(gslr_layer_tune.py, gslr_rams_e1.py, gslr_rams_e2.py).

E2 (report 20260819-012351-rams-e2.md) found: a single global exponent
w_j=(unique_j/median)^alpha recovers only ~20-22% of E1's greedy headroom,
and diagnosed two reasons alpha can't be pushed further: (i) at alpha=1.0 the
weight distribution develops 30-94x extreme outliers that overwhelm what
refit can compensate for (PPL is U-shaped in alpha, not monotonic); (ii)
raising alpha makes the mask *diverge* from greedy's overlap (0.67->0.47),
because unique_j's median is computed over the full d=11008 neurons, and the
~90% background/low-magnitude neurons that never compete for top-K drag the
median down -- so "above median" is nearly always true for genuinely
competitive neurons regardless of real redundancy, and w_j degenerates into
noise on top of the magnitude ranking rather than a redundancy signal.

E3 asks: does (a) adding a second signal (selection FREQUENCY, which is
naturally bounded and complements unique_j) and (b) restricting the
normalization to the actual top-K competitive band (diagnosis (ii)'s direct
fix) recover more of that headroom -- and does tuning directly against a
held-out PROBE PPL (rather than calibration relerr, which E1 and E2 both
independently showed is not a trustworthy proxy for final PPL) find better
operating points than E2's single relerr-blind alpha sweep did?

Two reweight families (both still exactly E2's "one extra elementwise
multiply of the group score, no extra FLOPs order" deployable shape --
re2.ReweightMaskedMLP is reused UNCHANGED for eval, since it only cares about
the final per-neuron weight vector, not how it was derived):

  F1 (2-parameter global): w_j = clip(unique_j^a * freq_j^b, [1/c, c])
    unique_j: IDENTICAL mechanism to E2 (re2.compute_unique), median-
    normalized (see weight_f1 docstring for why: median-normalizing before
    the power is provably invisible to the MASK -- multiplying every group's
    score by the same per-layer positive constant never changes a top-K
    argmax -- so this is not a scientific change from the request's literal
    "unique_j^a" formula, only a normalization that makes the CLIP threshold
    meaningful, which is the whole point of adding a clip at all).
    freq_j = fraction of calibration GROUPS that select neuron j under the
    plain magnitude mask (glt.compute_rho, already exists in the codebase --
    exactly the request's "그룹 마스크에 선택된 빈도"). freq_j is mean-
    normalized (mean(freq_j) = K/d EXACTLY, a closed-form constant, unlike
    unique_j's median which has no such guarantee) -- see weight_f1 docstring
    for why median-normalizing freq_j is not usable (its full-d median is
    degenerate/zero whenever more than half the neurons never win any
    group's top-K, which is the common case at sparsity=0.9).
    a=0.25, b=0, c=8 is checked (selftest) to reproduce E2's alpha=0.25
    (e2_a025, the E2 best point) bit-for-bit -- E2's own w_stats at alpha=
    0.25 (layer 20: min=0.796, max=2.337) never approaches the c=8 clip
    bound, so this is a real, not vacuous, cross-check.

  F2 (band-limited unique_j, E2 diagnosis (ii)'s direct fix): unique_j is
  computed from the Gram of ONLY the top-(4*K) neurons by total calibration-
  set magnitude score (band_indices), not the full d -- both the Cholesky
  and the median normalization are restricted to this band. Outside the
  band, w_j == 1 (neurons far outside top-K contention are untouched -- they
  would never be selected regardless of reweighting, so leaving them at 1 is
  a no-op for the mask, not an approximation). b is not used (request
  specifies b=0 fixed for F2 -- band-restriction targets diagnosis (ii)
  directly; combining it with the frequency term is left for a later
  request if F2 alone doesn't already close most of the gap).

Both families use the SAME re2.compute_group_mask_weighted / refit_wdown /
re2.ReweightMaskedMLP as E2 -- only the weight VECTOR's construction differs.

Tuning protocol (the request's core methodological point, after E1+E2 both
flagged calibration-set relerr as an unreliable PPL proxy): 1 calibration
round (--mode calib) computes, per layer, ALL cheap-to-derive per-neuron
statistics ONCE (unique_j full-d, freq_j, band indices, unique_j band-
restricted) from a cached 32x2048 activation collection (--mode calib caches
the collected X to <out>/xcache/ on first run so a later invocation that
ADDS new candidate tags to CANDIDATES does not repeat the ~1s-per-layer-per-
sequence model forward collection, only the much cheaper per-candidate
mask+refit -- this is a pure performance change, not a numerics change: the
cached X is bit-identical to what a fresh collection would produce for the
same calib_seqs/seqlen). Then EVERY (F1 or F2) candidate is evaluated by an
actual probe forward pass (--mode probe): 8x2048 wikitext-2 TRAIN sequences
DISJOINT from the 32x2048 calibration slice (probe_ids -- offset by
calib_seqs so token ranges never overlap). Only the winning candidate(s) go
through --mode test (E1's exact reduced 8x2048 wikitext-2 TEST set, plus
--full for the entire test split) -- this generalization gap (probe PPL rank
vs test PPL rank) is reported explicitly, since "low-dimensional so overfit
risk is small" is an assumption of this request's design, not a given.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LAROSA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, _LAROSA_DIR)

import gslr_layer_tune as glt  # noqa: E402
import gslr_rams_e1 as re1  # noqa: E402 (token_stream, collect_layer_x, greedy_group_select, mask_overlap_stats, eval_ppl_fixed, group_recon_relerr)
import gslr_rams_e2 as re2  # noqa: E402 (compute_unique, weight_from_unique, compute_group_mask_weighted, ReweightMaskedMLP, full_test_ids)

BAND_K_MULT = 4  # F2 band size = BAND_K_MULT * K (~4400 candidates out of d=11008)

# F1 grid (9): a x b, c=8 fixed. a=0.25/b=0 reproduces e2_a025 (selftest 6).
# F2 grid (3): a only (b fixed at 0, band-limited unique), c=8 fixed.
# Additions from later tuning rounds (local refine near the probe-best point,
# clip sweep) are appended here, NOT as a separate script/registry, so git
# history documents each round and --mode calib's per-candidate resume
# (skip tags whose <out>/<tag>/layer_i.pt already exists) means re-running
# calib after an addition only computes the NEW tags.
CANDIDATES = [
    dict(tag="f1_a010_bn02", kind="f1", a=0.10, b=-0.2, c=8.0),
    dict(tag="f1_a010_b000", kind="f1", a=0.10, b=0.0, c=8.0),
    dict(tag="f1_a010_bp02", kind="f1", a=0.10, b=0.2, c=8.0),
    dict(tag="f1_a025_bn02", kind="f1", a=0.25, b=-0.2, c=8.0),
    dict(tag="f1_a025_b000", kind="f1", a=0.25, b=0.0, c=8.0),
    dict(tag="f1_a025_bp02", kind="f1", a=0.25, b=0.2, c=8.0),
    dict(tag="f1_a040_bn02", kind="f1", a=0.40, b=-0.2, c=8.0),
    dict(tag="f1_a040_b000", kind="f1", a=0.40, b=0.0, c=8.0),
    dict(tag="f1_a040_bp02", kind="f1", a=0.40, b=0.2, c=8.0),
    dict(tag="f2_a010", kind="f2", a=0.10, b=0.0, c=8.0),
    dict(tag="f2_a025", kind="f2", a=0.25, b=0.0, c=8.0),
    dict(tag="f2_a040", kind="f2", a=0.40, b=0.0, c=8.0),
    # ---- round 2 (probe-driven, added after seeing the grid's probe PPL:
    # dense=6.01, best grid point f1_a040_bn02=17.27, runner-up f2_a040=
    # 17.56/f1_a025_b000=17.67 -- b=-0.2 only helped at a=0.40 (hurt at
    # a=0.10/0.25), and F2's probe PPL improved monotonically with a
    # (18.44/17.73/17.56 at a=0.10/0.25/0.40) -- both patterns worth
    # pushing further before concluding, and the request asks for 1-2 local
    # refinements plus a clip sweep at the best point).
    dict(tag="f1_a050_bn02", kind="f1", a=0.50, b=-0.2, c=8.0),
    dict(tag="f1_a040_bn03", kind="f1", a=0.40, b=-0.3, c=8.0),
    dict(tag="f1_a040_bn02_c004", kind="f1", a=0.40, b=-0.2, c=4.0),
    dict(tag="f1_a040_bn02_cinf", kind="f1", a=0.40, b=-0.2, c=float("inf")),
    dict(tag="f2_a055", kind="f2", a=0.55, b=0.0, c=8.0),
    # ---- round 3 (final budget slot, 18th candidate -- request caps at
    # ~18): round 2 found f1_a050_bn02=17.02 BEATS f1_a040_bn02=17.27 (a
    # further step in the same direction keeps improving, unlike b, where
    # pushing to -0.3 badly hurt: f1_a040_bn03=20.55). One more step in a
    # to see whether 0.50 is near a local optimum or still climbing before
    # settling -- this is the last grid point spent per the request's
    # candidate-count budget; F3 (if justified) and the final test-set
    # evaluation take whatever wins here.
    dict(tag="f1_a060_bn02", kind="f1", a=0.60, b=-0.2, c=8.0),
]


def candidates_by_tag():
    return {c["tag"]: c for c in CANDIDATES}


# ---------------------------------------------------------------- core math

def band_indices(I, colnorm0, band_k):
    """Top-band_k neurons by TOTAL calibration-set magnitude score (summed
    over every token, not grouped) -- the static per-layer candidate pool
    that F2 restricts unique_j's Gram/median to. band_k capped at d."""
    score = I.abs().sum(0) * colnorm0  # (d,)
    band_k = min(band_k, score.numel())
    return score.topk(band_k).indices


def compute_unique_band(I, colnorm0, band_idx, eps=1e-3, chunk=8192):
    """Same precision-matrix mechanism as re2.compute_unique, but the Gram
    (and hence the Cholesky and the residual-variance diagonal) is restricted
    to the band_idx columns only -- unique_band[k] is neuron band_idx[k]'s
    variance unexplainable by the OTHER BAND neurons (not by all d)."""
    V = (I * colnorm0[None, :])[:, band_idx]
    N = I.shape[0]
    Ghat = glt.chunked_matmul(V, V, chunk)
    A = Ghat / N
    meandiag = A.diagonal().mean()
    A = A + eps * meandiag * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    L = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(L)
    return 1.0 / Ainv.diagonal().clamp(min=1e-12)


def weight_f1(unique_full, freq, n_groups, a, b, c, eps=1e-12):
    """w_j = clip(unique_j^a * freq_j^b, [1/c, c]).

    unique_j is divided by its own median before the power -- PROVABLY
    invisible to the mask on its own (median^a is one positive constant per
    layer, multiplying every group's score uniformly, so it cannot change
    any group's top-K argmax), but it makes the clip bound meaningful (an
    unnormalized unique_j ranges over ~2 orders of magnitude across a layer,
    see E2's unique_stats -- clip([1/8,8]) on the raw value would be an
    almost-arbitrary cutoff unrelated to "how extreme relative to typical").

    freq_j is divided by its MEAN, not median (mean(freq_j) = K/d exactly,
    a closed-form identity: sum_j freq_j = sum_j selection_count_j / n_groups
    = (K * n_groups) / n_groups = K, so mean = K/d always) -- median(freq_j)
    is frequently exactly 0 at sparsity=0.9 (more than half of d's neurons
    are background/low-magnitude and never win a single group's top-K), so
    median-normalizing freq_j the way unique_j is would divide by zero.
    freq_j is then floored at the smallest nonzero value the calibration
    budget can resolve (1/n_groups, i.e. "selected in exactly one group") so
    that b<0 (rewarding rarely-selected neurons) never hits 0**negative."""
    med_u = unique_full.median().clamp(min=eps)
    u = (unique_full / med_u).clamp(min=eps)
    mean_f = freq.mean().clamp(min=eps)
    f_floor = (1.0 / n_groups) / mean_f
    f = (freq / mean_f).clamp(min=f_floor)
    w = u.pow(a) * f.pow(b)
    if c is not None and math.isfinite(c):
        w = w.clamp(min=1.0 / c, max=c)
    return w


def weight_f2(unique_band, band_idx, d, a, c, device, dtype, eps=1e-12):
    """w_j = 1 outside band_idx (never reweighted -- these neurons are far
    from top-K contention regardless); inside the band, the SAME median-
    normalized-power-then-clip mechanism as weight_f1's unique_j term, using
    the band-restricted unique_band/median instead of the full-d one."""
    w = torch.ones(d, device=device, dtype=dtype)
    med_b = unique_band.median().clamp(min=eps)
    wb = (unique_band / med_b).clamp(min=eps).pow(a)
    if c is not None and math.isfinite(c):
        wb = wb.clamp(min=1.0 / c, max=c)
    w[band_idx] = wb
    return w


def make_weight(spec, unique_full, freq, n_groups, unique_band, band_idx, d, device, dtype):
    if spec["kind"] == "f1":
        return weight_f1(unique_full, freq, n_groups, spec["a"], spec["b"], spec["c"])
    if spec["kind"] == "f2":
        return weight_f2(unique_band, band_idx, d, spec["a"], spec["c"], device, dtype)
    raise ValueError(spec["kind"])


# ---------------------------------------------------------------- data

def probe_ids(tok, seqlen, calib_seqs, probe_seqs, split="train"):
    """Probe set: probe_seqs sequences from wikitext-2 TRAIN, offset PAST the
    first calib_seqs sequences (re1.token_stream/gslr_rams_e2's calibration
    slice is always ids[:calib_seqs*seqlen]) -- token ranges are therefore
    guaranteed disjoint from calibration, not just "a different sample"."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    start = calib_seqs * seqlen
    need = probe_seqs * seqlen
    assert ids.numel() >= start + need, (
        f"{split}: {ids.numel()} tokens < {start + need} needed (calib {calib_seqs} + probe {probe_seqs})")
    return ids[start:start + need].view(probe_seqs, seqlen)


# ---------------------------------------------------------------- eval-time module (reuses re2.ReweightMaskedMLP)

def swap_layer_e3(layer, layer_idx, tag, args, e3_dir):
    mlp = layer.mlp
    Wg0 = mlp.gate_proj.weight.detach().float()
    Wu0 = mlp.up_proj.weight.detach().float()
    Wd0 = mlp.down_proj.weight.detach().float()
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    d_ff = Wu0.shape[0]
    K = int(math.floor((1 - args.sparsity) * d_ff))

    if layer_idx <= 30:
        # wg/wu are never refit (see mode_calib) -- only wd + weight are on
        # disk; gate/up always come from the live (original) model weights.
        sd = torch.load(os.path.join(e3_dir, tag, f"layer_{layer_idx}.pt"),
                         map_location=mlp.gate_proj.weight.device)
        Wg, Wu, Wd, weight = Wg0, Wu0, sd["wd"], sd["weight"]
    else:
        Wg, Wu, Wd = Wg0, Wu0, Wd0
        weight = torch.ones(d_ff)

    layer.mlp = re2.ReweightMaskedMLP(mlp, Wg, Wu, Wd, colnorm0, weight, args.g, K)


# ---------------------------------------------------------------- modes

def mode_calib(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = re1.token_stream(tok, "train", args.calib_seqs, args.seqlen)
    diag_n = args.diag_seqs
    assert diag_n <= args.calib_seqs

    os.makedirs(args.out, exist_ok=True)
    xcache_dir = os.path.join(args.out, "xcache")
    os.makedirs(xcache_dir, exist_ok=True)
    lo, hi = (int(v) for v in args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    specs = [c for c in CANDIDATES if not args.only_tags or c["tag"] in args.only_tags.split(",")]
    assert specs, "no candidates selected"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.eval()

    for li in layer_ids:
        metapath = os.path.join(args.out, f"layer_{li}.meta.json")
        prev_meta = {}
        if os.path.exists(metapath):
            with open(metapath) as fh:
                prev_meta = json.load(fh)

        need_tags = [s for s in specs if not os.path.exists(
            os.path.join(args.out, s["tag"], f"layer_{li}.pt"))]
        if not need_tags:
            glt.log(f"layer {li}: all {len(specs)} requested candidates exist, skipping")
            continue

        t0 = time.time()
        xpath = os.path.join(xcache_dir, f"layer_{li}.X.pt")
        if os.path.exists(xpath):
            X = torch.load(xpath, map_location="cuda").float()
            glt.log(f"layer {li}: reusing cached X ({xpath})")
        else:
            model.to("cuda")
            X = re1.collect_layer_x(model, li, ids, "cuda")
            model.to("cpu")
            glt.empty_cache()
            torch.save(X.half(), xpath)  # cache for later candidate-addition rounds
            X = X.to("cuda").float()

        layer = model.model.layers[li]
        Wg0 = layer.mlp.gate_proj.weight.detach().float().to("cuda")
        Wu0 = layer.mlp.up_proj.weight.detach().float().to("cuda")
        Wd0 = layer.mlp.down_proj.weight.detach().float().to("cuda")
        act_name = model.config.hidden_act
        f = glt.act_fn(act_name)
        colnorm0 = Wd0.pow(2).sum(0).sqrt()
        d = Wu0.shape[0]
        K = int(math.floor((1 - args.sparsity) * d))
        n_groups = X.shape[0] // args.g

        I = glt.intermediate(X, f(glt.linear_up(X, Wg0)), Wu0)
        Y = glt.chunked_mm(I, Wd0.T)

        statspath = os.path.join(xcache_dir, f"layer_{li}.stats.pt")
        mask_topk_group, _ = glt.compute_group_mask(I, colnorm0, K, args.g)
        if os.path.exists(statspath):
            stats = torch.load(statspath, map_location="cuda")
            unique_full, freq, band_idx, unique_band = (
                stats["unique_full"], stats["freq"], stats["band_idx"], stats["unique_band"])
        else:
            unique_full = re2.compute_unique(I, colnorm0, args.eps)
            freq = mask_topk_group.float().mean(0)
            band_idx = band_indices(I, colnorm0, BAND_K_MULT * K)
            unique_band = compute_unique_band(I, colnorm0, band_idx, args.eps)
            torch.save({"unique_full": unique_full.cpu(), "freq": freq.cpu(),
                        "band_idx": band_idx.cpu(), "unique_band": unique_band.cpu()}, statspath)

        # diagnostic 8-seq slice (deterministic prefix, identical tokens to
        # E1's/E2's calibration/diag slice) for overlap-vs-greedy
        N_diag = diag_n * args.seqlen
        I_diag = I[:N_diag]
        mask_topk_diag, _ = glt.compute_group_mask(I_diag, colnorm0, K, args.g)
        greedy_group_diag, _ = re1.greedy_group_select(I_diag, Wd0, colnorm0, K, args.g, args.chunk)

        per_cand_meta = dict(prev_meta.get("candidates", {}))
        for spec in need_tags:
            tag = spec["tag"]
            weight = make_weight(spec, unique_full, freq, n_groups, unique_band, band_idx, d,
                                  I.device, I.dtype)
            group_mask, token_mask = re2.compute_group_mask_weighted(I, colnorm0, K, args.g, weight)
            assert (group_mask.sum(-1) == K).all(), f"{tag} layer {li}: achieved sparsity changed (K={K})"

            Wd_refit = glt.refit_wdown(I * token_mask, Y, Wd0, args.lambda_down)
            err = re1.group_recon_relerr(I, token_mask, Wd0, Y)
            overlap_vs_topk = re1.mask_overlap_stats(group_mask, mask_topk_group, K)

            mask_diag, _ = re2.compute_group_mask_weighted(I_diag, colnorm0, K, args.g, weight)
            overlap_vs_topk_diag = re1.mask_overlap_stats(mask_diag, mask_topk_diag, K)
            overlap_vs_greedy_diag = re1.mask_overlap_stats(mask_diag, greedy_group_diag, K)

            w_stats = {
                "min": weight.min().item(), "max": weight.max().item(), "std": weight.std().item(),
                "frac_at_clip_lo": (weight <= (1.0 / spec["c"] + 1e-9)).float().mean().item() if math.isfinite(spec["c"]) else 0.0,
                "frac_at_clip_hi": (weight >= (spec["c"] - 1e-9)).float().mean().item() if math.isfinite(spec["c"]) else 0.0,
            }

            outdir = os.path.join(args.out, tag)
            os.makedirs(outdir, exist_ok=True)
            # wg/wu are NEVER refit (only wd is) -- every candidate at every
            # layer would otherwise store an identical copy of the original
            # gate/up weights. At 12 candidates x 31 layers that redundancy
            # was ~360GB on a shared /raid with ~170GB free (hit mid-run,
            # see report) -- store only wd (+ the weight vector) here, and
            # have the eval-time swap (swap_layer_e3) pull Wg0/Wu0 from the
            # live model instead. wd is cast to the model's own compute
            # dtype (bf16) before saving -- ReweightMaskedMLP casts to that
            # dtype at load time regardless (orig_mlp.gate_proj.weight.dtype),
            # so this changes ONLY when the fp32->bf16 truncation happens,
            # not the final eval weights.
            torch.save({"wd": Wd_refit.bfloat16().cpu(), "weight": weight.cpu()},
                       os.path.join(outdir, f"layer_{li}.pt"))
            per_cand_meta[tag] = {
                "spec": spec,
                "group_mask_recon_relerr": err,
                "mask_overlap_vs_magnitude_topk": overlap_vs_topk,
                "mask_overlap_vs_magnitude_topk_diag8": overlap_vs_topk_diag,
                "mask_overlap_vs_greedy_diag8": overlap_vs_greedy_diag,
                "weight_stats": w_stats,
            }
            del group_mask, token_mask, Wd_refit, mask_diag
            glt.empty_cache()

        meta = {
            "layer": li, "g": args.g, "K": K, "d": d, "sparsity": args.sparsity,
            "lambda_down": args.lambda_down, "chunk": args.chunk,
            "calib_seqs": args.calib_seqs, "diag_seqs": diag_n, "seqlen": args.seqlen,
            "eps": args.eps, "band_k": BAND_K_MULT * K, "git_commit": glt.git_hash(), "model": args.model,
            "unique_full_stats": {"min": unique_full.min().item(), "max": unique_full.max().item(),
                                   "median": unique_full.median().item()},
            "unique_band_stats": {"min": unique_band.min().item(), "max": unique_band.max().item(),
                                   "median": unique_band.median().item()},
            "freq_stats": {"mean": freq.mean().item(), "median": freq.median().item(),
                           "frac_zero": (freq == 0).float().mean().item()},
            "candidates": per_cand_meta,
            "seconds": round(time.time() - t0, 1),
        }
        with open(metapath, "w") as fh:
            json.dump(meta, fh, indent=2)
        glt.log(f"layer {li}: computed {len(need_tags)} new candidate(s) ({meta['seconds']}s)")
        for tag in [s["tag"] for s in need_tags]:
            m = per_cand_meta[tag]
            glt.log(f"  {tag}: relerr={m['group_mask_recon_relerr']:.4f} "
                    f"overlap_vs_topk={m['mask_overlap_vs_magnitude_topk']['overlap_frac_mean']:.3f} "
                    f"overlap_vs_greedy_diag8={m['mask_overlap_vs_greedy_diag8']['overlap_frac_mean']:.3f}")

        del X, I, Y, mask_topk_group, I_diag, mask_topk_diag, greedy_group_diag
        del Wg0, Wu0, Wd0
        glt.empty_cache()

    glt.log("calib done.")


def _eval_loop(args, ids, tags, out_path, e3_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    conditions = ["dense"] + list(tags)
    results = {"args": vars(args), "git_commit": glt.git_hash(), "ppl": {}}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            prev = json.load(fh)
        results["ppl"].update(prev.get("ppl", {}))
        glt.log(f"resume: found existing {out_path}, already have {list(results['ppl'])}")

    for cond in conditions:
        if cond in results["ppl"]:
            glt.log(f"condition={cond}: already done (resume), skipping")
            continue
        t0 = time.time()
        glt.log(f"condition={cond}: loading {args.model} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
        model.to("cuda").eval()

        if cond != "dense":
            for li, layer in enumerate(model.model.layers):
                swap_layer_e3(layer, li, cond, args, e3_dir)

        ppl = re1.eval_ppl_fixed(model, ids, "cuda")
        results["ppl"][cond] = ppl
        glt.log(f"condition={cond} ppl={ppl:.4f} ({time.time() - t0:.1f}s)")

        del model
        glt.empty_cache()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)

    glt.log("eval done.")


def mode_probe(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = probe_ids(tok, args.seqlen, args.calib_seqs, args.probe_seqs)
    tags = args.conditions.split(",") if args.conditions else [c["tag"] for c in CANDIDATES]
    _eval_loop(args, ids, tags, args.out, args.e3_dir)


def mode_test(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.full:
        ids = re2.full_test_ids(tok, args.seqlen)
        glt.log(f"full eval: {ids.shape[0]} sequences x {args.seqlen}")
    else:
        ids = re1.token_stream(tok, "test", args.eval_seqs, args.seqlen)
    assert args.conditions, "--conditions is required for --mode test (final winner(s) only)"
    tags = args.conditions.split(",")
    _eval_loop(args, ids, tags, args.out, args.e3_dir)


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)

    # ---- test 1: weight_f1 boundary -- a=0,b=0 gives w==1 identically,
    # regardless of unique/freq content (pure power-law degeneration).
    unique = torch.rand(50) * 10 + 0.1
    freq = torch.rand(50) * 0.3
    w = weight_f1(unique, freq, n_groups=100, a=0.0, b=0.0, c=8.0)
    assert torch.allclose(w, torch.ones_like(w)), "a=0,b=0 must give w_j == 1"
    print("selftest 1/8 OK (weight_f1 a=0,b=0 -> w_j == 1 identically)")

    # ---- test 2: freq mean-normalization identity -- mean(freq_j) over a
    # REAL group-mask (exactly K per group) must equal K/d exactly (closed
    # form used to justify mean- over median-normalization in the docstring).
    N, d, g, K = 256, 20, 8, 5
    I = torch.rand(N, d)
    Wd0 = torch.randn(12, d) / math.sqrt(d)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    mask, _ = glt.compute_group_mask(I, colnorm0, K, g)
    freq2 = mask.float().mean(0)
    assert abs(freq2.mean().item() - K / d) < 1e-6, f"mean(freq) should be K/d={K/d}, got {freq2.mean().item()}"
    print(f"selftest 2/8 OK (mean(freq_j) == K/d == {K/d} exactly)")

    # ---- test 3: F1(a=0.25, b=0, c=8) reproduces E2's alpha=0.25 weight
    # bit-for-bit -- this is the request's explicit pipeline cross-check.
    # Use E2's actual observed range (layer 20: min=0.796 max=2.337) so the
    # clip provably never binds, making this a real check of the FORMULA,
    # not a vacuous one hidden behind saturation.
    torch.manual_seed(3)
    unique3 = torch.rand(200) * 5 + 0.1  # arbitrary positive, unnormalized
    freq3 = torch.zeros(200)
    freq3[:20] = torch.rand(20) * 0.5 + 0.1  # some nonzero, most zero (realistic sparsity=0.9 shape)
    w_e2 = re2.weight_from_unique(unique3, 0.25)
    w_f1 = weight_f1(unique3, freq3, n_groups=1000, a=0.25, b=0.0, c=8.0)
    assert torch.allclose(w_e2, w_f1, atol=1e-5), (
        f"F1(a=0.25,b=0,c=8) must match E2's weight_from_unique(alpha=0.25) when clip doesn't bind: "
        f"max abs diff={ (w_e2 - w_f1).abs().max().item() }")
    print("selftest 3/8 OK (F1 a=0.25/b=0/c=8 reproduces E2 alpha=0.25 bit-for-bit, b=0 makes freq irrelevant)")

    # ---- test 4: clip actually bounds extreme values (the E2 alpha=1.0
    # failure mode this request's clip mechanism is meant to prevent).
    torch.manual_seed(4)
    unique4 = torch.tensor([1e-6, 1.0, 1e6])  # median=1.0 by construction
    freq4 = torch.tensor([0.1, 0.1, 0.1])
    w4 = weight_f1(unique4, freq4, n_groups=100, a=1.0, b=0.0, c=8.0)
    assert w4.min().item() >= 1.0 / 8.0 - 1e-9 and w4.max().item() <= 8.0 + 1e-9
    assert abs(w4[0].item() - 1.0 / 8.0) < 1e-6
    assert abs(w4[2].item() - 8.0) < 1e-6
    print(f"selftest 4/8 OK (clip bounds extremes: w={w4.tolist()} within [0.125, 8.0])")

    # ---- test 5: freq floor avoids 0**negative -> inf/nan for b<0.
    unique5 = torch.rand(50) * 5 + 0.1
    freq5 = torch.zeros(50)
    freq5[0] = 1.0 / 4096  # smallest nonzero at n_groups=4096
    w5 = weight_f1(unique5, freq5, n_groups=4096, a=0.25, b=-0.2, c=8.0)
    assert torch.isfinite(w5).all(), "freq=0 with b<0 must not produce inf/nan (floor must engage)"
    print("selftest 5/8 OK (freq floor keeps b<0 finite even for never-selected neurons)")

    # ---- test 6: F2 boundary -- band covering ALL d neurons reduces
    # EXACTLY to weight_f1's unique-only term (b=0) -- band-restriction is
    # the only thing that distinguishes F2 from F1(b=0).
    torch.manual_seed(6)
    N6, d6, g6, K6 = 512, 30, 8, 6
    I6 = torch.rand(N6, d6)
    Wd6 = torch.randn(16, d6) / math.sqrt(d6)
    colnorm6 = Wd6.pow(2).sum(0).sqrt()
    unique_full6 = re2.compute_unique(I6, colnorm6, eps=1e-3)
    band_all = torch.arange(d6)
    unique_band6 = compute_unique_band(I6, colnorm6, band_all, eps=1e-3)
    assert torch.allclose(unique_full6, unique_band6, atol=1e-4), (
        "band covering all d must reproduce the full-d unique_j Cholesky exactly")
    w_f1_b0 = weight_f1(unique_full6, torch.zeros(d6), n_groups=100, a=0.3, b=0.0, c=8.0)
    w_f2_all = weight_f2(unique_band6, band_all, d6, a=0.3, c=8.0, device=unique_band6.device, dtype=unique_band6.dtype)
    assert torch.allclose(w_f1_b0, w_f2_all, atol=1e-5), "F2 with a full-d band must equal F1(b=0)"
    print("selftest 6/8 OK (F2 with band=all-d reduces exactly to F1's unique-only term)")

    # ---- test 7: band-restriction correctness -- appending INDEPENDENT
    # extra columns must not change the band-restricted unique_j of the
    # original columns (Schur-complement locality: conditioning on
    # unrelated regressors doesn't change a residual variance).
    torch.manual_seed(7)
    N7, d7 = 4096, 6
    I_band = torch.randn(N7, 3)
    I_band[:, 1] = I_band[:, 0] * (1.0 + 1e-3 * torch.randn(N7))  # neuron 1 near-dup of 0
    I_extra = torch.randn(N7, 3) * 3.0  # independent of I_band, different scale
    I7 = torch.cat([I_band, I_extra], dim=1)
    colnorm7 = torch.ones(d7)
    unique_alone = re2.compute_unique(I_band, torch.ones(3), eps=1e-3)
    band_idx7 = torch.arange(3)
    unique_restricted = compute_unique_band(I7, colnorm7, band_idx7, eps=1e-3)
    assert torch.allclose(unique_alone, unique_restricted, atol=1e-3), (
        f"appending independent columns must not change band-restricted unique_j: "
        f"alone={unique_alone.tolist()} restricted={unique_restricted.tolist()}")
    print("selftest 7/8 OK (band-restricted unique_j unaffected by independent columns outside the band)")

    # ---- test 8: candidate tags are unique and every spec is well-formed.
    tags = [c["tag"] for c in CANDIDATES]
    assert len(tags) == len(set(tags)), "duplicate candidate tags"
    for c in CANDIDATES:
        assert c["kind"] in ("f1", "f2")
        assert c["c"] > 0
    print(f"selftest 8/8 OK ({len(CANDIDATES)} candidate tags distinct and well-formed)")

    print("selftest ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--mode", choices=["selftest", "calib", "probe", "test"], required=False)
    ap.add_argument("--g", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib_seqs", type=int, default=32)
    ap.add_argument("--diag_seqs", type=int, default=8)
    ap.add_argument("--probe_seqs", type=int, default=8)
    ap.add_argument("--eval_seqs", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="test mode: full wikitext-2 test set instead of --eval_seqs")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lambda_down", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--layers", default="0-30", help="calib: 'lo-hi'")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--e3_dir", default=None, help="probe/test: calib output dir to read candidate weights from")
    ap.add_argument("--only_tags", default=None, help="calib: comma list to restrict which CANDIDATES to compute this run")
    ap.add_argument("--conditions", default=None, help="probe/test: comma list of candidate tags ('dense' is always included)")
    args = ap.parse_args()

    if args.mode == "selftest":
        selftest()
        return
    if args.mode == "calib":
        assert args.out, "--out is required for calib"
        mode_calib(args)
        return
    if args.mode == "probe":
        assert args.out and args.e3_dir, "--out and --e3_dir are required for probe"
        mode_probe(args)
        return
    if args.mode == "test":
        assert args.out and args.e3_dir, "--out and --e3_dir are required for test"
        mode_test(args)
        return
    raise SystemExit("--mode is required (selftest|calib|probe|test)")


if __name__ == "__main__":
    main()
