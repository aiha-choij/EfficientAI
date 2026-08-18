"""RAMS E2: deployable static decorrelation reweighting (topic:
refit-aware-mask-selection, follow-up to gslr_rams_e1.py / E1, auto/rams-e1,
PR #6). Reuses E1's Gram/mask/refit/eval machinery (gslr_layer_tune.py,
gslr_group_ppl.py, gslr_rams_e1.py) -- see this file's imports.

E1 established (report 20260818-230002-rams-e1.md): at fixed K, an offline
greedy (chunked matching-pursuit) group-mask selection beats magnitude top-K
by a large margin (B vs A: -30.1% PPL on the reduced eval set), and most of
that gain is from SELECTION alone, not refit synergy (B0 vs A0: -56.6%).
Greedy is diagnostic-only (non-deployable: it re-solves an 18-round
combinatorial search inside every masked forward). E2 asks: how much of that
headroom can a DEPLOYABLE mechanism recover -- one that keeps the exact
runtime shape of top-K (score correction, then one group-summed top-K; extra
cost = one elementwise multiply of the group score, no extra FLOPs order)?

Mechanism (static per-neuron decorrelation reweight):
  S~_j(T) = w_j * S_j(T),   S_j(T) = sum_{t in T} |i_tj| * ||W_down0[:,j]||
  (S_j(T) is exactly compute_group_mask's existing group-summed score.)
  w_j = (unique_j / median_j unique_j) ^ alpha

unique_j ("how much of neuron j's variance is NOT explainable by the other
neurons"): from calibration, accumulate the Gram of the GAUGE-SCALED
activation xi_tj = i_tj * ||W_down0[:,j]|| (same xi as GSLR's IRLS penalty
gauge, see gslr_layer_tune.py's module docstring) over ALL calibration
tokens (not per-group -- this is a property of the neuron, computed once):
  Ghat = sum_t xi_t xi_t^T   (d x d, fp32)
Then unique_j = 1 / [(Ghat/N + eps*mean(diag(Ghat/N))*I)^-1]_jj -- the
diagonal of the regularized PRECISION matrix's inverse-diagonal is exactly
"residual variance of neuron j after linearly regressing it on every other
neuron" (Schur-complement identity for Gaussian precision matrices): a
neuron well-predicted by (i.e. redundant with) others has a SMALL residual
variance, hence a small unique_j, hence w_j < 1 (downweighted). alpha=0
degenerates unique_j's effect away entirely (w_j == 1 for all j), which must
reproduce compute_group_mask's plain top-K mask bit-for-bit -- checked as a
runtime assertion in mode_calib and as CPU selftest 4.

Four alpha arms {0, 0.25, 0.5, 1.0} (alpha=0 is the boundary-sanity arm, see
above). Calibration is STANDARDIZED to 32x2048 (matching B1D/arm A's budget)
-- the request explicitly says not to repeat E1 arm B's 8-seq-calibration
disadvantage. The reduced EVAL set stays 8x2048, identical sequences to E1
(reusing gslr_rams_e1.token_stream verbatim) so E2's numbers sit in the same
table as E1's dense/A0/A/B0/B.

Mask-overlap diagnostics (vs magnitude top-K and vs E1's greedy mask) use a
SEPARATE, cheap 8-seq slice: the first 8 of the 32 calibration sequences
(token_stream's slicing is a deterministic prefix, so this 8-seq slice is
*exactly* E1's calibration slice bit-for-bit -- no extra data collection
needed, just index the already-collected 32-seq activations). E1's saved
greedy layer_i.pt files did not persist the raw boolean mask (only the
refit weights + summary overlap-vs-topK stats), so the greedy mask itself is
cheaply RE-DERIVED here via gslr_rams_e1.greedy_group_select on this 8-seq
slice (cost matches E1's original per-layer sanity/calib timing, ~4s/layer):
this is comparing MASKS, not re-deriving weights, so no re-fit is needed for
the diagnostic itself. The alpha-arm's weight w_j used for this diagnostic
is the SAME w_j fit from the full 32-seq calibration (mirrors what actually
happens at deployment: a fixed static weight applied live to whatever batch
shows up), not a separately-fit 8-seq weight.
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
import gslr_group_ppl as ggp  # noqa: E402 (unused directly, kept for parity w/ e1 imports)
import gslr_rams_e1 as re1  # noqa: E402 (token_stream, collect_layer_x, greedy_group_select, mask_overlap_stats, eval_ppl_fixed)

ALPHAS = [0.0, 0.25, 0.5, 1.0]


def alpha_tag(alpha):
    return f"a{int(round(alpha * 100)):03d}"  # 0.0->a000, 0.25->a025, 0.5->a050, 1.0->a100


# ---------------------------------------------------------------- core math

def compute_unique(I, colnorm0, eps=1e-3, chunk=8192):
    """unique_j = residual variance of neuron j's gauge-scaled activation
    xi_j = i_.,j * colnorm0_j after linearly regressing on every other
    neuron's xi, from calibration Gram Ghat = sum_t xi_t xi_t^T (I: N x d
    fp32, colnorm0: d fp32). One Cholesky factorization per layer
    (torch.cholesky_inverse reuses it for the full inverse; only the
    diagonal is used)."""
    N, d = I.shape
    V = I * colnorm0[None, :]
    Ghat = glt.chunked_matmul(V, V, chunk)  # d x d fp32
    A = Ghat / N
    meandiag = A.diagonal().mean()
    A = A + eps * meandiag * torch.eye(d, device=A.device, dtype=A.dtype)
    L = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(L)
    unique = 1.0 / Ainv.diagonal().clamp(min=1e-12)
    return unique  # (d,) fp32


def weight_from_unique(unique, alpha):
    med = unique.median().clamp(min=1e-12)
    return (unique / med).clamp(min=1e-12).pow(alpha)


def compute_group_mask_weighted(I, colnorm0, K, g, weight):
    """Same score rule as gslr_layer_tune.compute_group_mask, but the
    group-summed score is MULTIPLIED by the static per-neuron weight before
    top-K (weight=ones must reduce to compute_group_mask exactly -- checked
    at alpha=0 and in selftest)."""
    N, d = I.shape
    score = I.abs() * colnorm0[None, :]
    gscore = score.view(-1, g, d).sum(1)
    gscore = gscore * weight[None, :]
    idx = gscore.topk(K, dim=-1).indices
    group_mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
    token_mask = group_mask.repeat_interleave(g, 0)
    return group_mask, token_mask


# ---------------------------------------------------------------- eval-time module

class ReweightMaskedMLP(torch.nn.Module):
    """Deployable E2 eval condition: live group top-K masking exactly like
    MaskedMLP, but the group score is multiplied by a static (calibration-
    fit, frozen at eval time) per-neuron weight before top-K. Runtime cost
    over plain top-K: one extra (d,)-broadcast elementwise multiply of the
    group score -- no extra FLOPs order, matches the request's 'deployable'
    framing (unlike GreedyMaskedMLP, which re-solves an 18-round search
    every forward and is explicitly non-deployable)."""

    def __init__(self, orig_mlp, Wg, Wu, Wd, colnorm0, weight, g, K):
        super().__init__()
        self.act_fn = orig_mlp.act_fn
        dtype = orig_mlp.gate_proj.weight.dtype
        dev = orig_mlp.gate_proj.weight.device
        self.gate_proj = torch.nn.Linear(Wg.shape[1], Wg.shape[0], bias=False, dtype=dtype, device=dev)
        self.up_proj = torch.nn.Linear(Wu.shape[1], Wu.shape[0], bias=False, dtype=dtype, device=dev)
        self.down_proj = torch.nn.Linear(Wd.shape[1], Wd.shape[0], bias=False, dtype=dtype, device=dev)
        with torch.no_grad():
            self.gate_proj.weight.copy_(Wg.to(dtype))
            self.up_proj.weight.copy_(Wu.to(dtype))
            self.down_proj.weight.copy_(Wd.to(dtype))
        self.register_buffer("colnorm0", colnorm0.float().to(dev))
        self.register_buffer("weight", weight.float().to(dev))
        self.g = g
        self.K = K

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        i = gate * up
        bsz, seqlen, d = i.shape
        assert bsz == 1, "group masking assumes bs=1 (matches calibration convention)"
        assert seqlen % self.g == 0
        i2 = i.view(seqlen, d)
        score = i2.float().abs() * self.colnorm0[None, :]
        gscore = score.view(-1, self.g, d).sum(1) * self.weight[None, :]
        idx = gscore.topk(self.K, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
        tok_mask = mask.repeat_interleave(self.g, 0).unsqueeze(0)
        return self.down_proj(i * tok_mask)


def swap_layer_e2(layer, layer_idx, alpha, args, e2_dir):
    mlp = layer.mlp
    Wg0 = mlp.gate_proj.weight.detach().float()
    Wu0 = mlp.up_proj.weight.detach().float()
    Wd0 = mlp.down_proj.weight.detach().float()
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    d_ff = Wu0.shape[0]
    K = int(math.floor((1 - args.sparsity) * d_ff))

    if layer_idx <= 30:
        tag = alpha_tag(alpha)
        sd = torch.load(os.path.join(e2_dir, tag, f"layer_{layer_idx}.pt"),
                         map_location=mlp.gate_proj.weight.device)
        Wg, Wu, Wd, weight = sd["wg"], sd["wu"], sd["wd"], sd["weight"]
    else:
        Wg, Wu, Wd = Wg0, Wu0, Wd0
        weight = torch.ones(d_ff)

    layer.mlp = ReweightMaskedMLP(mlp, Wg, Wu, Wd, colnorm0, weight, args.g, K)


# ---------------------------------------------------------------- modes

def mode_calib(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = re1.token_stream(tok, "train", args.calib_seqs, args.seqlen)  # 32 seqs, standard budget
    diag_n = args.diag_seqs  # 8, must be <= calib_seqs and a prefix (deterministic slicing)
    assert diag_n <= args.calib_seqs

    os.makedirs(args.out, exist_ok=True)
    lo, hi = (int(v) for v in args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.eval()

    for li in layer_ids:
        tags = [alpha_tag(a) for a in ALPHAS]
        metapath = os.path.join(args.out, f"layer_{li}.meta.json")
        done = os.path.exists(metapath) and all(
            os.path.exists(os.path.join(args.out, t, f"layer_{li}.pt")) for t in tags)
        if done:
            glt.log(f"layer {li}: exists, skipping (resume)")
            continue

        t0 = time.time()
        model.to("cuda")
        X = re1.collect_layer_x(model, li, ids, "cuda")
        model.to("cpu")
        glt.empty_cache()
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

        I = glt.intermediate(X, f(glt.linear_up(X, Wg0)), Wu0)
        Y = glt.chunked_mm(I, Wd0.T)

        unique = compute_unique(I, colnorm0, args.eps)
        unique_stats = {
            "min": unique.min().item(), "max": unique.max().item(),
            "median": unique.median().item(), "mean": unique.mean().item(),
        }

        # magnitude top-K mask on the FULL 32-seq calibration set, reused as
        # the alpha=0 sanity reference and as the overlap-vs-magnitude baseline
        mask_topk_group, _ = glt.compute_group_mask(I, colnorm0, K, args.g)

        # diagnostic 8-seq slice (deterministic prefix of the 32-seq calib
        # set == E1's exact greedy calibration slice) + fresh greedy mask
        N_diag = diag_n * args.seqlen
        I_diag = I[:N_diag]
        mask_topk_diag, _ = glt.compute_group_mask(I_diag, colnorm0, K, args.g)
        greedy_group_diag, _ = re1.greedy_group_select(I_diag, Wd0, colnorm0, K, args.g, args.chunk)

        per_alpha_meta = {}
        for alpha in ALPHAS:
            tag = alpha_tag(alpha)
            weight = weight_from_unique(unique, alpha)
            group_mask, token_mask = compute_group_mask_weighted(I, colnorm0, K, args.g, weight)

            if alpha == 0.0:
                assert torch.equal(group_mask, mask_topk_group), (
                    f"layer {li}: alpha=0 must degenerate to plain top-K bit-for-bit")

            Wd_refit = glt.refit_wdown(I * token_mask, Y, Wd0, args.lambda_down)
            err = re1.group_recon_relerr(I, token_mask, Wd0, Y)
            overlap_vs_topk = re1.mask_overlap_stats(group_mask, mask_topk_group, K)

            # diagnostic-scale mask (same static weight, applied to the 8-seq
            # slice) for overlap vs magnitude-topk-on-diag and vs greedy-on-diag
            mask_diag, _ = compute_group_mask_weighted(I_diag, colnorm0, K, args.g, weight)
            overlap_vs_topk_diag = re1.mask_overlap_stats(mask_diag, mask_topk_diag, K)
            overlap_vs_greedy_diag = re1.mask_overlap_stats(mask_diag, greedy_group_diag, K)

            w_stats = {
                "min": weight.min().item(), "max": weight.max().item(),
                "std": weight.std().item(),
                "frac_lt_0.5": (weight < 0.5).float().mean().item(),
                "frac_gt_2": (weight > 2.0).float().mean().item(),
            }

            outdir = os.path.join(args.out, tag)
            os.makedirs(outdir, exist_ok=True)
            torch.save({"wg": Wg0.cpu(), "wu": Wu0.cpu(), "wd": Wd_refit.cpu(),
                        "weight": weight.cpu()}, os.path.join(outdir, f"layer_{li}.pt"))
            per_alpha_meta[tag] = {
                "alpha": alpha,
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
            "eps": args.eps, "git_commit": glt.git_hash(), "model": args.model,
            "unique_stats": unique_stats,
            "alphas": per_alpha_meta,
            "seconds": round(time.time() - t0, 1),
        }
        with open(metapath, "w") as fh:
            json.dump(meta, fh, indent=2)
        glt.log(f"layer {li}: unique median={unique_stats['median']:.4g} "
                f"({meta['seconds']}s)")
        for tag, m in per_alpha_meta.items():
            glt.log(f"  {tag}: relerr={m['group_mask_recon_relerr']:.4f} "
                    f"overlap_vs_topk={m['mask_overlap_vs_magnitude_topk']['overlap_frac_mean']:.3f} "
                    f"overlap_vs_greedy_diag8={m['mask_overlap_vs_greedy_diag8']['overlap_frac_mean']:.3f}")

        del X, I, Y, unique, mask_topk_group, I_diag, mask_topk_diag, greedy_group_diag
        del Wg0, Wu0, Wd0
        glt.empty_cache()

    glt.log("calib done.")


def full_test_ids(tok, seqlen):
    """Full wikitext-2 test set, same tokenization/assembly convention as
    gslr_rams_e1.token_stream (so it differs only trivially, if at all, from
    the project's established full-set PPL convention in larosa/utils/
    eval_ppl.py -- both concatenate the raw test split and slice into
    non-overlapping seqlen chunks; using token_stream here keeps E2's
    reduced and full evals on one consistent pipeline). n_seqs = floor(total
    tokens / seqlen), no reshuffling."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids_flat = tok(text, return_tensors="pt").input_ids[0]
    n = ids_flat.numel() // seqlen
    return re1.token_stream(tok, "test", n, seqlen)


def mode_eval(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.full:
        ids = full_test_ids(tok, args.seqlen)
        glt.log(f"full eval: {ids.shape[0]} sequences x {args.seqlen}")
    else:
        ids = re1.token_stream(tok, "test", args.eval_seqs, args.seqlen)

    conditions = args.conditions.split(",")
    results = {"args": vars(args), "git_commit": glt.git_hash(), "ppl": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh)
        results["ppl"].update(prev.get("ppl", {}))
        glt.log(f"resume: found existing {args.out}, already have {list(results['ppl'])}")

    for cond in conditions:
        if cond in results["ppl"]:
            glt.log(f"condition={cond}: already done (resume), skipping")
            continue
        t0 = time.time()
        glt.log(f"condition={cond}: loading {args.model} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
        model.to("cuda").eval()

        if cond == "dense":
            pass
        elif cond.startswith("e2_"):
            alpha = {"e2_a000": 0.0, "e2_a025": 0.25, "e2_a050": 0.5, "e2_a100": 1.0}[cond]
            for li, layer in enumerate(model.model.layers):
                swap_layer_e2(layer, li, alpha, args, args.e2_dir)
        else:
            raise ValueError(cond)

        ppl = re1.eval_ppl_fixed(model, ids, "cuda")
        results["ppl"][cond] = ppl
        glt.log(f"condition={cond} ppl={ppl:.4f} ({time.time() - t0:.1f}s)")

        del model
        glt.empty_cache()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)

    glt.log("eval done.")


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)

    # ---- test 1: alpha=0 -> weight is exactly all-ones, regardless of unique.
    unique = torch.rand(50) * 10 + 0.1
    w0 = weight_from_unique(unique, 0.0)
    assert torch.allclose(w0, torch.ones_like(w0)), "alpha=0 must give w_j == 1 for all j"
    print("selftest 1/5 OK (alpha=0 -> w_j == 1 identically)")

    # ---- test 2: compute_group_mask_weighted(weight=ones) reduces EXACTLY
    # (bit-for-bit) to compute_group_mask -- the deployed degeneration this
    # request's boundary sanity depends on.
    N, h, d, g = 128, 10, 16, 8
    I = torch.rand(N, d)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    K = 5
    ones = torch.ones(d)
    gm_ref, tm_ref = glt.compute_group_mask(I, colnorm0, K, g)
    gm_w, tm_w = compute_group_mask_weighted(I, colnorm0, K, g, ones)
    assert torch.equal(gm_ref, gm_w) and torch.equal(tm_ref, tm_w)
    print("selftest 2/5 OK (weight=ones reduces to compute_group_mask bit-for-bit)")

    # ---- test 3: redundancy detection -- a neuron that's a near-exact linear
    # copy of another must get a SMALLER unique_j than an independent neuron
    # of comparable magnitude (this is the core mechanism the whole request
    # rests on: unique_j should downweight redundant neurons).
    torch.manual_seed(1)
    N2, d2 = 4096, 10
    I2 = torch.randn(N2, d2)
    I2[:, 1] = I2[:, 0] + 0.01 * torch.randn(N2)  # neuron 1 ~ near-copy of neuron 0
    colnorm2 = torch.ones(d2)
    unique2 = compute_unique(I2, colnorm2, eps=1e-3)
    indep_unique = unique2[2:].mean()
    assert unique2[0] < 0.3 * indep_unique and unique2[1] < 0.3 * indep_unique, (
        f"redundant pair must have much smaller unique_j: "
        f"unique[0]={unique2[0]:.4f} unique[1]={unique2[1]:.4f} indep_mean={indep_unique:.4f}")
    print(f"selftest 3/5 OK (redundant pair unique_j={unique2[0]:.4f}/{unique2[1]:.4f} "
          f"<< independent mean={indep_unique:.4f})")

    # ---- test 4: alpha>0 actually changes selection in the expected
    # direction -- downweighting a redundant, otherwise-dominant neuron can
    # knock it out of the top-K in favor of a less-redundant one.
    #
    # Construction notes (this failed on the first attempt with near-constant
    # "activations" -- worth recording why, since it's a real property of the
    # request's UNCENTERED Gram formula, not just a test bug): Ghat = sum_t
    # xi_t xi_t^T is the RAW second moment, not a covariance -- it is NOT
    # mean-subtracted. Two columns with a similar large MEAN look highly
    # "predictable from each other" under this Gram even with independent
    # per-token fluctuations (mean_i*mean_j dominates the cross term), so
    # near-constant big-mean columns spuriously read as mutually redundant
    # regardless of true correlation. Using zero-mean, large-VARIANCE columns
    # instead (mean 0, big std) avoids that confound and isolates genuine
    # fluctuation-correlation, which is what unique_j is meant to detect (and
    # is realistic for real MLP intermediate activations, which are not
    # near-constant). Separately, mixing in small-variance "background"
    # columns skews the median low (small variance alone drags unique_j down,
    # independent of redundancy) -- so the background columns here are given
    # variance comparable to the tested columns, so the median reflects a
    # genuine "typical, non-redundant" scale.
    torch.manual_seed(2)
    N3, d3, g3 = 2048, 8, 2048  # single group (g3==N3) averages out per-token noise
    I3 = torch.empty(N3, d3)
    I3[:, 0] = 5.0 * torch.randn(N3)                       # dominant, mean 0
    I3[:, 1] = I3[:, 0] * (1.0 + 1e-3 * torch.randn(N3))   # near-duplicate of 0 (correlated fluctuation)
    I3[:, 2] = 4.5 * torch.randn(N3)                       # independent, almost-as-dominant
    for j in range(3, d3):
        I3[:, j] = 1.0 * torch.randn(N3)                   # background, independent, comparable-order variance
    colnorm3 = torch.ones(d3)
    K3 = 2
    gm_plain, _ = glt.compute_group_mask(I3, colnorm3, K3, g3)
    assert gm_plain[:, 0].all() and gm_plain[:, 1].all(), "plain top-K should pick the duplicated pair (0,1)"
    unique3 = compute_unique(I3, colnorm3, eps=1e-3)
    w3 = weight_from_unique(unique3, alpha=1.0)
    gm_w3, _ = compute_group_mask_weighted(I3, colnorm3, K3, g3, w3)
    assert gm_w3[:, 2].any(), (
        f"reweighted top-K should recover the independent neuron 2 over the redundant pair "
        f"(unique={unique3.tolist()}, weight={w3.tolist()})")
    print(f"selftest 4/5 OK (alpha=1.0 knocks a redundant duplicate out of top-K in favor of an "
          f"independent neuron; unique={[round(u, 5) for u in unique3.tolist()]})")

    # ---- test 5: alpha_tag round-trips the 4 arms distinctly.
    tags = [alpha_tag(a) for a in ALPHAS]
    assert tags == ["a000", "a025", "a050", "a100"] and len(set(tags)) == 4
    print("selftest 5/5 OK (alpha_tag distinct for all 4 arms)")

    print("selftest ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--mode", choices=["selftest", "calib", "eval"], required=False)
    ap.add_argument("--g", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib_seqs", type=int, default=32)
    ap.add_argument("--diag_seqs", type=int, default=8)
    ap.add_argument("--eval_seqs", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="eval: use the full wikitext-2 test set instead of --eval_seqs")
    ap.add_argument("--chunk", type=int, default=64, help="greedy-diagnostic chunk size (matches E1)")
    ap.add_argument("--lambda_down", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-3, help="unique_j precision-matrix ridge, relative to mean(diag)")
    ap.add_argument("--layers", default="0-30", help="calib: 'lo-hi'")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--e2_dir", default=None, help="calib output dir / eval input dir")
    ap.add_argument("--conditions", default="dense,e2_a000,e2_a025,e2_a050,e2_a100")
    args = ap.parse_args()

    if args.mode == "selftest":
        selftest()
        return
    if args.mode == "calib":
        assert args.out, "--out is required for calib"
        mode_calib(args)
        return
    if args.mode == "eval":
        assert args.out, "--out is required for eval"
        assert args.e2_dir, "--e2_dir is required for eval"
        mode_eval(args)
        return
    raise SystemExit("--mode is required (selftest|calib|eval)")


if __name__ == "__main__":
    main()
