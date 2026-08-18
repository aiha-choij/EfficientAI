"""RAMS E1: refit-aware mask selection headroom diagnostic (topic:
refit-aware-mask-selection, plan doc plans/refit-aware-mask-selection-handoff-
2026-08-18.md on the research wiki; builds on gslr-stage1/2/2.5, PR #5,
auto/gslr-stage1 @ c126276).

RAMS hypothesis: the deployed group mask rule (top-K of the group-summed
gauge score S_j(T) = sum_{t in T} |i_tj|*||W_down0[:,j]||) picks CORRELATED/
redundant neurons together, because it is a pure magnitude criterion with no
awareness of what a downstream anchored-ridge refit (B1D) can already
reconstruct from a differently-chosen set. E1 asks ONLY: does headroom exist
at all for a better same-K selection? It measures an upper bound via an
offline, non-deployable greedy selection -- if the bound is close to the
current top-K+refit result (B1D), the mask-selection axis is dead and E2/E3
(deployable reweighting) should not be attempted.

Three arms, all g=16, sparsity=0.9 (K=floor(0.1*d)), layers 0..30 retuned /
31 always original weights (masked same as everyone else) -- mirrors B1D's
should_use_retuned convention exactly (see gslr_group_ppl.py):
  A  (baseline)  magnitude top-K mask (compute_group_mask, unchanged) +
                 B1D's existing refit (reused verbatim from
                 ~/workspace/gslr25/llama2-7b/b1d/16 -- NOT re-fit here; a
                 sanity condition re-derives it from the SAME calibration
                 convention and checks equivalence, see --mode calib_a0check)
  B0 (selection-only)  greedy mask, ORIGINAL (never refit) weights -- isolates
                 how much of B's gain (if any) is the selection change alone,
                 vs. requiring refit synergy.
  B  (headroom)  greedy mask, refit AGAINST THE GREEDY MASK (mask-rule /
                 weight consistency: refitting B1D's weights under a
                 different mask would unfairly penalize greedy, since the
                 anchored ridge never saw that mask during calibration).

Greedy selection design (--mode calib/sanity, function greedy_group_select):
this is deliberately NOT literal OMP with an incremental Cholesky-updated
per-atom coefficient refit. Rationale: an OMP coefficient refit would choose
the K-set that's optimal for COEFFICIENTS THAT DON'T EXIST at deployment --
the deployed masked forward always multiplies neuron j's contribution by the
REAL activation i_tj, never a refit value, so scoring candidates under a
hypothetical refit coefficient would silently answer a different question
("what if we could also reweight i") than E1 asks ("same i, better which-K").
Instead: chunked greedy / matching-pursuit with FIXED, EXACT atom
contributions v_j = i_.,j (x) W_down0[:,j] (outer product over the group's g
tokens). Each round (of ceil(K/chunk) rounds, chunk=64):
  1. correlate the CURRENT EXACT residual R (group's dense target minus the
     exact contribution of everything selected so far) against every not-yet-
     selected neuron: crit_j = 2*<R, v_j> - ||v_j||^2 (the exact reduction in
     ||residual||^2 from subtracting v_j, since v_j's "coefficient" is fixed
     at 1 -- this is NOT |<R,v_j>| alone, which would be the criterion for a
     freely-scalable atom).
  2. take the top `chunk` by crit_j, add to the selected set.
  3. subtract their EXACT (not approximate/re-orthogonalized) contribution
     from R.
Because R is always exact (never an approximation), a neuron highly
correlated with an already-selected one has automatically-suppressed crit_j
next round -- the RAMS-hypothesized "redundant co-selection" failure mode of
plain top-K is directly addressed without needing true orthogonalization.
Chunking (add 64/round, not 1) is a stated, reported cost control (request
section "비용 통제"); so is using full d each round rather than a magnitude-
prefiltered candidate pool (I measured the candidate-gather approach OOMs at
this group-batch size -- see report -- and dropped it; correlating over full
d costs the same matmul FLOPs as prefiltering to d/2..d/4 would only save a
roughly proportional fraction, not an order of magnitude, so the simpler/
safer full-d version was kept).

Calibration/eval token budgets (both REDUCED from B1D's 32-seq calibration
to 8x2048, matching the request's reduced eval set) -- also a reported cost
control: greedy's per-round correlation cost scales linearly in group count,
and 32-seq calibration x 31 layers x 17 rounds was not tractable on a shared
single A100 in this diagnostic's time budget (see report for the measured
5-layer sanity timing this was set from). This makes arm B's refit
calibration SMALLER-N than arm A's B1D (65536 vs 16384 tokens) -- flagged as
a caveat, not expected to matter much since 16384 tokens >> K=1100 free
columns of a 4096-dim ridge regression, but not independently verified here.

Eval-time masking for b0/b reuses gslr_group_ppl.py's MaskedMLP class shape
(same weight-copy / colnorm0-buffer plumbing) via GreedyMaskedMLP, but calls
greedy_group_select instead of a live top-K -- computed PER FORWARD CALL,
batch size forced to 1 (matches MaskedMLP's own bs=1 assumption, kept
unmodified rather than patched, to avoid diverging from the reused harness).
This makes b0/b's eval an actual live, per-layer-compounded forward exactly
like a0/gslr -- NOT a shortcut -- but is expensive (an offline greedy solve
inside every one of the 8 eval sequences' forward passes, all 31 masked
layers) and is explicitly NOT a deployable inference path (stated in the
request; repeated here for anyone re-running this script)."""

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
import gslr_group_ppl as ggp  # noqa: E402 (reused MaskedMLP/swap_layer/should_use_retuned)


# ---------------------------------------------------------------- data

def token_stream(tok, split, n_seqs, seqlen):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n_seqs * seqlen
    assert ids.numel() >= need, f"{split}: {ids.numel()} < {need} tokens"
    return ids[:need].view(n_seqs, seqlen)


# ---------------------------------------------------------------- greedy core

def greedy_group_select(I, Wd0, colnorm0, K, g, chunk=64, mm_dtype=torch.bfloat16):
    """See module docstring. I (N x d) fp32, N a multiple of g, group
    boundaries assumed aligned to N (caller must ensure sequence boundaries
    never split a group -- true whenever each concatenated sequence's length
    is itself a multiple of g, e.g. seqlen=2048, g=16).
    Wd0 (h x d), colnorm0 (d,): ALWAYS the ORIGINAL frozen down_proj / its
    column norms (anti-circularity, matches compute_group_mask's gauge
    convention -- never the arm's own possibly-refit Wd).
    Returns group_mask (n_groups x d) bool, token_mask (N x d) bool -- same
    shape/semantics as gslr_layer_tune.compute_group_mask, drop-in
    comparable (mask stats, metrics(), refit_wdown all accept either)."""
    device = I.device
    N, d = I.shape
    assert N % g == 0
    n_groups = N // g
    h = Wd0.shape[0]
    Ig = I.view(n_groups, g, d)

    WdT = Wd0.T.contiguous()  # d x h
    Y = torch.einsum("ngd,hd->ngh", Ig, Wd0)  # dense group target (fp32)
    R = Y.clone()
    normsq = Ig.pow(2).sum(1) * colnorm0.pow(2)[None, :]  # n_groups x d, cheap

    selected = torch.zeros(n_groups, d, dtype=torch.bool, device=device)
    n_rounds = math.ceil(K / chunk)
    WdT_mm = WdT.to(mm_dtype)
    for r in range(n_rounds):
        take = min(chunk, K - r * chunk)
        rWd = torch.einsum("ngh,dh->ngd", R.to(mm_dtype), WdT_mm).float()  # n_groups x g x d
        corr = (Ig * rWd).sum(1)  # n_groups x d
        crit = (2 * corr - normsq).masked_fill(selected, float("-inf"))
        idx = crit.topk(take, dim=-1).indices
        newsel = torch.zeros_like(selected).scatter_(1, idx, True)
        selected = selected | newsel
        # sparse residual update: only `take` (<<d) columns are new this round,
        # so gather them instead of a full-d matmul (this is the same exact
        # contribution as Ig*newsel @ WdT, just without multiplying the
        # ~d-take zero columns).
        Ic_new = torch.gather(Ig, 2, idx.unsqueeze(1).expand(-1, g, -1))  # n_groups x g x take
        WdN = WdT[idx]  # n_groups x take x h
        contrib = torch.einsum("ngt,nth->ngh", Ic_new, WdN)
        R = R - contrib

    group_mask = selected
    token_mask = group_mask.repeat_interleave(g, 0)
    return group_mask, token_mask


def group_recon_relerr(I, token_mask, Wd0, Y):
    """Same formula as gslr_layer_tune.metrics()'s group_mask_recon_relerr_g*."""
    ynorm = Y.pow(2).sum().sqrt()
    return (((I * token_mask) @ Wd0.T - Y).pow(2).sum().sqrt() / ynorm).item()


def mask_overlap_stats(mask_a, mask_b, K):
    """mask_a, mask_b: n_groups x d bool. Per-group Jaccard-style overlap
    (intersection / K, both masks select exactly K per group so intersection
    <= K) and union size (as a multiple of K), averaged over groups."""
    inter = (mask_a & mask_b).sum(-1).float()
    union = (mask_a | mask_b).sum(-1).float()
    return {
        "overlap_frac_mean": (inter / K).mean().item(),   # 1.0 = identical masks
        "union_over_k_mean": (union / K).mean().item(),   # 1.0 = identical, 2.0 = disjoint
        "n_groups": mask_a.shape[0],
    }


# ---------------------------------------------------------------- eval-time module

class GreedyMaskedMLP(torch.nn.Module):
    """b0/b eval condition: same weight-copy plumbing as gslr_group_ppl.
    MaskedMLP, but the mask is computed live via greedy_group_select instead
    of top-K. Selection ALWAYS uses the ORIGINAL frozen Wd0/colnorm0 (passed
    in separately from self.down_proj, which may be b's refit Wd) -- same
    anti-circularity rule as top-K."""

    def __init__(self, orig_mlp, Wg, Wu, Wd, Wd0_orig, colnorm0, g, K, chunk=64):
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
        self.register_buffer("Wd0_orig", Wd0_orig.float().to(dev))
        self.register_buffer("colnorm0", colnorm0.float().to(dev))
        self.g = g
        self.K = K
        self.chunk = chunk

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        i = gate * up
        bsz, seqlen, d = i.shape
        assert bsz == 1, "greedy group selection assumes bs=1 (matches MaskedMLP's convention)"
        assert seqlen % self.g == 0
        i2 = i.view(seqlen, d).float()
        _, tok_mask = greedy_group_select(i2, self.Wd0_orig, self.colnorm0, self.K, self.g, self.chunk)
        return self.down_proj(i * tok_mask.unsqueeze(0))


def swap_layer_greedy(layer, layer_idx, condition, args, greedy_dir):
    """condition in {"b0", "b"}. b0: original weights everywhere, greedy
    mask. b: layers 0..30 use greedy-refit Wd (from --greedy_dir/layer_i.pt),
    layer 31 (or any layer without a saved refit) stays original -- mirrors
    gslr_group_ppl.should_use_retuned exactly."""
    mlp = layer.mlp
    Wg0 = mlp.gate_proj.weight.detach().float()
    Wu0 = mlp.up_proj.weight.detach().float()
    Wd0 = mlp.down_proj.weight.detach().float()
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    d_ff = Wu0.shape[0]
    K = int(math.floor((1 - args.sparsity) * d_ff))

    if condition == "b" and layer_idx <= 30:
        sd = torch.load(os.path.join(greedy_dir, f"layer_{layer_idx}.pt"),
                         map_location=mlp.gate_proj.weight.device)
        Wg, Wu, Wd = sd["wg"], sd["wu"], sd["wd"]
    else:
        Wg, Wu, Wd = Wg0, Wu0, Wd0

    layer.mlp = GreedyMaskedMLP(mlp, Wg, Wu, Wd, Wd0, colnorm0, args.g, K, args.chunk)


# ---------------------------------------------------------------- fixed-set PPL

def eval_ppl_fixed(model, ids, device):
    """PPL over a FIXED (n_seqs x seqlen) token tensor, one sequence per
    forward call (bs=1, matches MaskedMLP/GreedyMaskedMLP's assumption).
    Same nll/ppl formula as larosa/utils/eval_ppl.py:eval_ppl_wikitext,
    restricted to exactly ids.shape[0] sequences instead of the full test
    split -- this IS the "reduced eval set" the request specifies."""
    n_seqs, seqlen = ids.shape
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []
    with torch.no_grad():
        for i in range(n_seqs):
            inputs = ids[i:i + 1].to(device)
            logits = model(inputs).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = inputs[:, 1:]
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (n_seqs * seqlen))
    return ppl.item()


# ---------------------------------------------------------------- modes

def collect_layer_x(model, layer_idx, ids, device):
    layer = model.model.layers[layer_idx]
    captured = []
    hook = layer.mlp.register_forward_pre_hook(
        lambda mod, inp: captured.append(inp[0].detach().squeeze(0).to(torch.float16).cpu()))
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i:i + 1].to(device))
    hook.remove()
    return torch.cat(captured, 0)


def mode_sanity(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = token_stream(tok, "train", args.calib_seqs, args.seqlen)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.to("cuda").eval()

    layers = [int(v) for v in args.layers.split(",")]
    results = {}
    for li in layers:
        t0 = time.time()
        layer = model.model.layers[li]
        Wg0 = layer.mlp.gate_proj.weight.detach().float()
        Wu0 = layer.mlp.up_proj.weight.detach().float()
        Wd0 = layer.mlp.down_proj.weight.detach().float()
        act_name = model.config.hidden_act
        f = glt.act_fn(act_name)
        colnorm0 = Wd0.pow(2).sum(0).sqrt()

        X = collect_layer_x(model, li, ids, "cuda").to("cuda").float()
        I = glt.intermediate(X, f(glt.linear_up(X, Wg0)), Wu0)
        Y = glt.chunked_mm(I, Wd0.T)
        d = Wu0.shape[0]
        K = int(math.floor((1 - args.sparsity) * d))

        _, mask_topk = glt.compute_group_mask(I, colnorm0, K, args.g)
        _, mask_greedy = greedy_group_select(I, Wd0, colnorm0, K, args.g, args.chunk)

        err_topk = group_recon_relerr(I, mask_topk, Wd0, Y)
        err_greedy = group_recon_relerr(I, mask_greedy, Wd0, Y)
        gsec = time.time() - t0
        passed = err_greedy <= err_topk + 1e-6
        results[li] = {
            "group_mask_recon_relerr_topk": err_topk,
            "group_mask_recon_relerr_greedy": err_greedy,
            "greedy_better_or_equal": passed,
            "seconds": round(gsec, 1),
        }
        glt.log(f"layer {li}: topk={err_topk:.4f} greedy={err_greedy:.4f} "
                f"{'OK' if passed else 'FAIL'} ({gsec:.1f}s)")
        del X, I, Y, mask_topk, mask_greedy
        glt.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"args": vars(args), "git_commit": glt.git_hash(), "results": results}, fh, indent=2)
    all_ok = all(r["greedy_better_or_equal"] for r in results.values())
    glt.log(f"sanity {'ALL OK' if all_ok else 'FAILED'} -- see {args.out}")


def mode_calib(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = token_stream(tok, "train", args.calib_seqs, args.seqlen)

    os.makedirs(args.out, exist_ok=True)
    lo, hi = (int(v) for v in args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.eval()

    for li in layer_ids:
        outpath = os.path.join(args.out, f"layer_{li}.pt")
        metapath = os.path.join(args.out, f"layer_{li}.meta.json")
        if os.path.exists(outpath) and os.path.exists(metapath):
            glt.log(f"layer {li}: exists, skipping (resume)")
            continue
        t0 = time.time()
        model.to("cuda")
        X = collect_layer_x(model, li, ids, "cuda")
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

        group_topk, mask_topk = glt.compute_group_mask(I, colnorm0, K, args.g)
        group_greedy, mask_greedy = greedy_group_select(I, Wd0, colnorm0, K, args.g, args.chunk)

        Wd_refit = glt.refit_wdown(I * mask_greedy, Y, Wd0, args.lambda_down)

        err_topk = group_recon_relerr(I, mask_topk, Wd0, Y)
        err_greedy = group_recon_relerr(I, mask_greedy, Wd0, Y)
        overlap = mask_overlap_stats(group_topk, group_greedy, K)

        torch.save({"wg": Wg0.cpu(), "wu": Wu0.cpu(), "wd": Wd_refit.cpu()}, outpath)
        meta = {
            "layer": li, "g": args.g, "K": K, "d": d, "sparsity": args.sparsity,
            "lambda_down": args.lambda_down, "chunk": args.chunk,
            "calib_seqs": args.calib_seqs, "seqlen": args.seqlen,
            "git_commit": glt.git_hash(), "model": args.model,
            "group_mask_recon_relerr_topk": err_topk,
            "group_mask_recon_relerr_greedy": err_greedy,
            "greedy_better_or_equal": err_greedy <= err_topk + 1e-6,
            "mask_overlap_vs_topk": overlap,
            "seconds": round(time.time() - t0, 1),
        }
        with open(metapath, "w") as fh:
            json.dump(meta, fh, indent=2)
        glt.log(f"layer {li}: topk_relerr={err_topk:.4f} greedy_relerr={err_greedy:.4f} "
                f"overlap={overlap['overlap_frac_mean']:.3f} union/K={overlap['union_over_k_mean']:.3f} "
                f"({meta['seconds']}s)")
        del X, I, Y, mask_topk, mask_greedy, group_topk, group_greedy, Wd_refit
        del Wg0, Wu0, Wd0
        glt.empty_cache()

    glt.log("calib done.")


def mode_eval(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = token_stream(tok, "test", args.eval_seqs, args.seqlen)

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
        elif cond in ("a0", "a"):
            ggp_cond = "a0" if cond == "a0" else "gslr"
            for li, layer in enumerate(model.model.layers):
                ggp.swap_layer(layer, li, ggp_cond, args, args.b1d_dir)
        elif cond in ("b0", "b"):
            for li, layer in enumerate(model.model.layers):
                swap_layer_greedy(layer, li, cond, args, args.greedy_dir)
        else:
            raise ValueError(cond)

        ppl = eval_ppl_fixed(model, ids, "cuda")
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

    # ---- test 1: K=d (dense) -> greedy selects everything, matches
    # compute_group_mask(K=d) exactly (both degenerate to "select all").
    N, h, d, g = 128, 10, 16, 8
    I = torch.rand(N, d)
    Wd0 = torch.randn(h, d) / math.sqrt(d)
    colnorm0 = Wd0.pow(2).sum(0).sqrt()
    group_topk, _ = glt.compute_group_mask(I, colnorm0, d, g)
    group_greedy, _ = greedy_group_select(I, Wd0, colnorm0, d, g, chunk=3)
    assert group_topk.all() and group_greedy.all(), "K=d must select every neuron in every group"
    print("selftest 1/4 OK (K=d degenerates to select-all for both rules)")

    # ---- test 2: exact-K count, every round (chunk doesn't evenly divide K)
    K = 5
    _, tok_mask = greedy_group_select(I, Wd0, colnorm0, K, g, chunk=3)
    gm, _ = greedy_group_select(I, Wd0, colnorm0, K, g, chunk=3)
    assert (gm.sum(-1) == K).all(), "every group must select exactly K neurons"
    print("selftest 2/4 OK (exact K per group even when chunk doesn't divide K)")

    # ---- test 3: definitional sanity -- on activations with a DELIBERATELY
    # duplicated (perfectly correlated) high-magnitude neuron pair, greedy's
    # group reconstruction error must be <= magnitude top-K's (top-K has no
    # mechanism to avoid picking a near-duplicate; greedy's exact residual
    # subtraction does). This is the same inequality the real sanity step
    # (--mode sanity) checks on real LLaMA activations.
    torch.manual_seed(1)
    N2, h2, d2, g2 = 64, 12, 20, 8
    I2 = torch.rand(N2, d2) * 0.1
    I2[:, 0] = 5.0 + 0.01 * torch.rand(N2)   # dominant neuron
    I2[:, 1] = I2[:, 0] * (1.0 + 1e-3 * torch.randn(N2))  # near-duplicate of neuron 0
    Wd2 = torch.randn(h2, d2) / math.sqrt(d2)
    Wd2[:, 1] = Wd2[:, 0] * (1.0 + 1e-3 * torch.randn(h2))  # duplicate's column is also near-identical
    colnorm2 = Wd2.pow(2).sum(0).sqrt()
    K2 = 3
    _, mtopk = glt.compute_group_mask(I2, colnorm2, K2, g2)
    _, mgreedy = greedy_group_select(I2, Wd2, colnorm2, K2, g2, chunk=1)
    Y2 = I2 @ Wd2.T
    e_topk = group_recon_relerr(I2, mtopk, Wd2, Y2)
    e_greedy = group_recon_relerr(I2, mgreedy, Wd2, Y2)
    assert e_greedy <= e_topk + 1e-6, (
        f"greedy must not be worse than top-K on a redundant-neuron construction: "
        f"greedy={e_greedy:.5f} topk={e_topk:.5f}")
    print(f"selftest 3/4 OK (redundant-neuron case: greedy={e_greedy:.5f} <= topk={e_topk:.5f})")

    # ---- test 4: mask_overlap_stats sanity -- identical masks -> overlap=1,
    # union/K=1; disjoint masks -> overlap=0, union/K=2.
    m1, _ = glt.compute_group_mask(I, colnorm0, 4, g)
    stats_same = mask_overlap_stats(m1, m1, 4)
    assert abs(stats_same["overlap_frac_mean"] - 1.0) < 1e-6
    assert abs(stats_same["union_over_k_mean"] - 1.0) < 1e-6
    a = torch.zeros(2, d, dtype=torch.bool); a[:, :4] = True
    b = torch.zeros(2, d, dtype=torch.bool); b[:, 4:8] = True
    stats_disjoint = mask_overlap_stats(a, b, 4)
    assert abs(stats_disjoint["overlap_frac_mean"] - 0.0) < 1e-6
    assert abs(stats_disjoint["union_over_k_mean"] - 2.0) < 1e-6
    print("selftest 4/4 OK (mask_overlap_stats: identical->1.0/1.0, disjoint->0.0/2.0)")

    print("selftest ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--mode", choices=["selftest", "sanity", "calib", "eval"], required=False)
    ap.add_argument("--g", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib_seqs", type=int, default=8)
    ap.add_argument("--eval_seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lambda_down", type=float, default=1.0)
    ap.add_argument("--layers", default="0,8,16,24,30", help="sanity: comma list; calib: 'lo-hi'")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--greedy_dir", default=None, help="calib output dir / eval input dir for arm B")
    ap.add_argument("--b1d_dir", default=os.path.expanduser("~/workspace/gslr25/llama2-7b/b1d/16"),
                     help="existing B1D layer_i.pt dir, reused verbatim for arm A")
    ap.add_argument("--conditions", default="dense,a0,a,b0,b")
    args = ap.parse_args()
    args.gslr_dir = args.b1d_dir  # gslr_group_ppl.swap_layer reads args.gslr_dir

    if args.mode == "selftest":
        selftest()
        return
    if args.mode == "sanity":
        mode_sanity(args)
        return
    if args.mode == "calib":
        assert args.out, "--out is required for calib"
        mode_calib(args)
        return
    if args.mode == "eval":
        assert args.out, "--out is required for eval"
        assert args.greedy_dir, "--greedy_dir is required for eval (arm b)"
        mode_eval(args)
        return
    raise SystemExit("--mode is required (selftest|sanity|calib|eval)")


if __name__ == "__main__":
    main()
