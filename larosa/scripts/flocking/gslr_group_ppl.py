"""GSLR stage-2, step 2: full-model wikitext-2 test PPL under group top-K
masking (topic: groupwise-flocking-tuning). Answers stage 2's judging
question: does gslr_multilayer_tune.py's retuning actually lower PPL versus
the untuned model, under the SAME group mask?

Four conditions (see stage-2 request table):
  dense       -- no masking, original weights everywhere. Sanity check: must
                 reproduce the repo's existing dense reference (~5.4736,
                 larosa/scripts/repro_topk_ppl.sh convention) since this
                 script's eval loop (eval_ppl_wikitext) is imported verbatim
                 from larosa/utils/eval_ppl.py, unmodified.
  a0          -- group top-K mask, ORIGINAL weights everywhere (untuned
                 baseline -- this is the denominator stage 2 judges against).
  gslr        -- group top-K mask, RETUNED weights (from gslr_multilayer_tune
                 .py's --out dir) for layers 0..30; layer 31 stays original
                 weights but IS masked (same masking, only weights differ --
                 per stage-2 spec, layer 31 was excluded from retuning, not
                 from masking).
  gslr_dense  -- RETUNED weights for layers 0..30, NO masking (diagnostic:
                 how much did retuning alone perturb the dense forward path,
                 independent of the masking judgment).

Masking convention matches gslr_layer_tune.py/gslr_multilayer_tune.py's
compute_group_mask exactly (same gauge score, same group definition), so
group-level statistics computed there (overlap, maskrecon, etc.) and PPL
measured here are directly comparable:
  score_tj = |i_tj| * ||W_down0[:,j]||_2   -- W_down0 is always the ORIGINAL
             (untuned) down_proj column norm, frozen for the whole run. This
             is the SAME anti-circularity requirement as training: the score
             gauge must not move when the weights being scored change,
             otherwise "top-K selection" and "what got tuned" chase each
             other in a circle.
  groups = g CONSECUTIVE tokens within one context_size-token sequence
           (never across sequence boundaries -- context_size must be a
           multiple of g, matches calibration seqlen convention).
  mask = top-K per group, K = floor((1-sparsity)*d).
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

import gslr_layer_tune as glt  # noqa: E402  (same dir as this script)
import gslr3_sketch as g3s  # noqa: E402  (stage-3 compensation-branch phi builders)
from utils.eval_ppl import eval_ppl_wikitext  # noqa: E402


class MaskedMLP(torch.nn.Module):
    """Drop-in replacement for LlamaMLP that applies a group top-K gauge-score
    mask to the intermediate activation before down_proj. apply_mask=False
    reproduces the plain dense forward exactly (used for the gslr_dense
    diagnostic condition).

    bonus (d,), optional (stage-2.5 GSLR B2 "reuse bonus"): added to the
    group score before Top-K, exactly mirroring
    gslr_layer_tune.compute_group_mask's bonus semantics -- this is what
    lets a B2-trained layer_i.pt (which stores its bonus alongside wg/wu/wd)
    be evaluated with the SAME mask rule it was fitted under. bonus=None
    (default, and what every B0/B1D/B1 layer_i.pt has) reproduces the exact
    plain top-K rule -- zero behavior change for those conditions."""

    def __init__(self, orig_mlp, Wg, Wu, Wd, colnorm0, g, K, apply_mask, bonus=None):
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
        if bonus is not None:
            self.register_buffer("bonus", bonus.float().to(dev))
        else:
            self.bonus = None
        self.g = g
        self.K = K
        self.apply_mask = apply_mask

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        i = gate * up
        if not self.apply_mask:
            return self.down_proj(i)
        bsz, seqlen, d = i.shape
        assert bsz == 1, "group masking assumes bs=1 (matches training-time calibration convention)"
        assert seqlen % self.g == 0, f"seqlen {seqlen} not divisible by group size {self.g}"
        i2 = i.view(seqlen, d)
        score = i2.float().abs() * self.colnorm0[None, :]
        gscore = score.view(-1, self.g, d).sum(1)
        if self.bonus is not None:
            gscore = gscore + self.bonus[None, :]
        idx = gscore.topk(self.K, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
        tok_mask = mask.repeat_interleave(self.g, 0).unsqueeze(0)
        return self.down_proj(i * tok_mask)


class CompMaskedMLP(MaskedMLP):
    """Stage-3 GSLR (topic: groupwise-flocking-tuning, follow-up to
    gslr-stage2.5 / 69f4f63): MaskedMLP's group top-K masked forward, plus a
    jointly-fit compensation branch Theta @ phi_t for the DROPPED (masked-
    out) neurons' contribution -- see gslr3_tune.py for how (Wd, Theta) and
    phi's own factors (pca_mu/pca_P for C1, or nothing extra for C2/C2a --
    their sketch factors are a deterministic function of the ORIGINAL,
    never-refit Wg0/Wu0/Wd0 + r_sk, rebuilt here rather than persisted per
    layer_i.pt) are fit. comp_kind in {"c1","c2","c2a"} selects phi's
    definition -- MUST exactly mirror gslr3_sketch.py's compute_c2_phi /
    apply_pca (this is the only other place phi is computed; any drift
    between training-time and eval-time phi would silently change what PPL
    measures)."""

    def __init__(self, orig_mlp, Wg, Wu, Wd, colnorm0, g, K, comp_kind, theta,
                 Wg0_orig, Wu0_orig, Wd0_orig, r_sk=None, pca_mu=None, pca_P=None):
        super().__init__(orig_mlp, Wg, Wu, Wd, colnorm0, g, K, apply_mask=True)
        self.comp_kind = comp_kind
        dev = self.down_proj.weight.device
        self.register_buffer("theta", theta.float().to(dev))
        if comp_kind == "c1":
            self.register_buffer("pca_mu", pca_mu.float().to(dev))
            self.register_buffer("pca_P", pca_P.float().to(dev))
        else:
            Ag, Bg, Au, Bu, Ad, Bd = g3s.build_c2_sketch(
                Wg0_orig.to(dev), Wu0_orig.to(dev), Wd0_orig.to(dev), r_sk)
            self.register_buffer("Ag", Ag)
            self.register_buffer("Bg", Bg)
            self.register_buffer("Au", Au)
            self.register_buffer("Bu", Bu)
            self.register_buffer("Ad", Ad)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        i = gate * up
        bsz, seqlen, d = i.shape
        assert bsz == 1, "group masking assumes bs=1 (matches training-time calibration convention)"
        assert seqlen % self.g == 0, f"seqlen {seqlen} not divisible by group size {self.g}"
        i2 = i.view(seqlen, d)
        x2 = x.float().view(seqlen, -1)
        score = i2.float().abs() * self.colnorm0[None, :]
        gscore = score.view(-1, self.g, d).sum(1)
        idx = gscore.topk(self.K, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
        tok_mask = mask.repeat_interleave(self.g, 0)
        base = self.down_proj(i * tok_mask.unsqueeze(0))

        if self.comp_kind == "c1":
            phi = (x2 - self.pca_mu[None, :]) @ self.pca_P.T
        else:
            ghat = self.act_fn((x2 @ self.Ag.T) @ self.Bg.T)
            if self.comp_kind == "c2a":
                uhat = up.float().view(seqlen, d)  # exact up-projection (diagnostic only)
            else:
                uhat = (x2 @ self.Au.T) @ self.Bu.T
            tail = (1.0 - tok_mask.float()) * (ghat * uhat)
            phi = tail @ self.Ad.T
        comp = (phi @ self.theta.T).unsqueeze(0)
        return base + comp.to(base.dtype)


def should_use_retuned(condition, layer_idx, only_layers=None):
    """Layer 31 (or any layer beyond what was retuned) always keeps its
    original weights, regardless of condition -- masking still applies to
    it under a0/gslr, only the WEIGHTS differ across conditions.

    only_layers (optional set of ints, GSLR-CSEQ E1 D0 addition): restricts
    which layers may use retuned weights to this subset (e.g. {16} for the
    D0 single-layer deployment diagnostic -- layer 16 retuned, layers 0-15
    and 17-30 stay at ORIGINAL weights even though gslr_dir may contain
    layer_i.pt for all of them, e.g. when gslr_dir points straight at an
    existing 31-layer B1D/A2 directory). only_layers=None (default) is the
    exact pre-existing behavior (every layer 0-30 eligible) -- zero
    behavior change for A0/B1D/B1/B2/stage-3 callers."""
    if not (condition in ("gslr", "gslr_dense") and layer_idx <= 30):
        return False
    if only_layers is not None:
        return layer_idx in only_layers
    return True


def swap_layer(layer, layer_idx, condition, args, gslr_dir, only_layers=None):
    mlp = layer.mlp
    Wg0 = mlp.gate_proj.weight.detach().float()
    Wu0 = mlp.up_proj.weight.detach().float()
    Wd0 = mlp.down_proj.weight.detach().float()
    colnorm0 = Wd0.pow(2).sum(0).sqrt()  # gauge: always the ORIGINAL W_down, frozen
    d_ff = Wu0.shape[0]
    K = int(math.floor((1 - args.sparsity) * d_ff))

    bonus = None
    theta = None
    sd = None
    if should_use_retuned(condition, layer_idx, only_layers):
        sd = torch.load(os.path.join(gslr_dir, f"layer_{layer_idx}.pt"),
                         map_location=mlp.gate_proj.weight.device)
        Wg, Wu, Wd = sd["wg"], sd["wu"], sd["wd"]
        bonus = sd.get("bonus")  # only B2 layer_i.pt files carry this key
        theta = sd.get("theta")  # only stage-3 C1/C2/C2a layer_i.pt files carry this key
    else:
        Wg, Wu, Wd = Wg0, Wu0, Wd0

    apply_mask = condition in ("a0", "gslr")
    if theta is not None and condition == "gslr":
        # stage-3 GSLR: compensation branch only applies under masking (the
        # gslr_dense diagnostic intentionally ignores it -- see
        # CompMaskedMLP's docstring / gslr3_tune.py's arm design note).
        meta = json.load(open(os.path.join(gslr_dir, f"layer_{layer_idx}.meta.json")))
        layer.mlp = CompMaskedMLP(
            mlp, Wg, Wu, Wd, colnorm0, args.g, K, meta["comp_kind"], theta,
            Wg0, Wu0, Wd0, r_sk=meta.get("r_sk"), pca_mu=sd.get("pca_mu"), pca_P=sd.get("pca_P"))
    else:
        layer.mlp = MaskedMLP(mlp, Wg, Wu, Wd, colnorm0, args.g, K, apply_mask, bonus=bonus)


def selftest():
    torch.manual_seed(0)

    class DummyMLP(torch.nn.Module):
        def __init__(self, h, d):
            super().__init__()
            self.act_fn = torch.nn.SiLU()
            self.gate_proj = torch.nn.Linear(h, d, bias=False)
            self.up_proj = torch.nn.Linear(h, d, bias=False)
            self.down_proj = torch.nn.Linear(d, h, bias=False)

    h, d, g, seqlen, K = 6, 10, 2, 8, 4
    orig = DummyMLP(h, d)
    Wg = orig.gate_proj.weight.detach().clone()
    Wu = orig.up_proj.weight.detach().clone()
    Wd = orig.down_proj.weight.detach().clone()
    colnorm0 = Wd.pow(2).sum(0).sqrt()

    mm = MaskedMLP(orig, Wg, Wu, Wd, colnorm0, g, K, apply_mask=True)
    x = torch.randn(1, seqlen, h)
    with torch.no_grad():
        out = mm(x)
        gate = mm.act_fn(mm.gate_proj(x))
        up = mm.up_proj(x)
        i = (gate * up).view(seqlen, d)
        score = i.abs() * colnorm0[None, :]
        gscore = score.view(-1, g, d).sum(1)
        assert gscore.shape[0] == seqlen // g
        idx = gscore.topk(K, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
        for p in range(gscore.shape[0]):
            assert mask[p].sum().item() == K, "each group must select exactly K neurons"
        tok_mask = mask.repeat_interleave(g, 0).unsqueeze(0)
        expected = mm.down_proj(i.view(1, seqlen, d) * tok_mask)
    assert torch.allclose(out, expected, atol=1e-6), "MaskedMLP output != brute-force recomputed mask"
    print("selftest 1/5 OK (group mask: exactly K per group, matches brute-force recompute)")

    mm_dense = MaskedMLP(orig, Wg, Wu, Wd, colnorm0, g, K, apply_mask=False)
    with torch.no_grad():
        out_dense = mm_dense(x)
        gate = mm_dense.act_fn(mm_dense.gate_proj(x))
        up = mm_dense.up_proj(x)
        expected_dense = mm_dense.down_proj(gate * up)
    assert torch.allclose(out_dense, expected_dense, atol=1e-6)
    print("selftest 2/5 OK (apply_mask=False reproduces unmasked dense forward exactly)")

    assert should_use_retuned("gslr", 0) is True
    assert should_use_retuned("gslr", 30) is True
    assert should_use_retuned("gslr", 31) is False, "layer 31 must never use retuned weights"
    assert should_use_retuned("gslr_dense", 31) is False
    assert should_use_retuned("a0", 5) is False, "a0 condition never uses retuned weights, any layer"
    assert should_use_retuned("dense", 5) is False
    print("selftest 3/5 OK (should_use_retuned: layer>30 and non-gslr conditions excluded)")

    # ---- GSLR-CSEQ E1 D0: only_layers restricts retuned-weight eligibility
    # to a subset (e.g. the single-layer deployment diagnostic) without
    # touching the None (all layers 0-30) default behavior above.
    assert should_use_retuned("gslr", 16, only_layers={16}) is True
    assert should_use_retuned("gslr", 15, only_layers={16}) is False
    assert should_use_retuned("gslr", 17, only_layers={16}) is False
    assert should_use_retuned("gslr_dense", 16, only_layers={16}) is True
    assert should_use_retuned("gslr", 31, only_layers={16, 31}) is False, \
        "layer>30 exclusion still applies even if listed in only_layers"
    assert should_use_retuned("a0", 16, only_layers={16}) is False, \
        "only_layers must not resurrect a0 (a0 never uses retuned weights)"
    print("selftest 3b/5 OK (only_layers=None unchanged; only_layers={16} restricts to layer 16, "
          "layer>30 exclusion and a0-never-retuned still hold)")

    # ---- stage-2.5 B2: bonus=None must be a strict no-op (verified above,
    # test 1 never passes bonus); a strong bonus on one neuron must force it
    # selected every group, matching gslr_layer_tune.compute_group_mask's
    # bonus semantics exactly (same "add to gscore before top-K" rule).
    bonus = torch.zeros(d)
    bonus[0] = 1e6
    mm_bonus = MaskedMLP(orig, Wg, Wu, Wd, colnorm0, g, K, apply_mask=True, bonus=bonus)
    with torch.no_grad():
        out_bonus = mm_bonus(x)
        gate = mm_bonus.act_fn(mm_bonus.gate_proj(x))
        up = mm_bonus.up_proj(x)
        i = (gate * up).view(seqlen, d)
        score = i.abs() * colnorm0[None, :]
        gscore = score.view(-1, g, d).sum(1) + bonus[None, :]
        idx = gscore.topk(K, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
        assert mask[:, 0].all(), "a dominant bonus on neuron 0 must force it selected in every group"
        tok_mask = mask.repeat_interleave(g, 0).unsqueeze(0)
        expected_bonus = mm_bonus.down_proj(i.view(1, seqlen, d) * tok_mask)
    assert torch.allclose(out_bonus, expected_bonus, atol=1e-6), \
        "MaskedMLP with bonus != brute-force recompute with the same bonus added to gscore"
    assert not torch.allclose(out_bonus, out, atol=1e-6), "a strong bonus should change MaskedMLP's output"
    print("selftest 4/5 OK (B2 bonus: forces dominant neuron selected, matches brute-force gscore+bonus)")

    # ---- stage-3 GSLR: CompMaskedMLP's compensation branch must brute-
    # force match a manual mask+phi+theta recompute, for both comp_kind
    # (C1 PCA and C2 sketch-tail), and (sanity) a Theta of all zeros must
    # exactly reproduce plain MaskedMLP's masked-only output.
    Wg0o, Wu0o, Wd0o = Wg.clone(), Wu.clone(), Wd.clone()
    r = 3
    theta = torch.randn(h, r) * 0.1

    pca_mu = torch.randn(h)
    pca_P = torch.randn(r, h)
    pca_P = pca_P / pca_P.norm(dim=1, keepdim=True)
    mm_c1 = CompMaskedMLP(orig, Wg, Wu, Wd, colnorm0, g, K, "c1", theta,
                           Wg0o, Wu0o, Wd0o, pca_mu=pca_mu, pca_P=pca_P)
    with torch.no_grad():
        out_c1 = mm_c1(x)
        gate = mm_c1.act_fn(mm_c1.gate_proj(x))
        up = mm_c1.up_proj(x)
        i = (gate * up).view(seqlen, d)
        score = i.abs() * colnorm0[None, :]
        gscore = score.view(-1, g, d).sum(1)
        idx = gscore.topk(K, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, idx, True)
        tok_mask = mask.repeat_interleave(g, 0)
        base = mm_c1.down_proj(i.view(1, seqlen, d) * tok_mask.unsqueeze(0))
        phi = (x.view(seqlen, h) - pca_mu[None, :]) @ pca_P.T
        expected_c1 = base + (phi @ theta.T).unsqueeze(0)
    assert torch.allclose(out_c1, expected_c1, atol=1e-5), \
        "CompMaskedMLP (c1) != brute-force mask+PCA-phi+theta recompute"

    r_sk = 4
    theta_c2 = torch.randn(h, r_sk) * 0.1
    Ag, Bg, Au, Bu, Ad, Bd = g3s.build_c2_sketch(Wg0o, Wu0o, Wd0o, r_sk)
    mm_c2 = CompMaskedMLP(orig, Wg, Wu, Wd, colnorm0, g, K, "c2", theta_c2,
                           Wg0o, Wu0o, Wd0o, r_sk=r_sk)
    with torch.no_grad():
        out_c2 = mm_c2(x)
        ghat = mm_c2.act_fn((x.view(seqlen, h) @ Ag.T) @ Bg.T)
        uhat = (x.view(seqlen, h) @ Au.T) @ Bu.T
        tail = (1.0 - tok_mask.float()) * (ghat * uhat)
        phi2 = tail @ Ad.T
        expected_c2 = base + (phi2 @ theta_c2.T).unsqueeze(0)
    assert torch.allclose(out_c2, expected_c2, atol=1e-4), \
        "CompMaskedMLP (c2) != brute-force mask+sketch-phi+theta recompute"

    mm_c2_zero = CompMaskedMLP(orig, Wg, Wu, Wd, colnorm0, g, K, "c2", torch.zeros(h, r_sk),
                                Wg0o, Wu0o, Wd0o, r_sk=r_sk)
    with torch.no_grad():
        out_c2_zero = mm_c2_zero(x)
    assert torch.allclose(out_c2_zero, base, atol=1e-5), \
        "CompMaskedMLP with theta=0 must reproduce plain masked-only output exactly"
    print("selftest 5/5 OK (CompMaskedMLP: c1/c2 match brute-force mask+phi+theta recompute, "
          "theta=0 == plain masked output)")

    print("selftest ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--g", type=int, required=False)
    ap.add_argument("--lambda_rel", type=float, default=None,
                     help="metadata only -- names which gslr_dir config this run measures")
    ap.add_argument("--gslr_dir", default=None,
                     help="dir with layer_{i}.pt from gslr_multilayer_tune.py (required for gslr/gslr_dense)")
    ap.add_argument("--conditions", default="dense,a0,gslr,gslr_dense")
    ap.add_argument("--only_layers", default=None,
                     help="GSLR-CSEQ E1 D0: comma-separated layer indices allowed to use "
                          "retuned weights under gslr/gslr_dense (e.g. '16' for the "
                          "single-layer deployment diagnostic); default (unset) = all "
                          "layers 0-30, matching prior behavior")
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--context_size", type=int, default=2048)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    assert args.out, "--out is required"
    conditions = args.conditions.split(",")
    needs_gslr = any(c in ("gslr", "gslr_dense") for c in conditions)
    if needs_gslr:
        assert args.gslr_dir, "--gslr_dir is required for gslr/gslr_dense conditions"
    if any(c in ("a0", "gslr") for c in conditions):
        assert args.g, "--g is required for a0/gslr conditions (mask group size)"
    only_layers = (set(int(v) for v in args.only_layers.split(","))
                   if args.only_layers else None)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    results = {"args": vars(args), "git_commit": glt.git_hash(), "ppl": {}}

    for cond in conditions:
        t0 = time.time()
        glt.log(f"condition={cond}: loading {args.model} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
        model.to("cuda").eval()

        if cond != "dense":
            for li, layer in enumerate(model.model.layers):
                swap_layer(layer, li, cond, args, args.gslr_dir, only_layers=only_layers)

        with torch.no_grad():
            ppl = eval_ppl_wikitext(model, tok, "cuda", dataset=dataset,
                                     context_size=args.context_size)
        results["ppl"][cond] = ppl
        glt.log(f"condition={cond} ppl={ppl:.4f} ({time.time() - t0:.1f}s)")

        del model
        glt.empty_cache()
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)

    glt.log("done.")


if __name__ == "__main__":
    main()
