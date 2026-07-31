# Experiment: refit-l0l1-validate-3b

Status: DONE (2026-07-31)
Date: 2026-07-31

## Hypothesis tested
Single verification point (request spec "실행 순서"): at s=0.9, g=1, does a
closed-form refit of down_proj (L1) recover a clear share of the PPL lost
to freezing a C2-score top-K mask (L0), with no other repair mechanism?
Confirm before submitting the full matrix.

## What we're testing over alternatives
Cheapest possible signal: one (s,g) point, smallest available Llama-3.x
model, PPL only (no lm-eval harness yet).

## Prior art check
No local-loss-refit journal history yet (topic init same day). Anchors
carried from oracle-residual-sparsity (LLaMA2-7B, gateway A100, Phase-3
gate): dense 5.4736, C1-topk 5.5210/5.7296/8.1083 at s=0.5/0.7/0.9 — not
directly comparable here (different model family/scale + instruct tuning),
used only as a sanity reference for "is 90% masking supposed to hurt a
lot".

## Expected outcome
Go if ΔL1 = PPL(L1) − PPL(L0) is a clear improvement, well outside the
~1e-3 backend-noise floor oracle-residual-sparsity established. No-go/null
if ΔL1 is negligible (masking error would then be mostly token-idiosyncratic,
not something a fixed linear refit of down_proj can absorb).

## Reproducibility
- Git tag: none yet (branch auto/local-loss-refit, commit 52b09bd at run
  time — see RESULT_JSON `git_commit` field in both result JSONs).
- Job ID: 050-20260731-103000-refit-l0l1-validate-3b
- Host/GPU: a100-40-2 (pinned via -H, this gateway itself has 4 usable
  A100-40GB + a display GPU), 1 GPU >=15GiB
- Command: `env PY=/home/choij/miniconda3/envs/larosa/bin/python bash
  larosa/scripts/refit/run_validate_l0l1.sh /raid/LLM/llama3.2-3b-instruct
  /home/choij/workspace/refit/llama3.2-3b-instruct`
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a100-40-2;
  this filesystem is shared across the cluster's execution hosts via
  /raid, confirmed by all prior topics' journal notes using identical
  paths regardless of dispatched host)
- Config: s=0.9, g=1, lambda=0.01, calibration wikitext103 (c4 streaming
  left untested per oracle-residual-sparsity's Open Questions — avoided
  for a first run), nsamples=512, seqlen=2048, seed=42
- Key deps: gateway conda env `~/miniconda3/envs/larosa`, transformers
  4.46.3, torch 2.6.0+cu124
- Model deviation (documented, not silently substituted): request spec
  asked for a plain pretrained Llama-3.2-3B for the dev pass; only the
  **instruct** checkpoint (`llama3.2-3b-instruct`) is present locally at
  /raid/LLM. Used as-is for this speed check (relative L0-vs-L1 delta is
  the readout, not absolute PPL) — confirmed compatible first via a CPU-only
  structural smoke test (real config, rope_type=llama3, tied embeddings,
  3.2B params, no GPU time spent) before submitting.
- Unit tests at tag: test_refit_units.py (5/5 pass) + test_oracle_units.py
  regression (unchanged, still passes) — both run directly on CPU, not
  via qsub (not GPU experiments).

### Results
(artifact: job log at ~/workspace/runs/20260731-103000-refit-l0l1-validate-3b/log;
result JSONs under ~/workspace/refit/llama3.2-3b-instruct/results/;
elapsed 15 min including model load x3, calibration, 2 PPL evals)

| condition | s | g | PPL | achieved sparsity |
|---|---|---|---|---|
| L0 | 0.9 | 1 | 21.5859 | 0.9000 |
| L1 | 0.9 | 1 | 19.9510 | 0.9000 |

ΔL1 = L1 − L0 = **−1.635 PPL** (≈ −7.6% relative), same achieved sparsity
(mask is identical between L0 and L1 by construction — only the down_proj
weight differs). This is far outside any plausible noise floor for a
single-seed PPL run (compare to oracle-residual-sparsity's ~1e-3 backend/
arch noise floor on a similarly-scaled eval).

Absolute PPL (19–22) is much higher than the LLaMA2-7B anchors (5.5–8.1) —
expected: different model family/scale, instruct-tuned checkpoint evaluated
on raw wikitext-2 continuation (not its tuned distribution), and this
dev-pass calibration corpus (wikitext103) differs from what the LLaMA2-7B
anchors used. Absolute numbers are not comparable across these; ΔL1 (same
model, same mask, same eval) is the valid readout, per the request spec's
own primary-metric definition.

Eval-log caveat carried over from oracle_mlp's known label-swap bug (gist
"Pitfalls" item 7): printed "mlp h1/h2" lines are always 0 and "attn h2"
shows the true ~90% MLP-intermediate sparsity — read `achieved_sparsity_mean`
from RESULT_JSON (computed directly from our own `refit_mlp` per-layer
state), not the printed per-batch lines.

### Interpretation
**Go.** ΔL1 at s=0.9, g=1 is clear and well outside noise — closed-form
refit of down_proj alone, with zero other repair machinery, recovers a
real share of masking's PPL cost. Proceeds to the reduced-cost matrix per
plan (gist "Next Experiments" / cost-reduction ladder rung 1: g∈{1,32} all
s, g∈{8,128} s=0.9 only), submitted same day as
`refit-l0l1-matrix-3b` (dev model, same instruct checkpoint) and
`refit-l0l1-matrix-8b` (main model, Llama-3.1-8B, confirmed a real
pretrained base checkpoint at /raid/LLM/Llama-3.1-8B — no fallback to
LLaMA2-7B needed).

L2 (sequential refit) is not yet built — this single-point result only
speaks to L1. It remains the next implementation task; nothing here
suggests skipping it.
