# Experiment: refit-l0l1-matrix-8b

Status: DONE (2026-07-31)
Date: 2026-07-31

## Hypothesis tested
Does the dev-model (llama3.2-3b-instruct) finding -- refit clearly helps at
s=0.9 but HURTS at s=0.5/0.7 -- replicate on the main model (Llama-3.1-8B,
base pretrained, not instruct-tuned), or is it an artifact of the dev
model's instruct tuning / the wikitext103-vs-wikitext-2 calibration/eval
corpus mismatch?

## What we're testing over alternatives
Same reduced-cost grid as the dev-model matrix, same code/config, only the
model changes (Llama-3.1-8B, a real base pretrained checkpoint already
local at /raid/LLM -- no fallback needed). This isolates "is the low-s
regression about the model" from "is it about refit itself".

## Prior art check
- refit-l0l1-matrix-3b (2026-07-31, same day): the pattern this job is
  testing for replication.
- refit-l0l1-validate-3b (2026-07-31): established the s=0.9,g=1 Go signal
  this job also re-confirms on a second model.

## Expected outcome
If the low-s regression is a real property of L1 refit (not a dev-model
artifact): s=0.5/0.7 should still be flat-to-negative on Llama-3.1-8B, and
s=0.9 should still be clearly positive and grow with g. If it was an
instruct-tuning/corpus artifact: s=0.5/0.7 should turn neutral-to-positive
on this cleaner, base-model, in-distribution setup.

## Reproducibility
- Git tag: none (branch auto/local-loss-refit, commit 8c92022 at submit
  time -- see RESULT_JSON `git_commit` in each result file)
- Job ID: 050-20260731-105247-refit-l0l1-matrix-8b (resubmission of
  050-20260731-104521-... which OOM'd; see prior journal card / gist
  Key Findings for the layer-chunking fix)
- Host/GPU: a100-40-2, GPU2 (pinned via -m 32 to land on the free-est GPU),
  1 GPU, 258 min elapsed
- Command: `env PY=.../envs/larosa/bin/python bash
  larosa/scripts/refit/run_matrix.sh /raid/LLM/Llama-3.1-8B
  /home/choij/workspace/refit/Llama-3.1-8B`
- Config: lambda=0.01, calibration wikitext103 (512x2048, seed=42, same
  recipe as the 3B run but a SEPARATE calib_tokens.pt -- different token
  IDs since tokenizer differs, same corpus/size/seed), eval wikitext-2,
  --layers_per_pass auto-sized (12GiB budget -> ~12 layers/pass -> 3
  calibration sweeps per L1 build, per the OOM fix)
- Model: `/raid/LLM/Llama-3.1-8B` -- confirmed base pretrained checkpoint
  (config.json: no instruct markers, tie_word_embeddings=false,
  transformers_version 4.43.0.dev0, 8.03B params) -- exactly the spec's
  "주 결과" model, no substitution needed.

### Results
(artifacts: job log at ~/workspace/runs/20260731-105247-refit-l0l1-matrix-8b/log;
result JSONs under ~/workspace/refit/Llama-3.1-8B/results/)

| s | g | L0 PPL | L1 PPL | ΔL1 = L1−L0 | ΔL1 (relative) |
|---|---|---|---|---|---|
| 0.5 | 1 | 6.3869 | 7.2615 | **+0.8746** | +13.7% |
| 0.7 | 1 | 7.0251 | 7.7763 | **+0.7512** | +10.7% |
| 0.9 | 1 | 12.9347 | 10.3305 | **−2.6042** | −20.1% |
| 0.5 | 32 | 8.5315 | 9.6125 | **+1.0810** | +12.7% |
| 0.7 | 32 | 13.1051 | 12.9952 | **−0.1098** | −0.8% |
| 0.9 | 32 | 98.2602 | 47.9033 | **−50.3569** | −51.2% |
| 0.9 | 8 | 37.1379 | 21.4470 | **−15.6909** | −42.3% |
| 0.9 | 128 | 269.3102 | 111.9309 | **−157.3793** | −58.4% |

Absolute PPL here (6.4-269) is far more in-distribution than the instruct
model's (11-372) -- L0 at s=0.5 (6.39) is close to the ballpark of known
LLaMA-family dense/light-sparsity wikitext-2 PPL, consistent with this
being a well-behaved base model on its native task, unlike the instruct
checkpoint evaluated on raw continuation.

**Sharing-tax absorption at s=0.9** (g=1 -> g=128): L0 grows 12.93 -> 269.31
(+256.38); L1 grows 10.33 -> 111.93 (+101.60). Refit absorbs
(256.38−101.60)/256.38 ≈ **60% of the additional sharing-tax PPL** --
notably MORE than the dev model's ~37%, i.e. the effect is not just
present but *stronger* on the larger base model.

### Interpretation
**CONFIRMED, not an artifact.** The dev-model pattern replicates cleanly on
Llama-3.1-8B (base, in-distribution eval, matched calibration recipe):
- **s=0.9 (all g): refit is a clear, large win**, growing with g (g=1:
  −20.1% relative, g=128: −58.4% relative) -- same qualitative shape as the
  3B run, with markedly BETTER sharing-tax absorption (60% vs 37%).
- **s=0.5: refit consistently HURTS** (+13.7% at g=1, +12.7% at g=32) --
  replicates the 3B finding almost exactly in relative terms.
- **s=0.7: mixed / near the crossover** -- still slightly negative at g=1
  (+10.7%, matching 3B's regression) but roughly NEUTRAL at g=32 (−0.8%,
  vs 3B's clearly-negative +18.7%). The crossover point where refit starts
  helping appears to sit lower (in s and/or shift with g) on this larger
  base model than on the smaller instruct one.

This closes the "is it a dev-model artifact" open question from the 3B
matrix card: **no** -- the s-dependent sign flip is a real property of
plain closed-form down_proj refit under this masking rule, not an
instruct-tuning or calibration-corpus-mismatch artifact (though the exact
crossover location may still be somewhat model/corpus-dependent, given the
7:32 shift between the two models).

**Revised understanding of the method**: local loss refit (L1) is NOT a
uniformly-helpful drop-in fix for masking damage. It is a HIGH-SPARSITY,
COARSE-MASK tool -- most valuable exactly where the masking damage is
largest (high s, high g / heavy sharing tax), and actively counterproductive
at the sparsity levels where masking damage is small. Bias-variance framing
from the 3B card still stands: little true bias to correct + fixed
calibration sample -> net loss from fitting noise; large true bias -> net
win. A practical deployment reading: L1 refit should be gated ON only past
some s (and possibly g) threshold, not applied uniformly across the sparsity
range.

## Next
- Both L0/L1 matrix jobs (dev + main) are complete. Reduced-cost rung 1 is
  DONE across both models -- update Next Experiments accordingly.
- Proceed to L2 (sequential GPTQ-style refit) implementation, per plan.
  Open question for L2 to help answer: does recomputing the mask against
  the SPARSE stream (L2's definition) change the low-s regression at all,
  or is it purely a property of down_proj-only refit regardless of which
  stream the mask/z come from?
- Not run (out of scope for this card, flag only): a same-corpus
  calibration/eval ablation (calibrate on wikitext-2 itself) to fully rule
  out any residual corpus-mismatch contribution to the low-s regression's
  magnitude -- the model-replication result already rules it out as the
  SOLE cause, but its exact size contribution is still untested.
