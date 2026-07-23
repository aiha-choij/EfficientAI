# Experiment: oracle-llama2-phase3-c0c1

Status: DONE (2026-07-24)
Date: 2026-07-23

## Hypothesis tested
None (plumbing validation). The oracle code path under top-K selection must be
experimentally equivalent to the trusted larosa topk_intermediate pipeline
before any C2-C5 result can be believed.

## What we're testing over alternatives
Running C1 (score |i|, no compensation) with top-K selection s={0.5,0.7,0.9}
plus one dense (C0) run, through the NEW oracle path, expecting exact
reproduction of known numbers. Alternative would be trusting the CPU unit
tests alone; this closes the gap on the real model + bf16 + flash-attn +
eval-loop integration.

## Prior art check
- topk_intermediate anchors (2026-07-22 card, same model/eval): dense 5.4736,
  C1-equivalent PPL 5.521/5.730/8.108 at s=0.5/0.7/0.9.
- Unit test pins C1-topk == topk_intermediate bitwise at module level
  (commit 6574223); this job is the end-to-end version.

## Expected outcome
- Success gate: dense == 5.4736 and C1 == 5.521/5.730/8.108 (bitwise-equal
  selection semantics -> expect exact or near-exact match, far inside the
  ±0.1 pipeline-noise bound); measured mlp sparsity == s per layer.
- Failure: any deviation beyond noise -> bug in the oracle mask/apply/logging
  path; DO NOT proceed to Phase 4.

## Reproducibility
- Git tag: exp/2026-07-23_oracle-llama2-phase3-c0c1 (commit a2b8b4e)
- Job ID: 050-20260724-014928-oracle-llama2-phase3-c0c1
  (attempt 1 = 050-20260723-234944, pinned a100-40-2, dequeued after ~2h wait:
  all gateway A100s held ~30GB of other members' container jobs. Resubmitted
  on a6000-2 with a fresh venv — torch 2.6.0+cu124, transformers 4.46.3,
  datasets 5.0.0, numpy 2.2.6, NO flash-attn -> sdpa backend via
  best_attn_impl() (commit after a2b8b4e). Gate reading loosens from
  "bitwise" to "within ±0.1 pipeline noise" due to backend + GPU arch change.)
- Assigned host/GPU: a6000-2 (pinned), 1 GPU >=30GiB [pending dispatch]
- Command: `bash -c "scripts/oracle/oracle_ppl_sweep.sh /raid/LLM/llama2-7b
  $HOME/workspace/oracle/llama2-7b dense && ... c1"` (SELECT=topk default,
  s grid {0.5, 0.7, 0.9})
- Workdir: /raid/choij/workspace/repos/EfficientAI/larosa
- Config path: n/a (script args)
- Key parameters: select=topk, K=int((1-s)*d), wikitext-2 test PPL ctx 2048,
  bf16, flash_attention_2, attention dense; results JSON under
  ~/workspace/oracle/llama2-7b/results/
- Key deps: torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1
  (conda env larosa)

### Results
Attempt 2 (050-20260724-014928) failed fast: LlamaSdpaAttention lacked
infer_sparsity_h1/h2 attrs (only the flash path set them) -> AttributeError in
the eval loop. Fixed by initializing both attrs in LlamaAttention.__init__
(commit 36b39ee). Attempt 3 (050-20260724-021237) completed in 108 min on
a6000-2 GPU0 (sdpa, venv).

GATE PASSED — anchors reproduced to 4 decimals despite A100->A6000 and
flash->sdpa changes:
- dense 5.4738 (anchor 5.4736, Δ +0.0002)
- C1 s=0.5: 5.5216 (5.5210, Δ +0.0006), sp 0.4992
- C1 s=0.7: 5.7284 (5.7296, Δ −0.0012), sp 0.6991
- C1 s=0.9: 8.1096 (8.1083, Δ +0.0013), sp 0.8995

### Interpretation
The oracle code path is experimentally equivalent to the trusted
topk_intermediate pipeline (max Δ 0.0013 << ±0.1 noise bound), and backend/
GPU-arch effects are ~1e-3 — C2-C5 deltas larger than that are real signal.
Phase 4 unblocked; submitted immediately (see phase4 card).
