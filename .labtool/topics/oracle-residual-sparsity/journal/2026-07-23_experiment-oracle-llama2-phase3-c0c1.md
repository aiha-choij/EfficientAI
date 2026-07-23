# Experiment: oracle-llama2-phase3-c0c1

Status: PENDING
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
- Job ID: 050-20260723-234944-oracle-llama2-phase3-c0c1
- Assigned host/GPU: a100-40-2 (pinned), 1 GPU >=30GiB [pending dispatch]
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
(pending)

### Interpretation
(pending)
