# Experiment: oracle-llama2-e2-s2-b2048

Status: PENDING
Date: 2026-07-28

## Hypothesis tested
H4/S2 at doubled budget: the slr_input edge over plain rank persists at
B_eff = 2048 (+12.4% compute), and the rank-share optimum stays in the
mixed region (E1 found ~25% at B_eff=1024). Also closes the old open item
"plain uniform r=2048 reference arm" in the same job.

## What we're testing over alternatives
Bracketing rank share {50%, 25%, 12.5%} around E1's optimum instead of
re-running a screening pass — E1 showed offline L2 does not predict the
fine ordering, so the bracket is chosen on PPL evidence and measured
directly on PPL. lr_r2048 rides along as the within-family reference.

## Prior art check
- E1 card (2026-07-28, same topic): gate passed — s2 r256:k1536
  5.5961/5.7526/6.9417 vs lr_r1024 5.7365/5.9152/7.2294; mixed split beat
  pure sparse; s=0.5 comp arms still trail C1.
- Whitening round (2026-07-24): rank 512→1024 gave −0.52/−0.64/−1.53 —
  diminishing-returns question for 1024→2048 is open; r=2048 arm was
  queued then deferred (subsumed here).
- E0 (2026-07-27): S2 screening pass; S1 dead.
- No Dead End conflicts.

## Expected outcome
- Success: best s2 arm beats BOTH lr_r2048 and E1's 6.9417 @ s=0.9.
  Stretch: ≤ 6.79 @ s=0.9 (C3 + 0.15). Read where the rank-share optimum
  moves (up/down/flat) to seed E3 allocation.
- Informative failure: s2 edge shrinks or vanishes vs lr_r2048 → the
  sparse channel path mainly substitutes for MISSING rank; at high rank
  the residual R is small and top-|x| selection matters less — then E3
  should allocate budget per layer rather than per family, and the
  deployable sweet spot stays at B_eff≈1024.
- Sanity: lr_r2048 must beat lr_r1024 (7.2294 @ s=0.9); if not, suspect a
  build/eval bug, do not interpret.

## Reproducibility
- Git tag: exp/2026-07-28_oracle-llama2-e2-s2-b2048 (9b36cd4; local =
  github = gateway = a6000-2 clone all synced)
- Job ID: 050-20260729-022242-oracle-llama2-e2-s2-b2048
- Assigned host/GPU: a6000-2 (pinned via -H), 1 GPU >=30GiB [pending dispatch]
- Command: `bash -c "export PY=/home/choij/workspace/venv-larosa/bin/python;
  scripts/oracle/run_e2_s2_b2048.sh /raid/LLM/llama2-7b
  /home/choij/workspace/oracle/llama2-7b"`
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a6000-2)
- Config path: n/a — parameters in run_e2_s2_b2048.sh
- Key parameters: condition c4, select topk, s {0.5,0.7,0.9}; arms:
  plain lr rank 2048 (factors/plain_r2048) + slr_input abs (r:k)
  {1024:2048, 512:3072, 256:3584} (factors/slr_r{R}_k{K}); stats_dir
  stats/c4 unchanged; results c4_{plain_r2048|slr_r*_k*}_topk_s*.json.
  Pipeline: 03_build_M x4 (spectra skipped) -> 04_eval_ppl x12.
- Key deps: torch 2.6.0+cu124, transformers 4.46.3, venv
  /home/choij/workspace/venv-larosa, sdpa backend, RTX A6000
- Estimated cost/time: 1 GPU, ~12-16 h wall (12 PPL evals + 4 builds;
  E1's 9 evals + 3 builds took ~10 h incl. queue)

### Results

### Interpretation
