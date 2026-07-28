# Experiment: oracle-llama2-e2-s2-b2048

Status: DONE (2026-07-28) — results recorded; full interpretation pending user reading
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
(artifacts: 12 result JSONs c4_plain_r2048_topk_s*.json and
c4_slr_r{1024_k2048,512_k3072,256_k3584}_topk_s*.json on a6000-2
~/workspace/oracle/llama2-7b/results/; job 050-20260729-022242 STATE 완료,
33 min; log tail "E2 B2048 sweep complete"; achieved sparsity
0.5000/0.7000/0.9001 in all runs)

wikitext-2 PPL at B_eff=2048 (+12.4% compute); dense 5.4738, C3 exact
5.5051/5.6283/6.6381, E1 best (B_eff=1024) 6.9417 @ s=0.9:

| arm | s=0.5 | s=0.7 | s=0.9 |
|-----|-------|-------|-------|
| lr_r2048 (LR reference) | 5.5329 | 5.6652 | 6.7098 |
| s2 r1024:k2048 | 5.5140 | 5.6420 | 6.6674 |
| s2 r512:k3072  | 5.5074 | 5.6336 | 6.6421 |
| s2 r256:k3584  | **5.5049** | **5.6298** | **6.6344** |

- Sanity check PASSED: lr_r2048 (6.7098) beats lr_r1024 (7.2294) @ s=0.9.
- Pre-agreed gate (user, 2026-07-27) PASSED: best s2 arm beats BOTH
  lr_r2048 (6.6344 < 6.7098) and E1's 6.9417. Stretch target ≤6.79 also met.
- All three s2 arms land on the exact-compensation rule within
  −0.004…+0.029 PPL @ s=0.9 (C3 = 6.6381); r256:k3584 is nominally 0.0037
  BELOW C3, which is at the pipeline noise floor (~0.001–0.01) — read as a
  tie, not as beating the exact form.
- Rank-share ordering is preserved from E1 (sparser is better within the
  budget: 12.5% < 25% < 50% < 100% rank share) but the spread has collapsed
  from 0.29 PPL at B_eff=1024 to 0.075 at B_eff=2048.
- Arithmetic note (derived, not measured): 2hr = h² exactly at r = h/2 =
  2048, so EVERY arm in this round costs the same as computing Mx exactly
  (16.78 M MACs/token/layer = 12.40% of the dense FFN). C3 itself is
  available at this budget.

### Interpretation
(Gate outcome applied mechanically per the user's pre-agreed rule of
2026-07-27; the deeper reading below is FLAGGED FOR USER CONFIRMATION and
must not be treated as settled.)

- Recorded fact: the E2 gate passed on every stated criterion.
- Open question raised by the arithmetic note, for the user to rule on:
  because B_eff=2048 equals the cost of exact Mx, this round reads as a
  SATURATION check rather than a deployable operating point — at this
  budget one could simply compute Mx. If accepted, the consequence is that
  the deployable frontier lives at B_eff ≤ 1024 and the next round should
  probe DOWNWARD (B_eff = 512, 256) rather than upward, with per-layer
  (r_l, k_l) allocation as the second axis.
- Report covering baseline → MGR → LR → SLR with plots and the compute
  accounting: results/reports/mgr-slr-report-2026-07-28.html
  (artifact: https://claude.ai/code/artifact/48a40f99-6da8-4d34-ac09-729a506604f9)
