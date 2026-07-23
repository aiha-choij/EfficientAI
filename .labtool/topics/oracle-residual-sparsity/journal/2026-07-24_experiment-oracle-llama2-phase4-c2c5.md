# Experiment: oracle-llama2-phase4-c2c5 (4 jobs, one per condition)

Status: PENDING
Date: 2026-07-24

## Hypothesis tested
Main table of the topic. At matched per-token compute (top-K, s = 0.5/0.7/0.9):
- H1 (primary): C3 (residual score + exact mean-gate tail compensation) beats
  C1 (plain |i|) — ΔPPL(C3) < ΔPPL(C1) at equal s, especially at s=0.9.
- H3: C4 (rank-512 compensation, deployable form) stays close to C3.
- H2: C2 (|i|·col_norm) beats C1 on score correction alone.
- C5 (ḡ* u²-weighted mean) vs C3: better calibration constant?

## What we're testing over alternatives
One job per condition so failures are isolated and the queue can interleave;
C3 first (primary readout), then C4 (H3), then C2/C5. All inputs (stats/c4
g_bar, r=512 factors) come from the phase-2 job — no recalibration, so every
condition sees the identical calibration constants.

## Prior art check
- Phase 3 gate (2026-07-23 card): oracle path ≡ topk_intermediate, max Δ
  0.0013; dense 5.4738. C1 row of the main table comes from that job.
- Phase 0 (2026-07-22 card): r more concentrated in 30/32 layers (+3%p induced
  sparsity at matched p; med|r|/med|i| 0.64-0.92); inversion at layers 30-31;
  r=512 keeps 54-60% Frobenius energy mid-stack -> C4 risk flagged.

## Expected outcome
- Success (full go): C3 clearly under C1 at s=0.7/0.9 (the +0.256/+2.635
  degradations shrink meaningfully) AND C4 ≈ C3 (gap << C3's gain).
- Partial go: C3 wins but C4 collapses toward/below C1 -> rank problem
  (consistent with the Frobenius flag); next lever is bigger r or
  exclude_layers=[30,31].
- No-go: C3 − C1 gain < noise (~±0.01) at all s -> mean-gate decomposition
  buys nothing on the accuracy axis despite distribution-level concentration.
- C2 vs C1 isolates the score-correction share of any C3 gain.

## Reproducibility
- Git tag: exp/2026-07-24_oracle-llama2-phase4-c2c5 (commit 36b39ee)
- Job IDs (all submitted 2026-07-24 04:04, pinned a6000-2, 1 GPU >=30GiB,
  serial on GPU0 unless more free up):
  - 050-20260724-040432-oracle-llama2-phase4-c3
  - 050-20260724-040432-oracle-llama2-phase4-c4
  - 050-20260724-040432-oracle-llama2-phase4-c2
  - 050-20260724-040432-oracle-llama2-phase4-c5
- Command per job: `bash -c "export PY=$HOME/workspace/venv-larosa/bin/python;
  scripts/oracle/oracle_ppl_sweep.sh /raid/LLM/llama2-7b
  $HOME/workspace/oracle/llama2-7b <cond>"` (SELECT=topk, s grid {0.5,0.7,0.9})
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a6000-2)
- Inputs: stats/c4 (g_bar, g_bar_star), factors/r512 (C4 only), col_norm from
  weights at load
- Env: a6000-2 venv — torch 2.6.0+cu124, transformers 4.46.3, datasets 5.0.0,
  numpy 2.2.6, sdpa backend (no flash-attn), RTX A6000
- Eval: wikitext-2 test, eval_ppl_wikitext_with_inference_sparsity, ctx 2048

### Results
(pending)

### Interpretation
(pending)
