# Experiment: oracle-llama2-phase0-calib

Status: PENDING
Date: 2026-07-22

## Hypothesis tested
H1 early signal (spec section 4): the mean-gate residual r = u*(g - g_bar) is
more concentrated than the intermediate i = u*g on LLaMA2-7B — its top-p
induced-sparsity curve must sit ABOVE the i curve. Also produces everything
Phase 3/4 need: calibration stats (g_bar, g_bar_star, E[g^2]) and C4
compensation factors (r=512).

## What we're testing over alternatives
Distribution-level preview before spending GPU time on the full C1-C5 PPL
sweeps — if r is not more concentrated than i, H1 dies here for ~1 job
instead of ~5 sweep jobs. Two calibration corpora (c4 primary, wikitext-103
second) to verify g_bar is corpus-stable before trusting it as a fixed model
constant.

## Prior art check
No prior calibration/distribution runs in this line (first stats-based
experiment; the old topic's Top-K mechanism was calibration-free). Infra
lessons inherited from 2026-07-22_experiment-larosa-llama2-topk-int-ppl:
pin -H a100-40-2 (models at /raid/LLM), absolute -d path (dispatcher chokes
on literal `~`). Phase-1 unit tests (gist Key Findings) verified the c3/c4
algebra, so phase-0 metrics reflect distributions, not bugs.

## Expected outcome
- Success (go for H1): r's induced-sparsity curve above i's at every p in
  {0.7, 0.8, 0.9, 0.95, 0.99} for most layers; headline gap at p=0.9 clearly
  positive. g_bar c4-vs-wikitext Pearson high (~>0.95) across layers.
  Secondary: Hoyer/kurtosis higher for r; comp_win_frac (P(|g-g_bar|<|g|))
  well above 0.5; r=512 Frobenius energy retention per layer recorded for H3.
- Failure (no-go signal): r curve at or below i curve — residual concentration
  hypothesis rejected at the distribution level; rethink before any sweep.
- Infra failure mode: c4 streaming download fails on the execution server ->
  script falls back to wikitext103 as primary (logged WARN, no Pearson check);
  rerun c4 later if needed.

## Reproducibility
- Git tag: exp/2026-07-22_oracle-llama2-phase0-calib (commit 6bf95b1)
- Job ID: 050-20260723-155254-oracle-llama2-phase0-calib
- Assigned host/GPU: a100-40-2 (pinned), 1 GPU >=30GiB [index pending dispatch]
- Command: `bash scripts/oracle/run_phase2.sh` (defaults: model
  /raid/LLM/llama2-7b, out $HOME/workspace/oracle/llama2-7b)
- Workdir: /raid/choij/workspace/repos/EfficientAI/larosa
- Config path: n/a (all parameters are script args/defaults)
- Key parameters: calibration 512 seq x 2048 tok per corpus (wikitext-103
  first, then c4 streaming with fallback), seed 42, batch 1, bf16 forward /
  fp32-fp64 accumulation; phase-0 on 32 calib sequences, p grid
  {0.7,0.8,0.9,0.95,0.99}; M = W_d diag(g_bar) W_u, SVD rank 512;
  outputs under ~/workspace/oracle/llama2-7b/{stats,phase0,factors/r512},
  primary corpus recorded in primary_stats.txt
- Key deps: torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1
  (conda env larosa)

### Results
(pending)

### Interpretation
(pending)
