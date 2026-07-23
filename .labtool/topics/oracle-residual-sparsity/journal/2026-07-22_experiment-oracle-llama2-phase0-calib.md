# Experiment: oracle-llama2-phase0-calib

Status: DONE (2026-07-23)
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
Job completed in ~440 min on a100-40-2 GPU 0. c4 streaming download WORKED —
primary stats = stats/c4, wikitext103 as corpus-B. Artifacts under
a100-40-2:~/workspace/oracle/llama2-7b/{stats/{c4,wikitext103},phase0,factors/r512}.
(matplotlib absent in env larosa -> phase0_curves.png skipped; CSV complete.)

- **H1 preview: GO.** r-curve above i-curve in 30/32 layers. Mean induced
  sparsity (over layers): p=0.7: i 0.722 vs r 0.755 (+3.2%p); p=0.8: 0.623 vs
  0.655 (+3.3%p); p=0.9: 0.475 vs 0.505 (+3.0%p); p=0.95: +2.5%p; p=0.99:
  +1.4%p. Gap peaks mid-stack (layers 6-18: +4.5 to +5.3%p at p=0.9) and goes
  NEGATIVE in the last two layers (30: -0.2%p, 31: -2.6%p).
- Hoyer/kurtosis consistently higher for r than i (e.g. mid layers 0.51-0.55
  vs 0.40-0.47), except layer 31 Hoyer (0.625 vs 0.660).
- comp_win_frac P(|g-g_bar|<|g|) > 0.5 in all layers (0.54-0.68; lowest at 31).
- **g_bar corpus stability: WEAK in early layers.** Pearson(c4, wikitext103)
  per layer ranges 0.48-0.995; layers 2-4 at 0.48-0.53, most mid layers
  0.87-0.93, layer 31 at 0.995. Well below the ~0.95 hoped for.
- CV^2 median explodes in late layers (40+ for layers 24-30 vs 3-6 mid-stack)
  — gate varies strongly relative to its mean exactly where the gap shrinks.
- r=512 factors: Frobenius energy retained 0.54-0.60 (layers 4-17, worst
  0.540 at layer 7), rising to 0.94/0.985 at layers 30/31. M is NOT strongly
  low-rank in Frobenius terms mid-stack — H3 risk flag (though what matters
  is (M_hat-M)x on the real x distribution, tested in Phase 4).

### Interpretation
Distribution-level go for H1: subtracting the calibration mean gate
concentrates the intermediate in 30/32 layers, worth ~3%p of induced sparsity
at matched p — modest, and by itself smaller than the +15%p (C4 vs C2)
full-go bar of spec section 8, so the PPL sweeps must show whether
compensation converts this into a larger accuracy-side gap. Three actionable
structure findings: (1) the effect concentrates mid-stack and INVERTS in the
last two layers — exclude_layers=[30,31] (keep them dense) is a cheap variant
to try if C3/C4 underperform; (2) g_bar is corpus-sensitive in early layers
(Pearson ~0.5), so C3/C5 results inherit calibration noise there; (3) M has a
heavy singular tail mid-stack (r=512 keeps only ~55-60% Frobenius energy), so
C4-vs-C3 degradation is the thing to watch for H3. Proceed to Phase 3
(dense + C1 sweep), then Phase 4 C2-C5.
