# Init: block-sparse-compensation

Date: 2026-07-31

## Initial hypothesis
H4 (primary): block-shared mask sharing tax recoverable ≥50% by block-wise
compensation (C7/C8). H5: per-token low-rank gate estimate (ĝ, C8) beats
mean-gate (ḡ, C7) compensation because sharing-tax neurons are exactly the
ones whose gate deviates most from ḡ. Full math in `spec.md`.

## Starting phase
Hypothesis stage → about to start Phase 1 (implementation: block_size +
C7a/C7/C8a/C8 conditions + 5 unit tests, CPU-only).

## Notes
- This topic is the direct successor of two closed/paused threads:
  `local-loss-refit` (refit alone can't recover g>1 sharing tax — static
  linear correction) and `coactivation-block-structure` P3 (bare block mask
  without compensation is catastrophic, PPL 4.5k-24k). Advisor consensus
  2026-07-31 motivated this topic; see spec.md §1 for the full context.
- Before starting Phase 1, closed out the `coactivation-block-structure` P3
  journal card's empty Interpretation section and refreshed that topic's
  gist (Active Jobs/Next Experiments were stale — both P3 jobs were already
  STATUS=ok). See that topic's gist Key Findings/Dead Ends for the summary;
  the P3 finding (clustered beats random 2-3x but both catastrophic vs
  anchor) is the empirical grounding for why this topic's recovery-rate
  metric matters.
- Branch: `auto/block-sparse-compensation`, from `main` (302593a) — separate
  from `auto/local-loss-refit` (PR #1, unmerged), which is where the §4
  refit-honesty corrections (C1/C2/M1) will be recorded, per spec §5.
