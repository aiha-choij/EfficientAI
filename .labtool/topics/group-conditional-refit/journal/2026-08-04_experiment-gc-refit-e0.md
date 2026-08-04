# 2026-08-04 — experiment: gc-refit-e0

Status: PENDING

## Hypothesis tested
The optimal refit solution W*(μ) varies materially with the mask pattern μ:
anchored-ridge refit solutions fit on disjoint mask-pattern clusters should
reconstruct their own cluster's held-out tokens better than the other
cluster's (heterogeneity gain = cross/within MSE − 1 > 0). This is the
pre-registered gate (spec §8 A1) for building GC-refit at all.

## What we're testing over alternatives
- vs going straight to E1 (implementing GC-refit): E0 costs one calibration
  sweep and answers the falsifiable core assumption first — if neuron
  correlations are global (mask-insensitive), marginal refit is already
  optimal and E1 is void.
- Part 2 (oracle block-mean compensation) separates block-shared vs
  token-specific composition of the dropped signal — sizes the ceiling for
  token-axis compensation sharing (proposal 2) in the same job.

## Prior art check
- No prior mask-conditional/heterogeneity work in any topic journal
  (searched: mask-conditional, conditional refit, heterogeneity,
  varying-coefficient). Neuron clustering exists but is a different axis
  (coactivation-block-structure P1-P3: clusters neurons, not mask patterns).
- Fusion result same day (oracle-residual-sparsity): joint anchored ridge
  per-token s=0.9 → PPL 6.195, beating all three thresholds (6.94/6.71/6.64)
  — Stage-1 machinery works; strengthens the case for testing conditional
  refinement.
- Known confounds inherited: 0-shrinkage prior bug (use W_down-anchored
  ridge only); score-family split (this topic uses residual score, matching
  the block-comp stack, NOT |i|·colnorm).

## Expected outcome
- Positive: median per-layer heterogeneity gain ≥ ~15% with consistent sign
  across layers → proceed to E1 (backfitting GC-refit).
- Negative: gain < ~5% (cross ≈ within) → record "mask-conditional
  structure not material" as a Dead End for this topic; pivot to C8-side
  improvements (asymmetric ranks, C9). (Criteria provisional — finalize on
  first numbers, house convention.)
- Part 2: block-mean PPL near C8a (37.2) → dropped signal mostly
  block-shared; near C7 (4952) → token-specific, per-token estimation stays
  mandatory.

## Reproducibility
- Git tag: n/a — request-driven experiment; implementation happens on the
  gateway branch `auto/group-conditional-refit` (agent will record commit
  hashes in its report; agent-pr for any code change). Mac repo unchanged.
- Request ID: 20260804-171529-gc-refit-e0 (gateway a100-40-2, supervisor/
  agent flow; jobs will be linked via `qsub -r`)
- Assigned host/GPU: [pending dispatch — per-job, see request report]
- Command: `ask -n gc-refit-e0 -f /tmp/gc-refit-e0-request.md` (prompt
  snapshot: gateway `requests/*/20260804-171529-gc-refit-e0/prompt.md`)
- Config path: n/a — full experiment definition in the request prompt
- Key parameters: LLaMA2-7B, P3′ g=16 × B=64, s=0.9, residual score;
  calibration c4 512×2048 split 448 fit / 64 held-out by sequence;
  2 balanced Hamming k-means mask clusters; anchored ridge λ ∈ {0.01, 0.1};
  fp32 stats; artifacts → `~/workspace/gcrefit/llama2-7b/`
- Key deps: gateway conda env `larosa` (torch 2.6.0+cu124,
  transformers 4.46.3)

## Notes
- Spec with derivation and critical review: research-wiki
  `plans/group-conditional-refit-spec.md` (§4 E0 row, §8 A1).
- Pre-registered: E0 negative blocks E1 (do not implement GC-refit on a
  negative E0).

### Results
Source artifacts: gateway report `~/workspace/reports/20260804-171529-gc-refit-e0.md`
(final, status DONE 2026-08-04 21:40), `~/workspace/gcrefit/llama2-7b/
part1_e0_report.json`, `part2_oracle_blockavg.json` (git `06f4056`);
code in PR #4 (branch `auto/group-conditional-refit`, 3 commits).

**Part 1 (heterogeneity gate)** — LLaMA2-7B, 32 layers, fit-set clusters
exactly 28672/28672 balanced, sanity (closed-form MSE ≤ anchor MSE)
0 violations at both λ:

| λ | median gain (cross/within − 1) | mean | pre-registered verdict |
|---|---|---|---|
| 0.01 | 8.14% | 8.40% | grey zone (5–15%) |
| 0.1 | 4.78% | 4.95% | negative (<5%) |

Only 1/32 layers exceeds the 15% positive threshold at λ=0.01 (layer 0,
17.52%); none at λ=0.1. Gain roughly halves when λ goes 0.01→0.1 —
direction consistent with the λ=0.01 gain being partly a small-sample
overfit artifact (not confirmed; would need larger fit-set / dense λ
sweep). within-vs-marginal improvement is mostly NEGATIVE (cluster-
conditional solutions usually worse than the pooled marginal solution
on held-out). Caveat: held-out nearest-centroid assignment is heavily
imbalanced on some layers (e.g. layer 26: 4,832 vs 126,240 tokens), so
minority-cluster MSEs are noisy; does not change the qualitative verdict.

**Part 2 (oracle block-mean compensation)** — PPL **1590.13**
(g=16, B=64, achieved s_block=0.9012) vs C8a 37.17 / C8 48.89 /
C7 4952.39 / C7a 6874.22 (same setting, residual score). On log scale
the result sits ~72% of the way toward the no-compensation end →
the dropped signal is dominated by token-specific components; block-mean
sharing of even the EXACT dropped contribution is far from sufficient.
Block-local mean does beat the global mean gate (C7) by 3.1× — real but
practically insufficient.

**Infra findings by the agent during the run** (report §infra): dispatcher
tmux was stalled (restarted by the agent after verifying other users'
jobs are isolated); MINFREE=40 can never be satisfied on 40 GiB cards
(40959 MiB free < 40960 — submitted with -m 39); one unreproduced
duplicate-pending anomaly during the stall window.

### Interpretation
