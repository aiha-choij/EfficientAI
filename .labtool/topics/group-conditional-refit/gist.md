# group-conditional-refit — Does the optimal refit solution depend on the mask pattern?

## Status
active — started 2026-08-04. Spec: research-wiki
`plans/group-conditional-refit-spec.md` (derivation, critical review §8,
E-ladder). Bridges `local-loss-refit` (marginal refit baseline) and
`block-sparse-compensation` (C8 stack).

## Notation
```
Marginal refit: W̃⁰ = single anchored-ridge solution pooled over all
  calibration mask patterns (the existing refit).
GC-refit: W(μ) = W⁰ + Σ_b δ_b Δ_b — first-order expansion of the
  mask-conditional solution W*(μ) in dropped-neuron-block indicators δ_b.
  Closed-form (varying-coefficient ridge); ℓ2,1 across blocks selects which
  corrections survive; rank-r truncation per block.
Setting: LLaMA2-7B P3′ (token block g=16 × neuron block B=64, s=0.9,
  residual score |u(g−ḡ)|·colnorm). Number anchors: no-comp 6874 / C7 4952 /
  C8 48.9 / C8a 37.2 / in-family anchor 6.93 / dense 5.4736.
```

## Hypothesis
Refit works by correlation-based reallocation (dropped neuron's share moved
to surviving correlated neurons). The optimal coefficients depend on WHICH
neurons are dropped; the marginal solution is a mask-averaged compromise.
If mask-conditional heterogeneity is real and material, a closed-form
per-dropped-block correction (feasible only under group sparsity: finite
block vocabulary + amortized assembly over g tokens) closes part of the
C8→C8a gap at zero training cost.

Key risk (spec §8 A1): if neuron correlations are global, W*(μ) is
insensitive to μ and the whole method is void — hence gate experiment E0
before any implementation of E1.

## Key Findings
- 2026-08-04 (adjacent evidence, oracle-residual-sparsity fusion job):
  joint anchored ridge (MGR × refit, per-token s=0.9) PPL 6.195 — beats all
  three pre-registered thresholds (SLR 6.94 / TIS frontier 6.71 / exact
  compensation 6.64). Stage-1 machinery of GC-refit validated per-token.

## Dead Ends
(none yet)

## Open Questions
1. E0: does the mask-conditional heterogeneity exist? (gate; running)
2. If E0 positive but E1 falls short: few-pattern vocabulary (mask-cluster
   solutions) instead of first-order expansion — remember APC's low ceiling.
3. Identification vs importance tension: backbone blocks are rarely dropped
   (few conditional samples) yet their rare drops may hurt most (spec §8 A4).

## Next Experiments
1. ~~E0 heterogeneity diagnostic~~ — SUBMITTED as gateway request (see
   Active Jobs). Includes Part 2: oracle block-mean compensation PPL
   (bounds the token-axis-shared fraction of the dropped signal).
2. (contingent on E0 positive) E1: backfitting GC-refit on P3′ setting,
   arms {C8, C8+marginal refit, C8+GC-refit(τ, r sweep)}, judged against
   matched-FLOPs r_sk-increase control (E2). Spec §4-5.

## Active Jobs
- Gateway request `20260804-171529-gc-refit-e0` (a100-40-2, agent-driven) —
  E0 mask-cluster refit cross-evaluation + oracle block-mean compensation.
  Report: `a100-40-2:~/workspace/reports/20260804-171529-gc-refit-e0.md`
  (after completion) or `requests/active/.../report.md` (during).

## Boundaries (do not duplicate)
- `local-loss-refit` (gateway topic): marginal refit — baseline numbers,
  do not re-measure.
- `block-sparse-compensation` (gateway topic): C7/C8/C8a stack and P3′
  artifacts — reuse, never re-cluster (comparability).
- `oracle-residual-sparsity`: per-token fusion (Stage-1 analog) lives there.
- `groupwise-flocking-tuning`: upstream (selection-path) shaping — opposite
  axis; its PoC interpretation is still pending user confirmation.

## Pointers
- Spec: research-wiki `plans/group-conditional-refit-spec.md` (mac-local).
- Request prompt snapshot: gateway
  `~/workspace/requests/active/20260804-171529-gc-refit-e0/prompt.md`.
- Planned code home: gateway branch `auto/group-conditional-refit`
  (solve_refit from `auto/refit-honesty-corrections`, P3′ mask path from
  `auto/block-sparse-compensation`).
