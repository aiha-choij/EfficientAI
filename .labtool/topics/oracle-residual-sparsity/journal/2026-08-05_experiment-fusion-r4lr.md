# 2026-08-05 — experiment: fusion-r4lr (ALS-learned low-rank tail map)

Status: PENDING

## Hypothesis tested
Round 4 showed the nonlinear tail gain does NOT survive post-hoc SVD
truncation of the learned dense tail map W_tail (g=1: 6.10 -> 6.43 at
r_sk=688). Hypothesis: learning the tail map in FACTORED form Bt·At
(rank r_t = r_sk) from the start — alternating least squares where each
half-step is an anchored closed-form solve on the SAME famA Gram —
retains most of the full-map gain at deployable cost.

ALS scheme (per layer, per lambda; warm start = post-hoc truncation,
composite anchor = rank-r_t SVD of W_d, per r4's exact-estimate logic):
- (i) fix At: features [m*r; Ax; At psi] -> joint anchored solve for
  [W_d, B, Bt] via Gram-block projection (no new calibration passes);
- (ii) fix (W_d, B, Bt): At = P^-1 (Bt^T Ce + lam*Dpsi*P*Ad0)
  (Gpsi + lam*Dpsi I)^-1, P = Bt^T Bt, Ce = residual correlation.
3 rounds (--als_iters).

## Prior art check
- Round 4 (2026-08-04, same card series): r4trunc failure is the direct
  motivation; r4lr's eval body is identical to r4trunc's (only the
  solved factors differ) — clean A/B.
- flocking PoC: alternating closed-form solves precedent (IRLS+CG);
  lesson "anchor every solve at originals" applied to both half-steps.
- Deploy compute at r_sk=r_t: 0.162 + 3*r_sk*(h+d)/(3hd)
  (= ~0.39 at 688; ~0.62 at 1376).

## Expected outcome
- Success: r4lr well below r4trunc at matched rank (g=1 target:
  < ~6.2 vs trunc 6.434; g=16 target: < ~6.8 vs trunc 7.238), ideally
  approaching the full-map r4 (6.099 g=1 @688; 6.266 g=16 @1376).
- Failure: r4lr ~ r4trunc — factored optimization cannot reach the gain
  either; would imply the tail map is intrinsically high-rank and push
  toward structured alternatives (shared down-sketch + learned scaling).
- Selftest guarantee: ALS in-sample loss <= post-hoc truncation's
  (verified on CPU), so a large eval regression would indicate
  generalization, not optimization, failure.

## Reproducibility
- Git tag: `exp/2026-08-04_fusion-r4lr` (commit 224879d md5 2248...,
  main)
- Job IDs: 050-20260805-000526-fusion-r4lr-g16 (g=16, r_sk in
  {688,1376}, out llama2-7b-g16lr); 050-20260805-000528-fusion-r4lr-g1
  (g=1, r_sk=688, out llama2-7b-g1lr). Both -H a100-40-2 -g 1 -m 32,
  verify clone workdir.
- Command core: 06_refit_fusion.py --arms r4lr --als_iters 3 (default),
  rank 256, input_k 1536, s 0.9, lam 0.1, calib wt103 128x2048, lpp=1.
- Key deps: gateway conda `larosa`; bf16-projection eval path.
- Selftest (CPU) 7/7 pre-submit: adds "ALS in-sample <= post-hoc
  truncation in-sample" on tiny dims.

### Results
Both jobs STATUS=ok (runs 20260805-000526/-000528; artifacts
~/workspace/fusion/llama2-7b-{g16lr,g1lr}/fusion3_results.json).
All at s=0.9, lam=0.1, r_t = r_sk, 3 ALS rounds.

| setting | post-hoc trunc | **ALS (r4lr)** | full-map r4 | deploy compute (r4lr) |
|---|---|---|---|---|
| g=1, r_sk=688 | 6.434 | **6.1225** | 6.099 | ~0.39 |
| g=16, r_sk=1376 | 7.238 | **7.0103** | 6.266 | ~0.62 |
| g=16, r_sk=688 | (not run) | 8.6279 | (not run) | ~0.39 |

- **Per-token: the deployable-form problem is SOLVED.** ALS recovers
  essentially all of the full-map gain (6.099 -> 6.123, only +0.024)
  at rank-688 cost. The deployable arm now sits BELOW the linear
  ceiling (6.150) and below R2 (6.195) — the nonlinear gain survives
  deployment when the low-rank structure is imposed during learning
  instead of after.
- **Block regime: partial.** ALS beats truncation (7.238 -> 7.010) but
  a substantial gap to the full map remains (6.266); and halving the
  sketch rank collapses quality (8.63) — in the block regime both the
  estimate and its output map need more capacity. The g=16 tail map
  appears intrinsically higher-rank (one mask serves 16 heterogeneous
  tokens, so the residual the tail must express is richer).
- Consistent with the selftest guarantee (ALS in-sample <= truncation's),
  the eval orderings show no optimization anomaly.

### Interpretation
