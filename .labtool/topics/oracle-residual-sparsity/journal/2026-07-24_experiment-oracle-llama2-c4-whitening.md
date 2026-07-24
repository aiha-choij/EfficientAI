# Experiment: oracle-llama2-c4-whitening

Status: DONE (2026-07-24)
Date: 2026-07-24

## Hypothesis tested
The C4 (rank-r compensation) collapse is caused by (a) a flat spectrum of M
and/or (b) plain SVD ignoring the input distribution. Whitened SVD —
factorizing M @ C with C = cholesky(E[xx^T] + eps I) so the approximation
minimizes error under the REAL input distribution — plus per-layer rank
allocation by whitened spectral energy, should close a meaningful part of the
C4−C3 gap at equal (mean) rank.

## What we're testing over alternatives
Whitening targets cause (b) directly and is measurable offline before any
PPL run (Task-0 trunc_err). Per-layer allocation exploits the strong layer
spread in spectral concentration (Frobenius retention 0.54→0.985) instead of
wasting rank on already-low-rank late layers. Alternative "just raise uniform
r" is captured as the r_bar=1024 arm for comparison.

## Prior art check
- Phase 4 (2026-07-24 card): C4 plain r=512 worse than C1 everywhere; comp
  error +0.78 PPL at s=0.5; C4−C3 gap 0.75/0.93/2.13. Baseline preserved at
  factors/r512 + results/c4_topk_s*.json.
- Phase 0: r=512 Frobenius energy 0.54-0.60 mid-stack, 0.94/0.985 layers
  30/31 → allocation headroom.
- Unit test (commit tagged): whitened full-rank reconstructs M to 9e-7;
  whitened < plain on anisotropic synthetic input.

## Expected outcome
- Task 0: trunc/tail ratio should show plain-SVD error comparable to or
  exceeding the tail signal at s_op=0.7/0.9 (explaining the collapse), and
  whitened trunc_err clearly lower than plain at r=512.
- Success: C4-whitened (r=512 uniform or alloc r_bar=512) recovers a large
  share of the C4−C3 gap — concretely, beats C1 at s=0.7/0.9 (plain C4 did
  not) and moves toward C3 (+0.155/+1.164). Bonus: alloc beats uniform at
  equal r_bar; r_bar=256 usable.
- Failure: whitened ≈ plain → cause (a) flat spectrum dominates; rank-r
  compensation dead end at practical r → rethink (e.g., structured/sparse
  compensation) with user.

## Reproducibility
- Git tag: exp/2026-07-24_oracle-llama2-c4-whitening
- Job ID: 050-20260724-052033-oracle-llama2-c4-whitening
- Assigned host/GPU: a6000-2 (pinned), 1 GPU >=30GiB [pending dispatch]
- Command: `bash -c "export PY=$HOME/workspace/venv-larosa/bin/python;
  scripts/oracle/run_whitening.sh /raid/LLM/llama2-7b
  $HOME/workspace/oracle/llama2-7b"`
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a6000-2)
- Pipeline: 01 --xxt (reuses stats/c4/calib_tokens.pt, sigma-only save) →
  03 plain spectra + wht_r512 + wht_alloc{256,512,1024} (eps=1e-4·mean diag)
  → 07 diag (s_op 0.7/0.9, 4 seqs) → 04 c4 sweeps ×3s ×4 variants (top-K)
- Baselines preserved: factors/r512, results/c4_topk_s*.json untouched
- Env: a6000-2 venv (torch 2.6.0+cu124, transformers 4.46.3, sdpa), RTX A6000

### Results
Main job (050-20260724-052033) completed; commit on server c50c55f. Results
JSONs c4_wht_*.json + diag_trunc_vs_tail.csv + spectra_{plain,wht}.json under
oracle/llama2-7b/results/ (a6000-2, fetched to host).

C4 PPL by factor variant (dense 5.4738; C1/C3/plain-r512 from phase 3-4):

| s   | C1     | C3     | plain r512 | wht r512 | wht alloc256 | wht alloc512 | wht alloc1024 |
|-----|--------|--------|-----------|----------|--------------|--------------|---------------|
| 0.5 | 5.5216 | 5.5051 | 6.2537    | 6.4859   | 9.2926       | 7.0903       | 5.9570        |
| 0.7 | 5.7284 | 5.6283 | 6.5563    | 6.8483   | 10.0913      | 7.4886       | 6.1641        |
| 0.9 | 8.1096 | 6.6381 | 8.7638    | 9.7606   | 17.4118      | 10.2638      | 7.6336        |

Task-0 diagnostics (4 calib seqs, means over 32 layers):
- trunc_err E||(M_hat−M)x||: plain 2.239 vs whitened 1.941 (−13%) — whitening
  DOES reduce the L2 objective it optimizes, on real calibration inputs.
- trunc/tail ratio: plain 0.80 (s_op=0.7) / 0.59 (0.9); worst layer 7 (~1.0)
  — the r=512 compensation error is the same order as the tail signal it
  replaces.
- Whitened spectra are strongly concentrated: stable rank 16-129 per layer
  (vs plain's heavy tail), which is why tau-allocation assigns tiny ranks
  (budget 256 → min r_l = 5).

Key contradictions:
1. Whitened uniform r512 is WORSE in PPL than plain r512 at every s
   (6.486/6.848/9.761 vs 6.254/6.556/8.764) despite 13% lower L2 error.
2. At matched budget 512, whitened+alloc (7.09/7.49/10.26) also loses to
   plain uniform. alloc256 is catastrophic (9.3/10.1/17.4).
3. wht_alloc1024 is the best C4 yet (5.957/6.164/7.634; first C4 to beat C1
   at s=0.9) — but rank doubled, so whitening-vs-rank is confounded.

Completion arms (job 050-20260724-064258, DONE) — full C4 variant table
(dense 5.4738; C1/C3 for reference):

| s   | C1     | C3     | plain512 | wht512 | plain1024 | wht1024 | alloc1024 | alloc512 | alloc256 |
|-----|--------|--------|----------|--------|-----------|---------|-----------|----------|----------|
| 0.5 | 5.5216 | 5.5051 | 6.2537   | 6.4859 | **5.7365**| 5.7668  | 5.9570    | 7.0903   | 9.2926   |
| 0.7 | 5.7284 | 5.6283 | 6.5563   | 6.8483 | **5.9152**| 5.9638  | 6.1641    | 7.4886   | 10.0913  |
| 0.9 | 8.1096 | 6.6381 | 8.7638   | 9.7606 | **7.2294**| 7.3974  | 7.6336    | 10.2638  | 17.4118  |

2x2 decomposition at fixed budget:
- rank 512→1024 (plain uniform): −0.52/−0.64/−1.53 PPL — the only lever that
  works. plain_r1024 is the best C4: beats C1 at s=0.9 (7.23 < 8.11), gap to
  C3 shrinks to +0.23/+0.29/+0.59.
- whitening at fixed rank: +0.23/+0.29/+1.00 (r512), +0.03/+0.05/+0.17
  (r1024) — consistently harmful despite reducing E||(M_hat−M)x|| by 13%.
- tau-energy allocation at fixed budget: harmful at every budget
  (alloc1024 > wht1024 uniform > plain1024; alloc512 >> plain512).

### Interpretation
(provisional — presented to user with the completed table)

Both spec hypotheses about the C4 collapse are now answered: the cause is
(a) the flat (heavy-tailed) spectrum of M, NOT (b) plain SVD ignoring the
input distribution. Evidence: whitening achieves its own objective (−13%
E||(M_hat−M)x|| on real calibration inputs) yet makes PPL worse at every
rank and budget — input-distribution L2 is misaligned with downstream loss,
consistent with important low-variance (outlier-ish) directions being
down-weighted by whitening. Allocation fails for the same reason: it trusts
the whitened energy metric, starving early/mid layers (budget 256 → r_l=5).
The lever that works is brute rank: plain uniform r=1024 (compute +6.2%,
2r/3d) more than halves plain-r512's excess over C3 at s=0.9 and finally
beats C1. Decision point: r=2048 (+12.4%) to test convergence to C3 vs
accepting r=1024, vs output-side-weighted objectives as a smarter direction.
