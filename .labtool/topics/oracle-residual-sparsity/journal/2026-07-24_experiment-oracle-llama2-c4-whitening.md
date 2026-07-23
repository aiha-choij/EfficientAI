# Experiment: oracle-llama2-c4-whitening

Status: PENDING
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
(pending)

### Interpretation
(pending)
