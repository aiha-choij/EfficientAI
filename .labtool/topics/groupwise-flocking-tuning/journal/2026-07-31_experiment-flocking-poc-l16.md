# 2026-07-31 — experiment: flocking-poc-l16

Status: PENDING

## Hypothesis tested
A group x neuron l2,1 penalty inside a backprop-free local reconstruction
objective (W_up via IRLS + batched CG, W_down via anchored closed-form
refit, W_gate frozen, score gauge-fixed to original W_down column norms)
increases within-group neuron-selection overlap on a single FFN layer,
without a large dense-reconstruction cost.

## What we're testing over alternatives
- vs permutation (Idea C / coactivation-block-structure): weights CHANGE
  here to create overlap; permutation is function-preserving reordering.
- vs BlockFFN: same CLS-flavored objective but block-local tuning instead
  of from-scratch pretraining.
- Design choices under test (resolved 2026-07-31, gist Open Questions 1-2):
  IRLS (reweighted-l2, stays quadratic) over ISTA; limited W_up update
  (option a) over soft-mask (b) / W_down-only grouping (c); PoC objective =
  DENSE reconstruction + penalty (design A — mask enters only at eval).

## Prior art check
- local-loss-refit (gateway topic): 0-shrinkage ridge prior is a known
  confound → both solves here are W-anchored from the start.
- oracle-residual-sparsity whitening round: offline-L2 improvements can be
  anti-correlated with downstream quality → judgment on overlap/union/recon
  metrics (later PPL), never on the training objective value.
- coactivation P2/P3: beating a random control is not enough; absolute
  coverage is what matters → success gate uses absolute overlap too.
- larosa-intermediate-sparsity 2026-07-24: pre-tuning baselines
  (within-group overlap ~0.19-0.32 at s=0.9; union tax 6.0-6.3x at g=16;
  C(1)=0.316, chance 0.100).

## Expected outcome
- Success: some lambda_rel where within-group overlap reaches >= 0.5
  absolute OR >= 1.5x its pre-tuning value, with union tax visibly down and
  dense reconstruction relative error staying below the group-masked
  reconstruction error scale; lambda=0 sanity arm shows ~zero drift and
  unchanged metrics.
- Failure: overlap gain < 1.2x at every lambda, or gains only with
  runaway reconstruction error / weight drift. (Criteria provisional —
  finalize on first numbers, house convention.)

## Reproducibility
- Git tag: `exp/2026-07-31_flocking-poc-l16` (commit ad8c1b3)
- Job ID: 050-20260731-171738-flocking-poc-l16
- Assigned host/GPU: a6000-4 (pinned via -H), GPU [pending dispatch]
- Command: `bash -c "/home/choij/workspace/venv-larosa/bin/python
  scripts/flocking/poc_layer_tune.py --model /raid/LLM/llama2-7b --layer 16
  --sparsity 0.9 --group_sizes 16,64 --lambdas 0,1e-3,1e-2,1e-1,1
  --attn sdpa --out /home/choij/workspace/flocking/llama2-7b-l16"`
  (qsub `-H a6000-4 -g 1 -m 40 -d /home/choij/workspace/repos/EfficientAI/larosa`)
- Config path: n/a — parameters as script args (defaults: train/test 32x2048
  wikitext-2 train/test tokens, outer=2, irls=3, cg_iters=25, cg_tol=1e-4,
  mu_rel=0.03, lambda_down=0.01, K=floor(0.1*11008)=1100)
- Key parameters: LLaMA2-7B layer 16 only; s=0.9; trained group sizes
  g in {16,64}; lambda_rel in {0, 1e-3, 1e-2, 1e-1, 1} — NOTE lambda is
  normalized (dimensionless): penalty weight scales with a_med/a_j (output
  column importance) and s0 (median group-column norm), so the grid is
  relative, not absolute; the proposal's "1e-4..1e-2 + scale normalization"
  materialized as this 5-point relative grid.
- Key deps: python 3.10 venv `~/workspace/venv-larosa` (torch 2.6.0+cu124,
  transformers 4.46.3, datasets; sdpa — no flash-attn on a6000-4)
- Sync: local main ad8c1b3 pushed to origin; script scp'd mac -> a6000-4
  directly (md5 b8c9d6fc verified). Gateway repo deliberately NOT pulled —
  an active agent session (block-sparse-compensation request) owns that
  working tree on another branch.
- Selftest: `--selftest` (CPU) passed 4/4 before submission: CG matches
  direct solve; lambda=0 recovers W_up (rel err < 1e-3); large lambda
  shrinks group column norms > 20%; anchored refit recovers W_down.

## Notes
- Metrics emitted per (g, lambda): C(delta) for delta in {1..64},
  within_group_overlap_g{16,64}, union_tax_g{16,64},
  dense_recon_relerr, group_mask_recon_relerr_g{16,64} (shared mask from
  group-aggregated gauge-fixed score), wup/wdown drift. Baseline entry uses
  original weights, same pipeline.
- Cross-metric note: each trained (g, lambda) arm reports metrics at BOTH
  eval group sizes — transfer of flocking across g is visible for free.

### Results

### Interpretation
