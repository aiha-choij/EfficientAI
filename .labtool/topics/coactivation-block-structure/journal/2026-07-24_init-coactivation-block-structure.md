# 2026-07-24 — init: coactivation-block-structure

## Initial hypothesis
Neuron permutation (a free, exactly function-preserving transform for
i = u ⊙ g) guided by co-activation statistics can reorganize the FFN
intermediate dimension into blocks such that group-shared Top-K masks
(g = 16/32/64 tokens) become block selections, absorbing the union
inflation that makes naive mask sharing pay ~68% of per-token signal
(prior measurement, larosa-intermediate-sparsity 2026-07-24).

Strong null: clustered blocks must beat random-permutation blocks by
≥ ~1.3× on P2 structure metrics, else the permutation axis is rejected.

## Starting phase
Running experiments — P1 (co-activation statistics collection) is fully
specified in the spec; P2 (clustering + structure evaluation) needs only
P1 outputs; P3 (block-mask oracle PPL) is gated on P2.

## Notes
- Source spec: `~/Workspace/research-wiki/plans/coactivation-block-structure-spec.md`
  (self-contained; derived from 2026-07-24 group-sparsity discussion, Idea C;
  parent: `plans/predictor-group-sparsity-research-direction.md` §2③, §3.2, §7).
- Prior measurements come from topic `larosa-intermediate-sparsity`
  (journal 2026-07-24_experiment-larosa-llama2-topk-overlap.md).
- Concurrent-session etiquette: `oracle-residual-sparsity` is active in
  another session — keep current.md edits to own topic row + Active Jobs.
- Boundary: RB-Sparse thread owns rotation-basis block masks; this topic is
  original-basis permutation only.
