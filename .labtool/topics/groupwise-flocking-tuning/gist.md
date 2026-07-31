# groupwise-flocking-tuning — Can local ℓ2,1 tuning teach token groups to share neurons?

## Status
active — started 2026-07-31 (Idea D: CLS-aware Sparse-BRECQ). Parent doc:
research-wiki `plans/predictor-group-sparsity-research-direction.md` (§5
local reconstruction loss, §6.1 BlockFFN CLS objective, §4 PTQ-tool
transfer). Self-contained context was provided by the user on 2026-07-31
(preserved in the init journal card).

## Notation
```
FFN(x) = W_d i,  i = u ⊙ g,  u = W_u x,  g = σ(W_g x)
d = intermediate size (LLaMA2-7B: 11008), K = ⌊(1−s)·d⌋
S_t = Top-K surviving-neuron set of token t
C(δ) = E_t[|S_t ∩ S_{t+δ}|] / K_eff  (containment overlap); chance = K_eff/d
group = g consecutive tokens of one sequence (prefill chunk semantics)
ℓ2,1 group penalty: λ_g Σ_j √(Σ_{t∈group} i_j^{(t)2})  (column-wise group lasso
  on the group's activation matrix — pushes whole neurons off per group)
```

## Hypothesis
Measured constraint (larosa-intermediate-sparsity, 2026-07-24): at s=0.9
within-group overlap is only ~0.19–0.32 (1.9–3.2× chance) — training-free
group sharing loses ~70% of per-token selection. BlockFFN showed a
chunk-union (CLS-aware) objective CAN create flocking, but only via
from-scratch pretraining.

**Idea D**: graft the CLS objective onto the backprop-free local
reconstruction loss (parent doc §5) as a group×neuron ℓ2,1 penalty:

  L(θ) = E‖Λ^{1/2}(ŷ − y*)‖² + λ_g Σ_j √(Σ_{t∈group} i_j^{(t)2})

so that cheap block-local tuning (GPTQ-class cost) increases within-group
neuron-selection overlap ("flocking") without pretraining. Success axes
(pre-defined): (1) C(δ≤g) rises from 1.2–3.2× chance toward materially
higher; (2) at matched effective budget, group-shared mask PPL closes the
gap to the per-token anchor (dense 5.4736; per-token s=0.5/0.7/0.9 →
5.5210/5.7296/8.1083).

## Key Findings
(none yet)

## Dead Ends
(none yet)

## Open Questions
Design questions from the kickoff context (§4), in order:
1. How to combine ℓ2,1 with the pure-quadratic masked-Gram structure —
   proximal/ISTA step vs reweighted-ℓ2 (IRLS) that stays quadratic per
   iteration?
2. i is upstream of the §5 parameter set (W_up/W_gate frozen), so the ℓ2,1
   term is CONSTANT w.r.t. θ as originally scoped — the flocking lever
   requires relaxing the downstream-only principle. Candidates: (a) limited
   W_up update (TurboBoA Prop 3.1 style), (b) soft-mask probability
   penalty, (c) column-group structure on W̃_down only. Which breaks the
   principle least?
3. Group boundary: prefill fixed-g chunks vs decode time-window (δ≈16
   decay already caps the decode side). First target?
4. Layer-wise λ_g schedule — layer 31 needs no learning (already
   concentrated, Gini 0.705).
5. Evaluation ladder: post-tuning oracle C(δ) → group-mask PPL → (later)
   predictor coupling.

## Next Experiments
1. ~~Resolve design questions 1–2~~ — DONE 2026-07-31: IRLS (reweighted-ℓ2,
   quadratic per iteration) over ISTA; limited W_up update (option a) with
   W_gate frozen + score gauge-fixed to original W_down; PoC objective =
   dense reconstruction + penalty (design A). Rationale in the experiment
   card 2026-07-31_experiment-flocking-poc-l16.md.
2. ~~Single-layer PoC~~ — SUBMITTED (see Active Jobs).
3. (queued, contingent on PoC) design-B arm: frozen group-mask
   reconstruction objective; ISTA fallback if IRLS under-sparsifies;
   layer-wise λ_g schedule; multi-layer + PPL ladder.

## Active Jobs
- `050-20260731-171738-flocking-poc-l16` (a6000-4 pinned, PENDING) —
  single-layer ℓ2,1 IRLS PoC, LLaMA2-7B layer 16, s=0.9,
  g∈{16,64} × λ_rel∈{0,1e-3,1e-2,1e-1,1}. Card:
  journal/2026-07-31_experiment-flocking-poc-l16.md

## Boundaries (do not duplicate)
- Idea C (co-activation neuron permutation/blocking): separate topic
  `coactivation-block-structure` — permutation axis, no weight updates.
  This topic changes weights to CREATE overlap instead.
- Idea A (backbone + residual two-tier mask): training-free control /
  baseline for this topic.
- RB-Sparse (Dowon Kim): rotation + block-shared masks — no overlap with
  this rotation-free axis.
- oracle-residual-sparsity (paused): shares the weight-aware score
  definition ‖W_d[:,j]‖·|i_j| as common material.

## Pointers
- Overlap measurements: `topics/larosa-intermediate-sparsity/journal/`
  `2026-07-24_experiment-larosa-llama2-topk-overlap.md` + report
  `results/reports/2026-07-24_llama2-topk-overlap.{html,pdf}`.
- Code: `larosa/` — `config.sparse_mode='topk_intermediate'` verified
  (s=0 ≡ dense bitwise); hook example `scripts/analyze_topk_overlap.py`.
- Hosts: a6000-4 (venv `~/workspace/venv-larosa`, `--attn sdpa`,
  `/raid/LLM/llama2-7b`, GitHub fetch 불가 — sync via gateway tar/scp) or
  gateway a100-40-2 (conda env `larosa`).
- eval_ppl.py mlp/attn sparsity labels are SWAPPED (upstream bug) — use
  direct hooks.
