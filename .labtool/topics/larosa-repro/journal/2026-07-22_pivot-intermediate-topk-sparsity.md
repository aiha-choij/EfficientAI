# Pivot: LaRoSa reproduction → Intermediate activation Top-K sparsification

Date: 2026-07-22
Type: steer (topic closed as **done**, not failed)

## Previous direction (gist.md as it stood)

Reproduce LaRoSa (arXiv:2507.01299) Table 2 wikitext-2 PPL on LLaMA2-7B /
LLaMA3-8B / Qwen2.5-7B, then validate the accuracy half (Qwen lm_eval packaging)
and start RB-Sparse using the saved per-layer D matrices. LaRoSa's mode:
threshold-based sparsification (histogram calibration, α split h1=0.8p / h2=1.2p)
applied to attention q/k/v/o inputs AND FFN gate/up inputs (x, in rotated basis)
AND the down-projection input (i).

## Why we're pivoting

The reproduction goal was **achieved** (12/12 PPL points ±0.1 across 3 models) —
the thread ends by completion, and the sparsification mode itself is being
replaced. Per the universal-properties study (research-wiki
`universal-properties-activation-sparsity.md`, arXiv:2509.00454, Finding 1):
intermediate activations (i = u⊙g) are the most sparsity-tolerant FFN component
across all models and scales — consistently exceeding input, gate, and
up-projection sparsity. Input-side sparsity (LaRoSa's mode) is the most
constrained target. We therefore move the sparsification point to i.

## New direction

Topic `larosa-intermediate-sparsity`: FFN-only, per-token magnitude **Top-K on
the intermediate activation i** (down-projection input), original basis (no
rotation), target sparsity s ∈ {50, 70, 90}%. Attention is left dense.
Evaluation reuses the validated wikitext-2 PPL pipeline; LLaMA2-7B first, then
LLaMA3-8B + Qwen2.5-7B. Baseline comparison: dense (s=0) ΔPPL only.

Design decisions (user, 2026-07-22):
- **No rotation.** The existing D matrices are for the h-dim input x and have no
  use when only i (d-dim) is sparsified. A rotated-basis variant of intermediate
  Top-K is deferred to RB-Sparse.
- **New topic**, larosa-repro closed as done (its findings remain the trusted
  baseline).
- **Model order:** LLaMA2-7B validation first (incl. s=0 ≡ dense sanity check),
  then 3-model extension.
- **Baseline:** dense-only ΔPPL; no matched-s input-mode comparison (paper
  evidence accepted for the input-vs-intermediate ranking).

## Status of active experiments

None running. All three PPL reproduction cards are DONE
(2026-07-22_experiment-larosa-{llama2-7b,llama3-8b,qwen25-7b}-ppl.md).
The gateway watch request `20260723-084829-larosa-ppl-repro-watch` is complete.

Planned-but-not-started items from the old gist:
- **lm_eval accuracy packaging (Qwen2.5-7B-larosa 0.25)** — deprioritized, not
  orphaned: it validated the *old* mode's accuracy half, which is no longer the
  operating mode. Revisit only if LaRoSa-mode accuracy numbers are ever needed
  for a paper comparison table.
- **RB-Sparse (rotated-basis block-shared mask)** — carried forward: the new
  intermediate Top-K baseline is a natural prerequisite (RB-Sparse becomes
  "rotated variant of this"). Listed in the new topic's future work.

## Insights at time of pivot (verbatim from gist.md Key Findings)

- **Baseline reproduction (2026-07-22)**: paper Table 2 wikitext-2 PPL reproduced
  on all 3 models — LLaMA2-7B, LLaMA3-8B, Qwen2.5-7B — 12/12 points within ±0.1
  (max |Δ| 0.089); 10-seq calibration (vs paper's 16) sufficient.
  When relevant: any future LaRoSa-based experiment can trust this pipeline as
  its dense/sparse baseline; deviations beyond ~0.1 PPL signal a real change,
  not reproduction noise. Journals: 2026-07-22_experiment-larosa-{llama2-7b,llama3-8b,qwen25-7b}-ppl.md
- PPL reproduction needs no model packaging: `scripts/ppl_test_larosa_*.py` takes
  `--sparsity` and `--larosa_path` at runtime (packaging only matters for lm_eval).
- Sparsity 0.0 keeps everything in `top_k_new` → usable as a dense-equivalence
  sanity check of the rotation path.
- α is baked into the modeling code as sparse_level_h1 = 0.8p, h2 = 1.2p
  (not grid-searched per model as in the paper).

(Dead Ends and Open Questions remain in place in the closed gist.md.)
