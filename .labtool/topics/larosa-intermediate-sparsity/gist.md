# larosa-intermediate-sparsity — Per-token Top-K on the FFN intermediate activation

## Status
done — hypothesis confirmed on LLaMA2-7B (2026-07-22); thread closed by pivot
to `oracle-residual-sparsity` (see journal/2026-07-22_pivot-oracle-residual-sparsity.md)

## Notation (used throughout this topic)
FFN(x) = W_d i,  i = u ⊙ g,  u = W_u x,  g = σ(W_g x)

- `x`: FFN input (h-dim), `i`: intermediate activation (d-dim), `σ`: SiLU
- `u_j`, `g_j`, `i_j`: j-th element (scalar) of u, g, i
- `W_d[:,j]`: j-th column of W_d (h-dim vector), `W_d[j,:]`: j-th row
- d (intermediate size): LLaMA2-7B 11008, LLaMA3-8B 14336, Qwen2.5-7B 18944

## Hypothesis
Per-token magnitude Top-K sparsification of the FFN intermediate activation
i = u ⊙ g — FFN-only, down-projection input only, original basis (no rotation),
attention left dense — retains wikitext-2 PPL well at high sparsity
(s = 50/70/90%), because the intermediate activation is the most
sparsity-tolerant FFN component across models and scales
(universal-properties study, arXiv:2509.00454, Finding 1; research-wiki
`wiki/papers/universal-properties-activation-sparsity.md`).

Mechanism: for each token vector i ∈ R^d, keep the K = round((1−s)·d) elements
with largest |i_j|, zero the rest. s = fraction of zeroed neurons per token
vector. No calibration, no histograms, no rotation matrices — the Top-K
mechanism is data-free at setup time.

## Design decisions (2026-07-22 pivot Q&A)
- **No rotation**: Top-K in the original basis of i. The saved D matrices are
  h-dim (input-side) and unused here. Rotated-basis intermediate Top-K is the
  future RB-Sparse step, built on this baseline.
- **Sparsify only i**: attention q/k/v/o and FFN gate/up input paths are dense.
- **Baseline**: dense (s=0) ΔPPL only; no matched-s input-mode arm (the
  input-vs-intermediate ranking is accepted from the paper's evidence).
- **Model order**: LLaMA2-7B first (implementation validation), then
  LLaMA3-8B + Qwen2.5-7B.

## Key Findings
- **LLaMA2-7B intermediate Top-K (2026-07-22)**: hypothesis confirmed on the
  first model — with no rotation and no calibration, per-token Top-K on i gives
  wikitext-2 PPL 5.5210 at s=50% (+0.047 vs dense 5.4736), 5.7296 at 70%
  (+0.256), 8.1083 at 90% (+2.635); measured sparsity of i ≈ s at every level
  and s=0 is dense-identical to 4 decimals. Intermediate 50% ≈ input-mode 25%
  (5.5017), and intermediate 70% beats input-mode 50% (5.8167).
  When relevant: choosing the sparsification target for any LaRoSa-derived
  design — the intermediate point buys roughly 20-45pp extra sparsity over the
  input side at equal PPL on LLaMA2-7B.
  Journal: 2026-07-22_experiment-larosa-llama2-topk-int-ppl.md
- **Eval log labels are swapped (upstream bug, utils/eval_ppl.py:136-140)**:
  `eval_ppl_wikitext_with_inference_sparsity` appends `layer.mlp.*` into the
  attn lists and vice versa, so the printed "attn h1/h2" are really MLP values
  and "mlp h1/h2" are attention values.
  When relevant: reading measured-sparsity lines in any PPL log from this
  pipeline (incl. the upcoming LLaMA3/Qwen sweeps) — un-swap before judging
  placement gates.

## Dead Ends
(none yet)

## Open Questions
- Success threshold: the universal-properties paper defines critical sparsity
  via ≥99% of dense *accuracy*; no equivalent convention for PPL. For now we
  record ΔPPL vs dense at each s and judge against the trusted LaRoSa-mode
  anchors (e.g. LLaMA2-7B input-mode 50% → 5.87, ΔPPL +0.40) — decide a formal
  criterion once first numbers exist.
- Scoring refinement: plain |i_j| ignores how much neuron j actually contributes
  to the output — ‖W_d[:,j]‖·|i_j| (WiSparse/TEAL-style weight-aware scoring) is
  the obvious next arm if plain magnitude degrades early.
- Prefill vs decode: PPL eval is fully teacher-forced, so the distinction is
  invisible here; matters once generation/lm_eval enters.
- Empirical sparsity check: log measured zero-fraction of i per layer
  (count_zero_solo) and confirm ≈ s (guards against K off-by-one / masking bugs).

## Next Experiments
(none — topic closed by 2026-07-22 pivot; the weight-aware scoring arm is
absorbed as condition C2 of `oracle-residual-sparsity`)

## Future work
- **Backlog (preserved at pivot, user decision)** — 3-model extension:
  LLaMA3-8B (dense 6.1377) + Qwen2.5-7B (dense 6.8497), same s=0/0.5/0.7/0.9
  sweep via `repro_topk_ppl.sh {llama|qwen} <model_dir>`, one job per model.
  Why: universality claim needs a newer LLaMA and a different family
  (d = 14336 / 18944). Success: s=0 ≡ dense baseline ±0.1 per model + full
  3×3 ΔPPL table; hypothesis strengthened if the 50%-near-lossless pattern
  (ΔPPL ≲ 0.1) holds on both. May still be worth running as a PPL-side
  sanity line alongside the new topic's accuracy results.
- RB-Sparse: rotated-basis variant of intermediate Top-K (block-shared mask +
  eigenspace low-rank compensation; 2026-06-24 discussion, research-wiki r-sparse
  note). Needs a d-dim rotation (pass-2-class compute — run on a100-80).
- Weight-aware scoring arm (‖W_d[:,j]‖·|i_j|) if plain magnitude underperforms.
- lm_eval accuracy for the new mode (replaces the old-mode packaging task).

## Active Jobs
- (none)

## Pointers
- Motivating paper: arXiv:2509.00454, research-wiki
  `wiki/papers/universal-properties-activation-sparsity.md` (Finding 1, Fig. 2).
- Trusted baseline: larosa-repro topic — dense PPL 5.47 / 6.13 / 6.85; its
  pipeline reproduces paper Table 2 within ±0.1 (see its gist Key Findings).
- Code: `larosa/inference/mlp.py` (current modes: `_mlp_forward` threshold,
  `_mlp_forward_larosa` rotated threshold), `larosa/scripts/ppl_test_larosa_*.py`.
- Models on gateway: `/raid/LLM/llama2-7b`, `/raid/LLM/llama3-8b` (read-only),
  `~/workspace/models/Qwen2.5-7B`.
