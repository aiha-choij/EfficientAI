# Pivot: larosa-intermediate-sparsity -> oracle-residual-sparsity

Date: 2026-07-22
Type: steer (new topic)

## Previous direction (gist as of pivot)

Per-token magnitude Top-K sparsification of the FFN intermediate activation
i = u ⊙ g — FFN-only, down-projection input only, original basis (no rotation),
attention left dense — retains wikitext-2 PPL well at high sparsity
(s = 50/70/90%). Mechanism: keep K = round((1−s)·d) largest |i_j| per token,
zero the rest; no calibration, no histograms, no rotation matrices.
Validated on LLaMA2-7B against the trusted dense baseline (5.47); planned next:
(1) 3-model extension to LLaMA3-8B + Qwen2.5-7B, (2) weight-aware scoring arm
(‖W_d[:,j]‖·|i_j|) at s=0.7/0.9.

## Why we are pivoting

Not a failure — the hypothesis was CONFIRMED on LLaMA2-7B (s=50% → +0.047 PPL).
A full implementation spec arrived for the next-stage question: can a
mean-gate decomposition (r = u ⊙ (g − ḡ)) plus low-rank compensation push
critical sparsity beyond plain |i| Top-K at equal effective compute? The new
spec changes enough of the stack to warrant a fresh topic:
- Hypothesis: plain-magnitude Top-K → residual concentration + compensation
  (H1), weight-aware scoring (H2), rank-r compensation viability (H3)
- Selection: Top-K → top-p (report axis = achieved sparsity)
- Metric: wikitext-2 PPL → lm-eval zero-shot accuracy, critical sparsity
  (normalized acc ≥ 0.99), with PPL as a secondary continuous signal
- Models: LLaMA2-7B/LLaMA3-8B/Qwen2.5-7B → Llama-3.2-3B / Llama-3.1-8B /
  Gemma-3-4b-pt
- The old plan's weight-aware arm is absorbed into the new spec as condition C2.

## New direction

Oracle (activation-derived mask, no predictor/kernel/training) evaluation of
an ablation ladder C0–C6 on gated-MLP intermediate sparsity: C1 |i| baseline,
C2 weight-aware |i|·col_norm, C3 residual score with exact mean-gate tail
compensation (diagnostic), C4 deployable rank-r compensation via
M = W_down diag(ḡ) W_up, C5 u²-weighted ḡ*, C6 optional group masks.
Judged by critical sparsity at matched effective compute. Full spec preserved
verbatim in the new topic: `topics/oracle-residual-sparsity/spec.md`.
Implementation: standalone on the EfficientAI pipeline (user decision — no
R-Sparse fork; reimplement what the spec assumed reusable from R-Sparse).

## Status of active experiments

- No running or PENDING jobs at pivot time.
- Planned (not yet submitted) 3-model Top-K PPL extension (LLaMA3-8B +
  Qwen2.5-7B): preserved as backlog in the old gist's Future work (user
  decision), not orphaned — it may still run later as a PPL-side sanity line.

## Insights at time of pivot (preserved verbatim from gist.md)

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
