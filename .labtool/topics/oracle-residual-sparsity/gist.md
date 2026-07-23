# oracle-residual-sparsity — Mean-gate residual decomposition + rank-r compensation (oracle)

## Status
active

## Authoritative spec
`spec.md` in this topic (preserved verbatim, 2026-07-22). This gist is the
working summary; when they disagree, spec.md wins for the experiment design and
this gist wins for infra adaptations decided afterwards.

## Notation
FFN(x) = W_d i, i = u ⊙ g, u = W_u x, g = σ(W_g x) — σ is the model's own
act_fn (SiLU for LLaMA/Qwen, GELU-tanh for Gemma; never hardcode).
- ḡ[j] = E[g_j] (calibration mean), ḡ*[j] = E[u_j²g_j]/E[u_j²]
- residual r_j = u_j·(g_j − ḡ[j]); col_norm[j] = ‖W_d[:,j]‖₂
- Compensation M = W_d diag(ḡ) W_u ∈ R^{h×h}; rank-r via SVD, runtime 2hr/token

## Hypothesis
Oracle setting (mask from true activations; no predictor/kernel/training):
- **H1**: r = u⊙(g−ḡ) is more concentrated than i = u⊙g, so the mean-gate
  decomposition (sparse residual + linear tail compensation) holds accuracy at
  higher sparsity for equal effective compute.
- **H2**: weight-aware scoring |i_j|·col_norm[j] alone already beats plain |i_j|.
- **H3**: some r ≤ h/8 exists where rank-r approximation of M does not eat H1's gain.

Ablation ladder (spec §2): C0 dense / C1 |i| top-p / C2 +col_norm /
C3 residual score + exact tail compensation (diagnostic, not deployable) /
C4 rank-r deployable form / C5 ḡ* variant / C6 optional group mask (G=8/32).
Selection is top-p per token per layer; report axis is ACHIEVED sparsity.
Judgment: critical sparsity = max achieved sparsity with normalized accuracy
≥ 0.99 (lm-eval zero-shot 8-task suite); wikitext PPL as secondary signal.
Go/no-go (spec §8): Go if 8B C4(r≤d/8) critical sparsity ≥ C2 + 15%p;
No-go if C3 − C2 < 5%p.

## Design decisions (2026-07-22 pivot Q&A)
- **Selection = top-K, not top-p (user, 2026-07-23)**: PPL sweeps use exact
  per-token sparsity s (K = int((1−s)·d), same tie semantics as top_k_new) so
  C1 exactly reproduces the larosa topk_intermediate setup and **C3 vs C1 at
  equal s is the primary readout** (matched per-token compute; no achieved-
  sparsity interpolation). The spec's top-p mask stays implemented
  (select=topp) for spec-faithful runs; unit test pins C1-topk ==
  topk_intermediate bitwise. s grid: {0.5, 0.7, 0.9} only (user, 2026-07-23 —
  match the existing Top-K experiment's levels exactly).
- **Scope narrowed (user, 2026-07-22)**: LLaMA2-7B ONLY (h=4096, d=11008), to
  compare directly against the larosa Top-K results; evaluation is the SAME
  wikitext-2 PPL pipeline (`eval_ppl_wikitext_with_inference_sparsity`) — no
  lm-eval, no critical-sparsity-by-accuracy; C4 runs r = h/8 = 512 only.
  The spec's multi-model/accuracy program is deferred, not canceled.
- **No R-Sparse fork** (user decision): implement standalone in EfficientAI on
  the existing HF loading/eval pipeline; the spec's "reuse from R-Sparse" items
  (model loading, calibration loader, SVD util, lm-eval integration) are
  implemented fresh. Spec §3's "disable R-Sparse modules" is trivially
  satisfied — none exist here. Contamination rule still binding: attention and
  all linears stay dense; only the MLP forward is wrapped.
- Old topic's 3-model Top-K PPL extension preserved as backlog in
  larosa-intermediate-sparsity gist (not orphaned).
- All sparsification simulated as compute-then-mask (oracle-equivalent).

## Key Findings
- **Phase-0 distribution report, LLaMA2-7B (2026-07-23): H1 go with caveats.**
  Residual r = u⊙(g−ḡ) is more top-p-concentrated than i in 30/32 layers;
  mean induced-sparsity gap ≈ +3%p (p=0.7→0.9), peaking +5%p mid-stack
  (layers 6–18) and INVERTING in layers 30–31 (−0.2/−2.6%p). ḡ corpus
  stability is weak early (c4↔wikitext103 Pearson 0.48–0.53 at layers 2–4,
  ~0.9 mid-stack). M is not strongly low-rank mid-stack: r=512 retains only
  0.54–0.60 of Frobenius energy (layers 4–17), vs 0.94–0.985 at layers 30–31.
  When relevant: (a) if C3/C4 underperform, try exclude_layers=[30,31];
  (b) C4-vs-C3 gap is the H3 signal — the heavy singular tail predicts risk;
  (c) early-layer C3/C5 results carry calibration-corpus noise.
  Journal: 2026-07-22_experiment-oracle-llama2-phase0-calib.md
- **Phase 1 complete (2026-07-22)**: oracle conditions implemented as
  `sparse_mode='oracle'` in the existing modeling file
  (`inference/oracle_mlp.py` + small hooks in modeling_llama_larosa.py);
  scripts 01–04 + sweep runner under `scripts/oracle/`. All 4 spec-§2 unit
  tests pass on CPU fp32 tiny model: p=1 identity 9e-8, C4 full-rank ≡ C3
  3e-7, C4 p=1 error exactly (M̂−M)x (residual 1e-9), mask-vs-slice 2e-9;
  plus save/load round-trip bitwise-identical and topk_intermediate
  regression smoke OK.
  When relevant: trusting the c3/c4 algebra — the compensation identities
  were verified numerically, so PPL differences in later phases are signal,
  not implementation bugs.

## Dead Ends
(none yet)

## Open Questions
- ~~Rank grid~~ → resolved: r = 512 (h/8) only (user, 2026-07-22).
- ~~Model/eval choices~~ → resolved: LLaMA2-7B + wikitext-2 PPL (user).
- **Calibration corpus download on gateway**: 01_calibrate defaults to
  allenai/c4 en streaming — network access from execution servers worked for
  wikitext, c4 streaming untested. Fallback flag: `--dataset wikitext103`.
- **PPL success threshold**: same open convention as the old topic — judge
  ΔPPL vs dense (5.4736) against the Top-K anchors (C1-equivalent: 5.521/
  5.730/8.108 at s=50/70/90); spec's §8 %p margins translate to "C3/C4 hold
  ≤ Top-K's ΔPPL at ≥15%p higher achieved sparsity" — formalize once curves exist.

## Next Experiments
1. **Phase 3 — dense + C1 top-K sweep (LLaMA2-7B)**: oracle_ppl_sweep.sh
   (SELECT=topk) dense + c1 over s grid ×9. Sanity gate: C1 at s=0.5/0.7/0.9
   must REPRODUCE the topk_intermediate anchors (5.521/5.730/8.108) —
   selection is now bitwise-identical, so any deviation is pipeline noise
   only (~±0.1 at most, likely exact).
2. **Phase 4 — C2–C5 top-K sweep (LLaMA2-7B)**: one job per condition (c2,
   c3, c5; c4 with r=512, stats/factors already built). Primary readout:
   **ΔPPL(C3) vs ΔPPL(C1) at equal s** — how much does mean-gate
   compensation buy at matched compute. Then C4−C3 (H3), C2−C1 (H2). If
   C3/C4 underperform, variant with exclude_layers=[30,31].

## Active Jobs
- `050-20260723-234944-oracle-llama2-phase3-c0c1` — Phase 3 gate (dense + C1
  top-K). Journal: 2026-07-23_experiment-oracle-llama2-phase3-c0c1.md
- `050-20260723-234448-oracle-llama2-hist` — i/r histograms for the report.

## Pointers
- Spec: `spec.md` (this topic). Pivot record:
  `../larosa-intermediate-sparsity/journal/2026-07-22_pivot-oracle-residual-sparsity.md`
- Prior confirmed result to beat (C1-like, Top-K, PPL axis): LLaMA2-7B
  intermediate Top-K 50% → +0.047 PPL (larosa-intermediate-sparsity Key Findings).
- Code base: EfficientAI `larosa/inference/mlp.py` (existing mode plumbing:
  config.sparse_mode flag pattern from commit 40edf40), conda env `larosa` on
  gateway (flash-attn 2.7.4.post1 pinned).
- Models on gateway: /raid/LLM/llama2-7b, /raid/LLM/llama3-8b (read-only),
  ~/workspace/models/Qwen2.5-7B. HF cache symlinked to ~/workspace/cache.
- Compute note: M is h×h (8B: 4096² × 32 layers ≈ 1.1GB fp32 total) — SVD per
  layer fits easily on any A100; calibration + Phase-0 are single-GPU jobs.
