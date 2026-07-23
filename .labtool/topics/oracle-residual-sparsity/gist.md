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
- **[MAIN] Phase 4 table (2026-07-24): H1 confirmed, H2 rejected, H3
  partial-go.** LLaMA2-7B, top-K matched s, wikitext-2 PPL (dense 5.4738):
  C3 (residual score + exact mean-gate compensation) cuts C1's degradation by
  34/39/56% at s=0.5/0.7/0.9 — s=0.9 PPL 8.110 → 6.638 (ΔPPL +2.636 →
  +1.164). C2 (col_norm score only) ≈ C1 at 0.5, worse at 0.7/0.9 → the gain
  is ALL from compensation, not the score. C4 (rank-512) collapses below C1
  everywhere (+0.78 PPL comp error already at s=0.5) — the mid-stack
  Frobenius deficit bites, exactly as Phase 0 flagged. C5 (ḡ*) consistently
  a bit worse than C3's plain ḡ.
  When relevant: (a) the deployable form needs bigger r — r=1024 costs ~6.2%
  compute (2r/3d), r=2048 ~12.4%; (b) don't bother with col_norm-only or
  ḡ* variants going forward.
  Journal: 2026-07-24_experiment-oracle-llama2-phase4-c2c5.md
- **Phase 3 gate passed (2026-07-24)**: oracle path under top-K reproduces the
  topk_intermediate anchors to 4 decimals on a DIFFERENT GPU (A6000) and
  attention backend (sdpa) — dense 5.4738 vs 5.4736; C1 5.5216/5.7284/8.1096
  vs 5.5210/5.7296/8.1083 (max Δ 0.0013). When relevant: backend/arch effects
  are ~1e-3, so any C2–C5 delta above ~0.01 PPL is real signal; also the C1
  row of the main table is done. Journal: 2026-07-23_experiment-...phase3.md
- **|r| histograms confirm the zero-shift (2026-07-23)**: on 5 sample layers
  the |r| distribution sits left of |i| — med|r|/med|i| = 0.813/0.640/0.659/
  0.841/0.924 at layers 0/7/16/24/31; shift largest exactly where the induced-
  sparsity gap peaks, gone by layer 31. When relevant: layer-selective
  application (exclude late layers) is the natural refinement if C3/C4 gains
  are diluted. Data: a6000-2 ~/workspace/oracle/llama2-7b/phase0/histograms.json
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
1. **RUNNING — C4 whitening round** (spec-c4-whitening.md, user work order
   2026-07-24): sigma calibration → whitened SVD factors (uniform r=512 +
   budget-allocated r_bar {256,512,1024}) → Task-0 trunc-vs-tail diagnostics →
   C4 sweeps. Success: C4-whitened beats C1 at s=0.7/0.9 and closes toward C3.
   Scope note: doc mentions 8B/p-grid/normalized-acc; standing scope
   (7B / top-K {0.5,0.7,0.9} / PPL) applied — code is model-agnostic for a
   later 8B pass.
2. (if whitening ≈ plain) spectrum is genuinely flat → discuss structured
   compensation alternatives with user before more GPU spend.
3. (later) generalize to LLaMA3-8B / other family once the deployable form
   is settled.

## Active Jobs
- `050-20260724-052033-oracle-llama2-c4-whitening` (a6000-2) — whitened SVD +
  rank allocation round. Journal: 2026-07-24_experiment-oracle-llama2-c4-whitening.md

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
