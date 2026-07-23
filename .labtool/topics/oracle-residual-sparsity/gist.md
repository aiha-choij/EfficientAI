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
(none yet)

## Dead Ends
(none yet)

## Open Questions (left as questions per spec's own instruction)
- **Rank grid for C4**: spec fixes only r ≤ h/8 (H3). Proposed sweep
  r ∈ {h/32, h/16, h/8} (LLaMA-3.1-8B h=4096 → 128/256/512) — confirm.
- **8B model version**: spec says meta-llama/Llama-3.1-8B; gateway has
  /raid/LLM/llama3-8b (Llama-3-8B, dense PPL 6.1377 trusted). Download 3.1-8B
  (needs HF gated access) or accept 3-8B? Affects literature comparability only.
- **Model availability**: Llama-3.2-3B and gemma-3-4b-pt are not on the
  gateway yet; both are gated on HF (license acceptance + token). Need
  HF_TOKEN on the gateway before Phase 2.
- **lm-eval-harness**: not currently in the EfficientAI pipeline (larosa repro
  was PPL-only); needs install into conda env + a harness wrapper that keeps
  the OracleSparseMLP hooks active during harness forwards.
- **C3/C4/C5 dense-anchor**: normalized accuracy denominator is C0 measured in
  the same run/limit (spec §7.6) — one dense eval per model per limit setting.

## Next Experiments
1. **Phase 1 — infra + OracleSparseMLP + 4 unit tests** (spec §2): wrapper with
   buffers (g_bar, g_bar_star, col_norm, A, B), conditions {dense,c1..c5},
   stats_mode, exclude_layers, per-layer achieved-sparsity logging; scripts
   01–05 skeletons. Tests: p=1 identity (C3≡dense, atol 1e-3 bf16), C4 full-rank
   ≡ C3 (rtol 1e-3), rank diagnostic norm per layer, mask-vs-slice equivalence.
   CPU tiny-model first (same discipline as topk_intermediate 40edf40).
   Gate: all 4 tests pass.
2. **Phase 2 — calibration + Phase-0 distribution report (3B)**: c4-en 512
   seq × 2048 tok (fp32 accumulators), second ḡ from wikitext-103; induced
   sparsity curves i vs r, Hoyer/kurtosis, gate CV², ḡ corpus Pearson.
   Gate: report generated; **r curve above i curve = go signal for H1**.
3. **Phase 3 — C1 sweep 3B → 8B**: p grid ×11, lm-eval --limit 1000 + wikitext
   PPL. Gate: 8B C1 critical sparsity in 50–70% (literature sanity).
4. **Phase 4 — C2–C5 sweep, 8B main, repeat on 3B/Gemma**: main result table +
   full (no-limit) eval at 2–3 p values near each critical point.
5. **Phase 5 (optional) — C6 group masks, Instruct models.**

## Active Jobs
- (none)

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
