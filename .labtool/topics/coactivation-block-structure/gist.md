# coactivation-block-structure — Can neuron permutation make group masks block-structured?

## Status
active — started 2026-07-24 from spec
`~/Workspace/research-wiki/plans/coactivation-block-structure-spec.md`
(derived from the 2026-07-24 group-sparsity discussion, Idea C; parent doc:
research-wiki `plans/predictor-group-sparsity-research-direction.md`).

## Notation (shared with larosa-intermediate-sparsity)
```
FFN(x) = W_d i,  i = u ⊙ g,  u = W_u x,  g = σ(W_g x)
d = intermediate size (LLaMA2-7B: 11008),  L = 32 layers
S_t ⊂ {1..d} : Top-K surviving-neuron set of token t, K = ⌊(1−s)·d⌋
C(δ) = E_t[|S_t ∩ S_{t+δ}|] / K_eff   (containment overlap)
chance = K_eff / d
A_jj′  = E_t[1[j ∈ S_t]·1[j′ ∈ S_t]]          (same-token co-activation)
A^g_jj′ = E_{|t−t′|<g}[1[j ∈ S_t]·1[j′ ∈ S_t′]] (window-g co-activation)
PMI_jj′ = log(A_jj′ / (f_j·f_j′)),  f_j = selection frequency
```

## Hypothesis
Sharing intermediate Top-K masks across token groups (g = 16/32/64) pays a
heavy tax because neuron selection is dominantly token-dependent (prior
measurement: adjacent overlap only 3.15× chance at s=0.9, ~68% disagreement).
Permutation is a free, exactly function-preserving degree of freedom for
i = u ⊙ g (only monomial transforms commute with the elementwise structure —
parent doc §3.2). Hypothesis: reordering neurons into blocks by co-activation
statistics (PMI / Jaccard clustering) makes the group mask a block-selection
problem, absorbing union inflation at the block level.

Strong null to beat: if clustered blocks give < ~1.3× gain over random
permutation blocks on the P2 metrics, the permutation axis is rejected and
priority moves to shared-backbone + residual (Idea A) / learned predictor
(Idea D).

## Plan (from spec §3)
- **P1** — collect co-activation statistics A, A^g, f on sample layers
  {0, 8, 16, 24, 31} at s=0.9 (layer 31 mandatory: the exceptional layer),
  32 × 2048 tokens, wikitext-2, actual sparsified forward.
- **P2** — balanced clustering (spectral / METIS / balanced k-means; cf.
  MoEfication) on PMI, block sizes B ∈ {64, 128, 256}; evaluate
  within-block co-activation mass, block-level union ratio, block coverage
  @ budget — all reported as multiples over random permutation.
- **P3** (only if P2 promising) — block-mask oracle PPL with gauge-fixed
  score Σ‖W_d[:,j]‖·|i_j|, vs dense / per-token top-K / random-permutation
  blocks. Sweep s ∈ {0.7, 0.9} × g ∈ {16, 64} × B ∈ {64, 256} (selected).
- **P0** (optional, alongside P1) — group-tax curve: union inflation
  |∪_{t∈group} S_t|/K and group-shared top-K PPL.

## Key Findings
- **Group-union tax quantified at s=0.9 (2026-07-25)**: naive union sharing
  of per-token Top-K masks over contiguous token groups inflates the neuron
  budget (union tax = E[|∪_{t∈group} S_t|]/K_eff; floor 1×, saturation
  d/K_eff ≈ 9.96×) by 6.0–6.3× at g=16, 7.9–8.2× at g=32, 9.1–9.4× at g=64
  in layers 0/8/16/24 — the g=64 figure is 92–95% of saturation (a 64-token
  group touches ~10.1–10.4k of 11008 neurons), so naive union masks are
  effectively dense there. Layer 31 is again the exception (3.92/5.19/6.62×).
  Confirms and sharpens the spec §2 prediction; the permutation axis now
  rests entirely on P2 blocks absorbing this inflation. P1 outputs
  (A, A^16, A^64, f per layer {0,8,16,24,31}) saved at
  a6000-4:~/workspace/analysis/llama2_coactivation_s09.pt (6.8 GB).
  When relevant: any group-shared mask budget design (incl. RB-Sparse) —
  at g ≥ 64 naive union sharing is worthless; block-structured selection is
  mandatory for the group axis.
  Journal: journal/2026-07-24_experiment-coact-llama2-p1-stats.md
- **PPMI clustering beats random blocks, but coverage stays low (2026-07-25)**:
  spectral + balanced k-means on PPMI passes the pre-registered ≥1.3× strong
  null on within-block mass at 13/15 (layer, B) settings (L0 1.9–2.2×,
  L8–24 1.25–1.54×, L31 3.8–4.1×) and on dynamic per-token top-m block
  coverage at 15/15 (1.5–4.2×). But (i) the block-union tiling metric
  saturates at d/|union| — group unions touch all blocks under ANY balanced
  partition, killing naive "activate touched blocks" sharing; and (ii)
  absolute top-m coverage at the K budget is only 0.20–0.36 in mid layers
  (0.52–0.57 at L31) vs random 0.12–0.17 — block masks alone still lose
  65–80% of per-token signal at s=0.9.
  When relevant: designing any block-granular mask (P3, predictor targets,
  RB-Sparse) — budget must go through top-m block selection, expect large
  PPL damage without residual compensation, and treat L31 separately.
  Journal: journal/2026-07-25_experiment-coact-llama2-p2-blocks.md
- **P3 block-mask oracle PPL: bare shared mask is catastrophic regardless of
  clustering (2026-07-25, CONFIRMED)**: both P3 jobs completed (STATUS=ok).
  Clustered blocks beat random 2–3× in 3/4 (B,g) settings (structure does
  transfer to function — informative failure mode (i) rejected), but
  **absolute PPL is 4.5k–24k vs the 8.11 per-token anchor** in every arm —
  the pre-registered success gate (ΔPPL ≤ +1.0) is missed by 3–4 orders of
  magnitude. Root cause: P2's 0.20–0.36 per-layer top-m coverage compounds
  multiplicatively across all 32 masked layers. B=64,g=64 clustered≈random
  (near-tie) because g=64 groups already sit at 92–95% of union saturation
  (P1) — no partition can discriminate blocks there.
  When relevant: **block mask alone (no compensation) is a dead end as a
  deployable mechanism** — but the clustered partitions themselves
  (`llama2_p3_partitions_s09.pt`) remain a reusable asset. This is the
  empirical baseline that any input-dependent block compensation scheme
  must beat; see topic `block-sparse-compensation` (C7a/C7/C8a/C8), whose
  Phase 4 (P3′) reruns this exact setup with compensation added.
  Journal: journal/2026-07-25_experiment-coact-llama2-p3-blocks.md

## Dead Ends
- **Bare shared block mask (no compensation) as a standalone mechanism
  (2026-07-25)**: P3 confirmed PPL 4.5k–24k at s=0.9 regardless of
  clustering quality — unusable without a recovery term. Clustering itself
  is NOT a dead end (2–3× real gain over random); only "mask alone, no
  compensation" is closed. Superseded by `block-sparse-compensation`.

## Open Questions
- Which normalization (PMI vs Jaccard) is the better clustering substrate?
  P2 first pass uses PPMI (positive PMI — standard nonneg similarity for
  spectral methods); Jaccard kept as the fallback arm if PPMI blocks fail
  the strong null narrowly.
- ~~Window statistic A^g: which g values to accumulate?~~ Decided
  2026-07-24: {16, 64}, matching the P3 sweep grid; g=32 omitted.
- P3 success criterion is provisional (e.g. ΔPPL ≤ +1.0 vs per-token at
  s=0.9) — finalize after first numbers, per house convention.

## Next Experiments
1. ~~P3 — block-mask oracle PPL~~ DONE (both jobs STATUS=ok, 2026-07-25) —
   see Key Findings + Dead Ends above.
2. **Blocks + compensation (Idea A hybrid), contingent on P3's outcome,
   now underway as topic `block-sparse-compensation`** — P3's "clustered
   beats random but both catastrophic" result triggered exactly the
   "add compensation" branch of this line, not the "pivot to Idea D"
   branch. That topic's Phase 4 (P3′) reuses this topic's P2 partitions
   (`a6000-4:~/workspace/analysis/llama2_p3_partitions_s09.pt`, LLaMA2-7B)
   layered under the C7/C8 input-dependent block compensation. Track
   further progress there, not in this topic.

## Active Jobs
(none — P3 jobs both completed; this topic's active experimental thread has
moved to `block-sparse-compensation` topic Phase 4/P3′.)

## Boundaries / coordination
- RB-Sparse (Dowon Kim) owns block-shared masks *on top of rotation*; this
  topic is the rotation-free original-basis permutation axis — no overlap.
  The §2 tax measurements are relevant to both threads (worth sharing).
- Weight-aware score ‖W_d[:,j]‖·|i_j| is shared material with
  oracle-residual-sparsity H2 (P3 uses the same definition).
- The parent doc's §3.2 *theory framing* is on hold with the advisor; this
  topic proceeds as an experimental axis, independent of that framing.

## Pointers
- **Synthesis report (P1+P2+P3, Korean, 4 figures)**:
  `results/reports/2026-07-28_coactivation-block-structure.md`
  (figures + generator: `results/reports/figs/`)
- Spec: `~/Workspace/research-wiki/plans/coactivation-block-structure-spec.md`
  (mac-local; self-contained).
- Prior measurements: topic `larosa-intermediate-sparsity` gist Key Findings
  ("Cross-token selection overlap…", 2026-07-24) + journal
  `2026-07-24_experiment-larosa-llama2-topk-overlap.md`; report artifact
  https://claude.ai/code/artifact/c73a7f23-2ac7-4b27-9857-6c21a60d184f
- PPL anchors (trusted): dense 5.4736; per-token topk_intermediate
  s=0.5/0.7/0.9 → 5.5210/5.7296/8.1083 (a6000-2 sdpa remeasure
  5.5216/5.7284/8.1096; backend delta ~1e-3).
- Code: `larosa/inference/modeling_llama_larosa.py`
  (`sparse_mode='topk_intermediate'`, validated s=0 ≡ dense bitwise);
  hook pattern: `larosa/scripts/analyze_topk_overlap.py`.
- Execution hosts (2026-07-24): a6000-4 (venv ~/workspace/venv-larosa,
  sdpa only, model /raid/LLM/llama2-7b, repo synced via gateway scp/tar —
  no GitHub fetch); a100-40-2 gateway (conda `larosa`, flash-attn, A100s
  often busy).
- Known traps: qsub workdir must be an absolute path (literal `~` fails
  silently); dispatcher only places on fully-idle GPUs (util ≤ 10%);
  `runs` ELAPSED includes queue wait; eval_ppl.py mlp/attn sparsity labels
  are swapped (upstream bug) — prefer direct hooks.
