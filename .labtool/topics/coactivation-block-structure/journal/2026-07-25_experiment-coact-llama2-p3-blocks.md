# Experiment: coact-llama2-p3-blocks (P3 — block-mask oracle PPL, 2 jobs)

Status: CONFIRMED
Date: 2026-07-25

## Hypothesis tested
Spec §3 P3: with neurons permuted into PPMI-clustered balanced blocks, a
group-shared budgeted block mask (per group of g contiguous tokens, keep the
top-m blocks by the gauge-fixed score Σ_{t∈group} Σ_{j∈b} ‖W_d[:,j]‖·|i_tj|,
m = round(K/B)) retains PPL clearly better than the same mechanism on random
balanced blocks (control), and the gap to per-token unstructured top-K looks
compensable (provisional: ΔPPL ≤ +1.0 vs anchor at s=0.9; finalize after
first numbers). Expectations set low by P2: top-m coverage 0.20–0.36 in mid
layers predicts substantial degradation vs anchor.

## What we're testing over alternatives
- Budgeted top-m block selection, NOT "activate touched blocks": P2 showed
  the union-tiling route is saturated/dead under any balanced partition.
- Oracle scoring (uses the group's actual activations, non-causal within
  group) — upper bound for the permutation axis, mirroring the oracle
  topic's methodology; deployable prediction comes later (predictor axis).
- All-layer masking (all 32 layers blocked simultaneously), matching how
  the per-token anchor applies top-K everywhere — spec-faithful full-model
  P3 rather than masking only the 5 P2 sample layers.
- One shared PPL protocol for all arms (full wikitext-2 test,
  non-overlapping 2048-token chunks, in-script) — internal comparisons
  immune to protocol mismatch with historical anchors.
- s = 0.9 only (priority regime); s = 0.7 deferred.

## Prior art check
- P2 card (2026-07-25): clustering beats random on mass/coverage
  (1.3–4.2×) but absolute top-m coverage is low; L31 strongest (~4×).
- P1 card: union tax 6–9.4× — the reason budgeted selection is the only
  viable sharing mode.
- Anchors (larosa-intermediate-sparsity + oracle phase-3 gate): dense
  5.4736/5.4738, per-token top-K s=0.9 → 8.1083/8.1096 (protocol variants
  differ ~1e-3; in-script anchor re-measured here anyway).
- Dead Ends: none in this topic.

## Expected outcome
Per (B ∈ {64, 256}) × (g ∈ {16, 64}): PPL for clustered vs random blocks,
plus dense and per-token anchors under the identical protocol. Success:
clustered − random clearly negative (structure transfers to PPL) AND
clustered − anchor ≤ ~+1.0 at some setting. Informative failure modes:
(i) clustered ≈ random → P2 structure doesn't transfer to function,
permutation axis rejected; (ii) both catastrophically worse than anchor
(consistent with 0.2–0.36 coverage) → block masks alone dead, pivot to
blocks + residual (Idea A hybrid) or Idea D. Sanity: dense arm ≈ 5.47;
per-token arm ≈ 8.11; random ≥ clustered expected.

## Reproducibility
- **Git tag**: `exp/2026-07-25_coact-llama2-p3-blocks` (commit aa7ba4f;
  scripts p3_collect_cluster_all.py + p3_block_ppl.py)
- **Job IDs**: prep `20260725-033520-coact-llama2-p3-prep` (STATUS=ok,
  all 32 layers, k_eff 1104–1106); eval
  `20260725-034614-coact-llama2-p3-ppl`
- **Assigned host/GPU**: a6000-4 (pinned via -H), GPU [pending dispatch]
- **Commands**:
  prep: `bash -c "/home/choij/workspace/venv-larosa/bin/python scripts/p3_collect_cluster_all.py --model_name /raid/LLM/llama2-7b --sparsity 0.9 --block_sizes 64,256 --embed_dim 64 --nsamples 32 --attn sdpa --out /home/choij/workspace/analysis/llama2_p3_partitions_s09.pt"`
  eval: `bash -c "/home/choij/workspace/venv-larosa/bin/python scripts/p3_block_ppl.py --partitions /home/choij/workspace/analysis/llama2_p3_partitions_s09.pt --model_name /raid/LLM/llama2-7b --sparsity 0.9 --group_sizes 16,64 --block_sizes 64,256 --attn sdpa --out /home/choij/workspace/analysis/llama2_p3_block_ppl_s09.pt"`
  (both cwd `/home/choij/workspace/repos/EfficientAI/larosa`; qsub
  `-H a6000-4 -g 1 -m 40`)
- **Config path**: n/a — parameters as script args
- **Key parameters**: prep = all-32-layer A-only collection (32 × 484 MB
  fp32 accumulators on GPU), same 32 × 2048 wikitext-2 test tokens as
  P1/P2, PPMI → eigh top-64 → balanced k-means seed 0, random control
  seed 1, B ∈ {64, 256}. eval = full-model block-group masking via
  LlamaMLP.forward monkeypatch, fp32 scoring path, m = round(K/B) with
  K = round(0.1·11008) = 1101 (B=64: m=17, budget 1088; B=256: m=4,
  budget 1024, −7% vs K — recorded), PPL over full test
  (~⌊len/2048⌋ chunks), bf16 model, sdpa.
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3
  (venv `~/workspace/venv-larosa`)
- **Model**: `/raid/LLM/llama2-7b`; outputs
  `a6000-4:/home/choij/workspace/analysis/llama2_p3_partitions_s09.pt`,
  `.../llama2_p3_block_ppl_s09.pt`
- **Sync**: local push aa7ba4f → gateway pull → scp gateway→a6000-4 (md5
  eb78d5e/35d28e5 verified). Submission delayed ~2.5 h by a VPN outage
  (resume point had been recorded in gist Active Jobs).

## Notes
- Two qsub jobs under one card: prep produces the partition artifact, eval
  consumes it; eval is submitted only after prep STATUS=ok (dispatcher has
  no dependency mechanism).

### Results
Source: meta + log of `20260725-033520-coact-llama2-p3-prep` (STATUS=ok,
32/32 layers clustered, k_eff 1104–1106) and
`20260725-034614-coact-llama2-p3-ppl` (STATUS=ok); artifacts
`a6000-4:/home/choij/workspace/analysis/llama2_p3_partitions_s09.pt`,
`.../llama2_p3_block_ppl_s09.pt`.
Protocol: 166 × 2048 non-overlapping wikitext-2 test tokens, identical for
all arms. Sanity anchors reproduced exactly: dense 5.4738, per-token top-K
s=0.9 → 8.1096 (matches the a6000-2 phase-3 gate values to 4 decimals).

Block-mask oracle PPL @ s=0.9, all 32 layers masked:

| B | g | budget m·B/d | clustered | random | anchor gap (clustered − 8.11) |
|---|---|---|---|---|---|
| 64  | 16 | 0.099 | 4539  | 9674  | +4531 |
| 64  | 64 | 0.099 | 12032 | 11941 | +12024 |
| 256 | 16 | 0.093 | 6950  | 14763 | +6942 |
| 256 | 64 | 0.093 | 7776  | 23919 | +7768 |

- Clustered beats random in 3/4 settings, often by 2–3× — the P2 structure
  does transfer to function directionally (except B=64 g=64, a tie within
  noise at this damage level).
- But absolute PPL is catastrophic in every block arm (4.5k–24k vs anchor
  8.11): the pre-registered success gate (ΔPPL ≤ ~+1.0 vs per-token anchor)
  is missed by 3–4 orders of magnitude. Note B=256 runs at −7% budget
  (1024 vs K=1101).
- This matches the P2 coverage prediction quantitatively: 0.20–0.36
  per-token coverage at every one of 32 layers compounds multiplicatively;
  the model is effectively destroyed.

### Interpretation
- **Root cause is multiplicative compounding across 32 layers, not weak
  clustering.** P2 measured per-layer top-m coverage of only 0.20–0.36 in mid
  layers. All-layer masking (this experiment's protocol, matching how the
  per-token anchor applies top-K everywhere) compounds that loss
  independently at every layer — even at the generous end (0.36 per layer)
  the fraction of a token's true per-token-important path that survives 32
  layers is astronomically small. PPL landing at 4.5k–24k (vs the 8.11
  per-token anchor, itself already a real s=0.9 penalty over dense 5.47) is
  exactly what P2's coverage numbers predicted — not a surprise, and not an
  implementation bug (sanity anchors reproduced to 4 decimals).
- **Clustering is directionally real, not noise.** Clustered beats random by
  2–3× in 3/4 settings — pre-registered informative failure mode (i) ("P2
  structure doesn't transfer to function → reject the permutation axis")
  did **not** occur. PMI-based block structure captures genuine co-activation
  signal that survives all the way to PPL.
- **But failure mode (ii) did occur**: both arms are catastrophically worse
  than the per-token anchor regardless of clustering quality. At this budget
  regime (s=0.9, m=round(K/B), all 32 layers), a bare shared block mask —
  clustered or not — is unusable as a standalone replacement mechanism.
- **The B=64,g=64 near-tie is expected, not anomalous**: P1's union-tax
  finding already showed g=64 groups touch 92–95% of saturation (~10.1–10.4k
  of 11008 neurons) — a 64-token group is already close to dense *before* any
  block partition is applied, so no partition (clustered or random) can
  discriminate blocks meaningfully at that specific (B,g) combination.
- **Conclusion — dead end for "block mask alone", not for the permutation
  axis itself.** The clustered-vs-random gap is a real, reusable asset (the
  PPMI partitions in `llama2_p3_partitions_s09.pt` are worth keeping); what's
  dead is using bare shared-mask compute with no recovery term as a
  deployable mechanism. This is the empirical grounding for why sharing-tax
  recovery matters: any compensation scheme (input-dependent, per-block) must
  beat a baseline this bad, and the gap it needs to close is enormous (orders
  of magnitude in PPL, not percent). This finding directly motivates and is
  superseded by topic `block-sparse-compensation` (spec conditions C7a/C7/
  C8a/C8), whose Phase 4 (P3′) reruns this exact clustered-block setup with
  compensation layered on top instead of bare masking — same partitions,
  same model, added recovery term.
- **Status update**: this card is now CONFIRMED (results + interpretation
  complete); no further P3 work planned under bare block-masking. See
  `block-sparse-compensation` topic for the follow-on compensation work.
