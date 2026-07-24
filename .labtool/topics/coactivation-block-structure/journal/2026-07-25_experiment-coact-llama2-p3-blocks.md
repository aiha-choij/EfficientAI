# Experiment: coact-llama2-p3-blocks (P3 — block-mask oracle PPL, 2 jobs)

Status: PENDING
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
- **Job IDs**: prep `20260725-033520-coact-llama2-p3-prep`; eval
  [submitted after prep completes — same card, see Notes]
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

### Interpretation
