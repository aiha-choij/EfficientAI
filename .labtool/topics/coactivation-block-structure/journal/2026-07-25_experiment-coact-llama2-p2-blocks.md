# Experiment: coact-llama2-p2-blocks (P2 — clustering + structure evaluation)

Status: PENDING
Date: 2026-07-25

## Hypothesis tested
The strong null of the topic: balanced clustering of neurons on PPMI
co-activation similarity yields block structure that beats random balanced
partitions by ≥ ~1.3× on the structure metrics (within-block co-activation
mass, block-level union tiling, coverage @ budget). Below that threshold the
permutation axis is rejected and priority moves to shared-backbone +
residual (Idea A) / learned predictor (Idea D).

## What we're testing over alternatives
- Similarity: PPMI (positive PMI) — standard nonnegative substrate for
  spectral methods; Jaccard held back as fallback arm (gist Open Questions).
- Clustering: spectral embedding (normalized Laplacian of PPMI, top-64
  eigenvectors) + capacity-constrained (balanced) k-means. METIS skipped —
  not in venv, and spec lists it as one of several acceptable methods.
- Baseline: 3 random balanced partitions (seeds 1–3), exactly equal block
  sizes — isolates the clustering signal from the blocking itself.
- Static metrics from P1 file only; dynamic metrics (block-level union,
  per-token top-m block coverage) from ONE extra sparsified forward pass —
  P1 saved aggregated matrices, not per-token selections, so set-level
  group statistics need this pass (noted deviation from "P2 needs no model
  run"; it is a measurement pass, no new statistics collection).

## Prior art check
- P1 card (2026-07-24, same topic): union tax 6.0–6.3× @g=16, 9.1–9.4×
  @g=64 (92–95% of saturation) in layers 0–24; layer 31 exception. This is
  the inflation the blocks must absorb; at g=64 only block-level selection
  can save anything.
- Overlap card (larosa-intermediate-sparsity 2026-07-24): high-frequency
  pool consistent across s but ~68% per-token disagreement — the reason a
  nontrivial block structure is needed at all.
- Dead Ends: none in this topic.

## Expected outcome
Per (layer ∈ {0,8,16,24,31}) × (B ∈ {64,128,256}): clustered vs random
ratios for within-block mass on A/A^16/A^64, static coverage @ budget
(m = round(K_eff/B) blocks), dynamic per-token top-m block coverage, and
block-union tiling ratio (touched_blocks×B/|union|, g ∈ {16,64}; reported
as random/clustered so >1 favors clustering). Success: ratios ≥ 1.3× on
within-block mass / block-union at some (layer, B). Failure of the
hypothesis (≤1.3× everywhere) is a valid, decision-forcing result.
Sanity: all masses in [0,1]; block-union ≥ 1; clustered ratios ≥ ~1.0
(clustering should never lose to random on its own objective).

## Reproducibility
- **Git tag**: `exp/2026-07-25_coact-llama2-p2-blocks` (commit 26664fd)
- **Job ID**: `20260725-004032-coact-llama2-p2-blocks`
- **Assigned host/GPU**: a6000-4 (pinned via -H), GPU [pending dispatch]
- **Command**: `bash -c "/home/choij/workspace/venv-larosa/bin/python scripts/analyze_coactivation_blocks.py --stats /home/choij/workspace/analysis/llama2_coactivation_s09.pt --model_name /raid/LLM/llama2-7b --block_sizes 64,128,256 --seeds 3 --embed_dim 64 --group_sizes 16,64 --nsamples 32 --attn sdpa --out /home/choij/workspace/analysis/llama2_coactivation_blocks_s09.pt"`
  (cwd `/home/choij/workspace/repos/EfficientAI/larosa`; qsub `-H a6000-4 -g 1 -m 40`)
- **Config path**: n/a — parameters as script args
- **Key parameters**: input = P1 artifact (s=0.9, layers {0,8,16,24,31},
  A/A^16/A^64/f, 32×2048 wikitext-2 test tokens); PPMI → normalized
  Laplacian → torch.linalg.eigh top-64 eigvecs, row-normalized; balanced
  k-means 25 iters, capacity-greedy assignment, seed 0; random baseline
  seeds 1–3; dynamic pass = same 32×2048 tokens, sparsified forward,
  down_proj input hook; bf16, sdpa. Block partitions saved in output
  (reused by P3 as the permutation).
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3
  (venv `~/workspace/venv-larosa`)
- **Model**: `/raid/LLM/llama2-7b` (a6000-4 local copy); output
  `a6000-4:/home/choij/workspace/analysis/llama2_coactivation_blocks_s09.pt`
- **Sync**: local push 26664fd → gateway pull → scp gateway→a6000-4
  (md5 7f66376 verified)

### Results

### Interpretation
