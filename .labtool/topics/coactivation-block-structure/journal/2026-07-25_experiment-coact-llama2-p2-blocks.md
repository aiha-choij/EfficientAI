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
Source: meta + log of `20260725-004032-coact-llama2-p2-blocks` (gateway
`~/workspace/runs/.../{meta,log}`); raw partitions + metrics in
`a6000-4:/home/choij/workspace/analysis/llama2_coactivation_blocks_s09.pt`.
STATUS=ok, runtime 3 min 23 s (00:40:45 → 00:44:08 KST), a6000-4 GPU 0.

Clustered vs random-mean ratios (random = 3 balanced seeds; B row-stable,
showing B=64 / 128 / 256):

| layer | mass_A ratio | dyn_cov clustered (abs) | dyn_cov ratio | blkU16 / blkU64 ratio |
|---|---|---|---|---|
| 0  | 2.18 / 2.07 / 1.91× | 0.359 / 0.351 / 0.303 | 2.15–2.44× | 1.00× (saturated) |
| 8  | 1.36 / 1.32 / 1.25× | 0.252 / 0.244 / 0.206 | 1.51–1.66× | 1.00× (saturated) |
| 16 | 1.44 / 1.38 / 1.31× | 0.269 / 0.260 / 0.224 | 1.61–1.81× | 1.00× (saturated) |
| 24 | 1.54 / 1.47 / 1.41× | 0.283 / 0.274 / 0.240 | 1.69–1.94× | 1.00× (saturated) |
| 31 | 4.12 / 3.91 / 3.77× | 0.569 / 0.571 / 0.518 | 3.39–4.17× | 1.00× (saturated) |

- mass_A = within-block off-diagonal co-activation mass fraction (clustered
  absolute values 0.008–0.087; random 0.006–0.023). Static coverage @ budget
  shows the same ordering (e.g. L31: 0.53 vs 0.14 random).
- dyn_cov = per-token coverage of the top-m blocks (m·B ≈ K budget),
  measured on the real sparsified forward. Random baseline ≈ 0.12–0.17.
- blkU (block-union tiling, touched_blocks×B/|union|) is IDENTICAL for
  clustered and random at every layer and every B: the metric is saturated
  at its ceiling d/|union| (g=16: 1.57–1.66 ≈ 11008/(6.0–6.3×K_eff);
  g=64: 1.06–1.09; L31: 2.58/1.52). Group unions are so spread out that
  they touch essentially every block under ANY balanced partition at these
  B — the metric is uninformative here, not a tie in structure quality.
- Sanity: all masses in [0,1], blkU ≥ 1, clustered ≥ random everywhere. ✓

Pre-registered criterion check (≥1.3× vs random): within-block mass passes
at 13/15 (layer, B) settings (fails only L8 B=256 at 1.25×; L8 B=128 at
1.32× is borderline); dyn_cov passes everywhere (1.51–4.17×); block-union
is saturated/uninformative rather than passed or failed.

### Interpretation
