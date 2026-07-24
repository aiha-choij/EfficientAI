# Experiment: coact-llama2-p1-stats (P1 — co-activation statistics)

Status: PENDING
Date: 2026-07-24 (submitted 2026-07-25 00:00 KST)

## Hypothesis tested
Raw-material collection, not a directional hypothesis: gather same-token
co-activation A, window co-activation A^g (g ∈ {16, 64}), and selection
frequency f on sample layers {0, 8, 16, 24, 31} at s = 0.9, so that P2 can
test whether PMI/Jaccard clustering yields block structure beating the
strong null (≥ ~1.3× over random-permutation blocks). Also measures the
P0-lite group tax: union inflation |∪_{t∈group} S_t| / K_eff for
contiguous groups g ∈ {16, 32, 64}.

## What we're testing over alternatives
- Actual sparsified forward (as in the overlap card), not dense activations.
- s = 0.9 only — the regime where grouping is most needed (spec §3 P1).
- Sample layers instead of all 32 (d×d fp32 is ~484 MB/matrix); layer 31
  mandatory as the known outlier.
- Windows {16, 64} only, matching the P3 sweep grid (g=32 omitted —
  decision recorded in gist Open Questions).
- Raw counts + normalization constants saved; PMI/Jaccard derived in P2.

## Prior art check
- `larosa-intermediate-sparsity/journal/2026-07-24_experiment-larosa-llama2-topk-overlap.md`:
  same hook/env, 3-min runtime, STATUS=ok. Its 3 failed dispatch attempts
  motivated the a6000-4 `-H` pin, `-m 40`, absolute paths — reused here.
- Overlap anchors at s=0.9 for sanity: C(1)=0.316, random 0.187,
  chance 0.100, K_eff=1106.
- Dead Ends: none relevant (new topic).

## Expected outcome
Success: for each of the 5 layers, A / A^16 / A^64 / f saved with
normalization constants; sanity bounds hold — K_eff ≈ 1101 (= ⌊0.1·11008⌋),
chance ≈ 0.100, mean off-diagonal co-activation lift ≥ 1, union
inflation ≥ 1 and ≤ min(g, d/K_eff). Failure: counts outside bounds
(script bug), OOM, or job error. This card only collects; the go/no-go
(strong null) is judged in P2.

## Reproducibility
- **Git tag**: `exp/2026-07-24_coact-llama2-p1-stats` (commit 67a1aff;
  script added same commit)
- **Job ID**: `20260725-000051-coact-llama2-p1-stats`
- **Assigned host/GPU**: a6000-4 (pinned via -H), GPU [pending dispatch]
- **Command**: `bash -c "mkdir -p /home/choij/workspace/analysis && /home/choij/workspace/venv-larosa/bin/python scripts/analyze_coactivation.py --model_name /raid/LLM/llama2-7b --sparsity 0.9 --layers 0,8,16,24,31 --windows 16,64 --group_sizes 16,32,64 --nsamples 32 --attn sdpa --out /home/choij/workspace/analysis/llama2_coactivation_s09.pt"`
  (cwd `/home/choij/workspace/repos/EfficientAI/larosa`; qsub `-H a6000-4 -g 1 -m 40`)
- **Config path**: n/a — parameters as script args
- **Key parameters**: sparse_mode=topk_intermediate, s=0.9; layers
  {0,8,16,24,31}; windows {16,64} (window |t−t′|<g incl. self-pair, within
  sequence); group sizes {16,32,64} for union inflation; 32 × 2048
  wikitext-2 test tokens; selection from down_proj input hook (≠ 0);
  fp32 GPU matmul accumulation (counts ≤ ~8.3M < 2^24, exact); bf16 model;
  attn=sdpa (no flash-attn on a6000-4; backend effect ~1e-3 per phase-3 gate)
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3
  (venv `~/workspace/venv-larosa`)
- **Model**: `/raid/LLM/llama2-7b` (a6000-4 local copy); output artifact
  `a6000-4:/home/choij/workspace/analysis/llama2_coactivation_s09.pt`
  (~7.3 GB — run P2 on a6000-4 CPU or gateway to avoid the transfer)
- **Sync**: local push 67a1aff → gateway pull → scp of the script
  gateway→a6000-4 (md5 verified; a6000-4 cannot fetch GitHub)

### Results
Source: meta + log of `20260725-000051-coact-llama2-p1-stats`
(`~/workspace/runs/.../{meta,log}` on gateway); output artifact
`a6000-4:/home/choij/workspace/analysis/llama2_coactivation_s09.pt` (6.8 GB,
verified on disk). STATUS=ok, runtime 1 min 28 s (00:01:06 → 00:02:34 KST,
2026-07-25), a6000-4 GPU 0. No draft.md (analysis job, prints its own summary).

Per-layer sanity (all within expected bounds):

| layer | K_eff | chance K_eff/d | union/K g=16 | g=32 | g=64 |
|---|---|---|---|---|---|
| 0  | 1105 | 0.100 | 6.14x | 7.87x | 9.13x |
| 8  | 1108 | 0.101 | 6.34x | 8.17x | 9.37x |
| 16 | 1107 | 0.101 | 6.01x | 7.88x | 9.22x |
| 24 | 1106 | 0.100 | 6.34x | 8.21x | 9.43x |
| 31 | 1105 | 0.100 | 3.92x | 5.19x | 6.62x |

- union/K = E[|∪_{t∈group} S_t|] / K_eff (union inflation, the P0 group
  tax); upper bound d/K_eff ≈ 9.96 — at g=64, layers 0–24 sit at 92–95%
  of saturation (a 64-token group touches ~10.1–10.4k of 11008 neurons).
- Mean off-diagonal same-token co-activation lift
  (E[A_offdiag]/E[f_j·f_j′]) = 0.999–1.000 — this aggregate is ≈1 by
  construction (Σ_jj′ A ≈ K² and Σ f_j f_j′ ≈ K̄²); pairwise structure
  lives in the distribution and is extracted in P2 via PMI/Jaccard.
- K_eff 1105–1108 vs nominal ⌊0.1·d⌋ = 1100 (+0.5%): selection is read as
  (down_proj input ≠ 0), so exact zeros of i outside Top-K masking are
  indistinguishable and ties inflate slightly — same convention as the
  overlap card (its K_eff was 1106).
- All five layers saved with A, A^16, A^64, freq counts + normalization
  constants (tokens = 65536, per-window pair counts).

### Interpretation
