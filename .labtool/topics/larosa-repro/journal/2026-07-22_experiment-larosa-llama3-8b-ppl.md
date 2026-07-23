# Experiment: larosa-llama3-8b-ppl

**Status: DONE**

## Hypothesis tested
The upstream LaRoSa code reproduces paper Table 2 wikitext-2 PPL on LLaMA3-8B:
dense 6.13 → 25% 6.23 / 40% 6.60 / 50% 7.22 (tolerance ±0.1); sparsity 0.0
should match dense.

## Prior art check
`larosa-llama2-7b-ppl` (same runner, same day): all four points reproduced
within ±0.06 — pipeline validated. Env fixes from that run apply (flash-attn
2.7.4.post1 pin, dispatcher PCI_BUS_ID, gen_act pass-2 memory patch).

## Reproducibility
- **Git tag**: `exp/2026-07-22_larosa-llama3-8b-ppl` (commit 17043d5)
- **Job ID**: `20260723-101220-larosa-llama3-8b-ppl` (3rd attempt; see Notes)
- **Assigned host/GPU**: a100-40-2, GPU 0 (PCI)
- **Command**: `bash scripts/repro_ppl.sh llama /raid/LLM/llama3-8b ~/workspace/models/llama3_8b_larosa_Q` (cwd `~/workspace/repos/EfficientAI/larosa`)
- **Config path**: n/a — script args; sparsity is a runtime arg
- **Key parameters**: calibration wikitext-2 train 10×2048 + alpaca 300; sweep 0.0/0.25/0.4/0.5; ctx 2048; bf16; flash_attention_2; α h1=0.8p, h2=1.2p
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1
- **Model**: `/raid/LLM/llama3-8b` (shared read-only Meta-Llama-3-8B, 4-shard safetensors)

## Notes (submission history)
- Attempt 1 (`20260723-084738`) failed: dispatcher raced across cycles and put
  this job and the qwen job on the same GPU 0 (probe saw the GPU free before the
  first process allocated memory) -> OOM. Fixed in QCom dispatcher: RECENT
  grace-period reservation (GRACE_SEC=180).
- Attempt 2 (`20260723-085136`) failed alone on GPU 0: gen_act pass 2 loads the
  rotated model on top of residual buffers and OOMs at find_histogram (37.9GB).
  Finding: the PPL eval model only needs pass-1 D.pt (one per layer, mlp reuses
  attn Q) — runner now tolerates pass-2 failure when D.pt set is complete.
  Attempt 3 skips rotation (D.pt already complete) and runs the sweep directly.

### Results
Source: job log + meta of `20260723-101220-larosa-llama3-8b-ppl`
(`~/workspace/runs/.../{log,meta}` on a100-40-2). STATUS=ok, exit 0,
runtime 14 min (10:12:21 -> 10:26:14 KST) on one A100 40GB (PCI GPU 0);
rotation stage skipped (D.pt already complete from attempt 2's pass 1).

| sparsity | PPL (this run) | paper Table 2 | delta |
|---|---|---|---|
| 0.0 (dense-equiv) | 6.1377 | 6.13 (Dense) | +0.008 |
| 0.25 | 6.2297 | 6.23 | -0.000 |
| 0.40 | 6.5736 | 6.60 | -0.026 |
| 0.50 | 7.1307 | 7.22 | -0.089 |

All four points within +-0.1 (max |delta| = 0.089 at 50%). Success criterion met.

### Interpretation
(User, 2026-07-22) Hypothesis confirmed — reproduction succeeded on all models.
Together with LLaMA2-7B this completes the 3-model baseline reproduction; the
public code and our PPL pipeline are trusted as the baseline for further work.
