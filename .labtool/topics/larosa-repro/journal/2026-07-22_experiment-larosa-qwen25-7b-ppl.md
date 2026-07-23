# Experiment: larosa-qwen25-7b-ppl

**Status: DONE**

## Hypothesis tested
The upstream LaRoSa code reproduces paper Table 2 wikitext-2 PPL on Qwen2.5-7B:
dense 6.85 → 25% 6.90 / 40% 7.10 / 50% 7.42 (tolerance ±0.1); sparsity 0.0
should match dense. First run of the Qwen code path (Qwen2SparseForCausalLMRotate,
GQA dims differ from LLaMA).

## Prior art check
`larosa-llama2-7b-ppl` (same runner, same day): all four points reproduced
within ±0.06 — LLaMA code path validated. Qwen gen_act got the same pass-2
memory patch (081da28) but is untested until this run.

## Reproducibility
- **Git tag**: `exp/2026-07-22_larosa-qwen25-7b-ppl` (commit 17043d5)
- **Job ID**: `20260723-102000-larosa-qwen25-7b-ppl` (4th attempt; see Notes)
- **Assigned host/GPU**: a100-40-2, GPU 1 (PCI)
- **Command**: `bash scripts/repro_ppl.sh qwen ~/workspace/models/Qwen2.5-7B ~/workspace/models/qwen25_7b_larosa_Q` (cwd `~/workspace/repos/EfficientAI/larosa`)
- **Config path**: n/a — script args; sparsity is a runtime arg
- **Key parameters**: calibration wikitext-2 train 10×2048; sweep 0.0/0.25/0.4/0.5; ctx 2048; bf16; flash_attention_2; α h1=0.8p, h2=1.2p
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1
- **Model**: `~/workspace/models/Qwen2.5-7B` (our snapshot of Qwen/Qwen2.5-7B)

## Notes (submission history)
- Attempt 1 (`20260723-084738`) failed: co-tenant on GPU 0 with the llama3 job
  (dispatcher cross-cycle race); device_map=auto offloaded to CPU under the
  shared-memory pressure, ballooned to 36GB and OOMed even after the co-tenant
  died. Same dispatcher fix as llama3 attempt 1 (GRACE_SEC reservation).
- Attempt 2 (`20260723-085240`) failed alone on GPU 1: same pass-2 find_histogram
  OOM as llama3 (36.0GB; h3/h4 are 18944-dim so histogram sort is larger).
  Attempt 3 skips rotation (D.pt complete, 28 layers) and runs the sweep directly.
- Attempt 3 (`20260723-101220`) failed: dispatcher double-assigned this job and
  llama3 to GPU 0 in the same cycle — find_placement ran in a command-substitution
  subshell, so its TAKEN/RECENT bookkeeping never persisted. Fixed in QCom
  dispatcher (global-variable return instead of stdout). Attempt 4 on GPU 1.

### Results
Source: job log + meta of `20260723-102000-larosa-qwen25-7b-ppl`
(`~/workspace/runs/.../{log,meta}` on a100-40-2). STATUS=ok, exit 0,
runtime 11 min (10:20:22 -> 10:31:41 KST) on one A100 40GB (PCI GPU 1);
rotation stage skipped (D.pt already complete from attempt 2's pass 1).
Note: the qwen script prints "Larosa PPL" and uses eval_ppl_wikitext
(sparsity applied inside the model forward) — same measurement as llama.

| sparsity | PPL (this run) | paper Table 2 | delta |
|---|---|---|---|
| 0.0 (dense-equiv) | 6.8497 | 6.85 (Dense) | -0.000 |
| 0.25 | 6.9043 | 6.90 | +0.004 |
| 0.40 | 7.1112 | 7.10 | +0.011 |
| 0.50 | 7.4702 | 7.42 | +0.050 |

All four points within +-0.1 (max |delta| = 0.050). Success criterion met.

### Interpretation
