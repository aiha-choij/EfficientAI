# Experiment: larosa-llama2-7b-ppl

**Status: DONE**

## Hypothesis tested
The upstream LaRoSa code reproduces paper Table 2 wikitext-2 PPL on LLaMA2-7B:
dense 5.47 → 25% 5.51 / 40% 5.64 / 50% 5.87 (tolerance ±0.1). Sparsity 0.0 doubles
as a rotation-losslessness sanity check (should match dense 5.47).

## Prior art check
None found — journal has only the init card. Environment prepared in the init
session (conda env `larosa`, models on gateway).

## Reproducibility
- **Git tag**: `exp/2026-07-22_larosa-llama2-7b-ppl-2` (repo `aiha-choij/EfficientAI`, commit 081da28)
- **Job ID**: `20260723-070108-larosa-llama2-7b-ppl` (4th attempt; see Notes)
- **Assigned host/GPU**: a100-40-2, GPU 4 (PCI index; real A100 40GB)
- **Command**: `bash scripts/repro_ppl.sh llama /raid/LLM/llama2-7b ~/workspace/models/llama2_7b_larosa_Q` (cwd `~/workspace/repos/EfficientAI/larosa`)
- **Config path**: n/a — parameters passed as script args; sparsity set at runtime via `--sparsity`
- **Key parameters**: calibration wikitext-2 train 10×2048 (paper uses 16×2048 — recorded deviation) + alpaca 300; PPL sweep 0.0/0.25/0.4/0.5; ctx 2048; bf16; flash_attention_2; α baked in code as h1=0.8p, h2=1.2p
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1, lm-eval 0.4.3
- **Model**: `/raid/LLM/llama2-7b` (shared read-only copy of meta-llama/Llama-2-7b-hf, fp16 safetensors)

## Notes (submission history)
- Attempt 1 (`20260723-064304`) failed: flash-attn 2.8.3.post1 wheel required GLIBC_2.32; gateway is glibc 2.31. Fixed by installing the official `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310` wheel.
- Attempt 2 (`20260723-064655`) failed: dispatcher exported nvidia-smi (PCI) GPU index but CUDA's default FASTEST_FIRST enumeration mapped it to the 4GB DGX Display → CPU offload → OOM. Fixed in QCom dispatcher (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, QCom commit 4a3a82c).
- Attempt 3 (`20260723-064935`, tag `...-ppl` @ 70aa3fc) failed: upstream gen_act
  keeps pass-1 model remnants + fp64 H_attn/H_mlp (~8.6GB) alive while loading the
  pass-2 model → OOM on 40GB during find_histogram. Patched in 081da28 (free before
  pass-2 load; llama + qwen).
- Gateway agent watches the job hourly by name (request `20260723-070147-larosa-llama2-7b-ppl-watch3`).

### Results
Source: job log + meta of `20260723-070108-larosa-llama2-7b-ppl`
(`~/workspace/runs/.../{log,meta}` on a100-40-2). STATUS=ok, exit 0,
runtime 26 min (07:01:29 → 07:27:28 KST) on one A100 40GB (PCI GPU 0).

| sparsity | PPL (this run) | paper Table 2 | Δ |
|---|---|---|---|
| 0.0 (dense-equiv) | 5.4736 | 5.47 (Dense) | +0.004 |
| 0.25 | 5.5017 | 5.51 | −0.008 |
| 0.40 | 5.6167 | 5.64 | −0.023 |
| 0.50 | 5.8167 | 5.87 | −0.053 |

All four points within ±0.1 tolerance (max |Δ| = 0.053). Slight systematic
advantage over paper numbers at higher sparsity. Success criterion met.

### Interpretation

