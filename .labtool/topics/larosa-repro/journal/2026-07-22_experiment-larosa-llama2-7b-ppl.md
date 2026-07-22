# Experiment: larosa-llama2-7b-ppl

**Status: PENDING**

## Hypothesis tested
The upstream LaRoSa code reproduces paper Table 2 wikitext-2 PPL on LLaMA2-7B:
dense 5.47 → 25% 5.51 / 40% 5.64 / 50% 5.87 (tolerance ±0.1). Sparsity 0.0 doubles
as a rotation-losslessness sanity check (should match dense 5.47).

## Prior art check
None found — journal has only the init card. Environment prepared in the init
session (conda env `larosa`, models on gateway).

## Reproducibility
- **Git tag**: `exp/2026-07-22_larosa-llama2-7b-ppl` (repo `aiha-choij/EfficientAI`, commit 70aa3fc)
- **Job ID**: `20260723-064935-larosa-llama2-7b-ppl` (3rd attempt; see Notes)
- **Assigned host/GPU**: a100-40-2, GPU 4 (PCI index; real A100 40GB)
- **Command**: `bash scripts/repro_ppl.sh llama /raid/LLM/llama2-7b ~/workspace/models/llama2_7b_larosa_Q` (cwd `~/workspace/repos/EfficientAI/larosa`)
- **Config path**: n/a — parameters passed as script args; sparsity set at runtime via `--sparsity`
- **Key parameters**: calibration wikitext-2 train 10×2048 (paper uses 16×2048 — recorded deviation) + alpaca 300; PPL sweep 0.0/0.25/0.4/0.5; ctx 2048; bf16; flash_attention_2; α baked in code as h1=0.8p, h2=1.2p
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1, lm-eval 0.4.3
- **Model**: `/raid/LLM/llama2-7b` (shared read-only copy of meta-llama/Llama-2-7b-hf, fp16 safetensors)

## Notes (submission history)
- Attempt 1 (`20260723-064304`) failed: flash-attn 2.8.3.post1 wheel required GLIBC_2.32; gateway is glibc 2.31. Fixed by installing the official `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310` wheel.
- Attempt 2 (`20260723-064655`) failed: dispatcher exported nvidia-smi (PCI) GPU index but CUDA's default FASTEST_FIRST enumeration mapped it to the 4GB DGX Display → CPU offload → OOM. Fixed in QCom dispatcher (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, QCom commit 4a3a82c).
- Gateway agent watches the job hourly (request `20260723-065148-larosa-llama2-7b-ppl-watch2`).

### Results

### Interpretation
