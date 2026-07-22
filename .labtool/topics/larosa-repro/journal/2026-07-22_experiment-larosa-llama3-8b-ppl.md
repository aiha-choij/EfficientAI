# Experiment: larosa-llama3-8b-ppl

**Status: PENDING**

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
- **Job ID**: `20260723-084738-larosa-llama3-8b-ppl`
- **Assigned host/GPU**: a100-40-2, [pending dispatch]
- **Command**: `bash scripts/repro_ppl.sh llama /raid/LLM/llama3-8b ~/workspace/models/llama3_8b_larosa_Q` (cwd `~/workspace/repos/EfficientAI/larosa`)
- **Config path**: n/a — script args; sparsity is a runtime arg
- **Key parameters**: calibration wikitext-2 train 10×2048 + alpaca 300; sweep 0.0/0.25/0.4/0.5; ctx 2048; bf16; flash_attention_2; α h1=0.8p, h2=1.2p
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3, flash-attn 2.7.4.post1
- **Model**: `/raid/LLM/llama3-8b` (shared read-only Meta-Llama-3-8B, 4-shard safetensors)

### Results

### Interpretation
