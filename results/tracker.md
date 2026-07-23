# Experiment tracker

| Date | Topic | Name | Model | Budget | Key metric | Δ vs baseline | Journal |
|---|---|---|---|---|---|---|---|
| 2026-07-22 | larosa-repro | larosa-llama2-7b-ppl | LLaMA2-7B | 1×A100-40G, 26 min | wikitext-2 PPL @40%: 5.6167 (paper 5.64) | +0.143 vs dense 5.4736 | topics/larosa-repro/journal/2026-07-22_experiment-larosa-llama2-7b-ppl.md |
| 2026-07-22 | larosa-repro | larosa-llama3-8b-ppl | LLaMA3-8B | 1×A100-40G, 14 min | wikitext-2 PPL @40%: 6.5736 (paper 6.60) | +0.436 vs dense 6.1377 | topics/larosa-repro/journal/2026-07-22_experiment-larosa-llama3-8b-ppl.md |
| 2026-07-22 | larosa-repro | larosa-qwen25-7b-ppl | Qwen2.5-7B | 1×A100-40G, 11 min | wikitext-2 PPL @40%: 7.1112 (paper 7.10) | +0.262 vs dense 6.8497 | topics/larosa-repro/journal/2026-07-22_experiment-larosa-qwen25-7b-ppl.md |
| 2026-07-22 | larosa-intermediate-sparsity | larosa-llama2-topk-int-ppl | LLaMA2-7B | 1×A100-40G, 7 min | wikitext-2 PPL @s=50%: 5.5210 (70%: 5.7296, 90%: 8.1083) | +0.047 vs dense 5.4736 | topics/larosa-intermediate-sparsity/journal/2026-07-22_experiment-larosa-llama2-topk-int-ppl.md |
| 2026-07-23 | oracle-residual-sparsity | oracle-llama2-phase0-calib | LLaMA2-7B | 1×A100-40G, ~2h run | induced-sparsity gap (r−i) @p=0.9: +3.0%p mean, 30/32 layers positive | n/a (distribution phase; go signal) | topics/oracle-residual-sparsity/journal/2026-07-22_experiment-oracle-llama2-phase0-calib.md |
| 2026-07-24 | oracle-residual-sparsity | oracle-llama2-phase3-c0c1 | LLaMA2-7B | 1×A6000, ~10 min | C1 top-K PPL 5.5216/5.7284/8.1096 @s=0.5/0.7/0.9 | anchors reproduced, max Δ 0.0013 | topics/oracle-residual-sparsity/journal/2026-07-23_experiment-oracle-llama2-phase3-c0c1.md |
| 2026-07-24 | oracle-residual-sparsity | oracle-llama2-phase4-c2c5 | LLaMA2-7B | 1×A6000, ~20 min | C3 PPL 6.6381 @s=0.9 (C2 8.2759, C4 8.7638, C5 6.8889) | −1.472 vs C1 8.1096 @s=0.9 (−56% ΔPPL) | topics/oracle-residual-sparsity/journal/2026-07-24_experiment-oracle-llama2-phase4-c2c5.md |
