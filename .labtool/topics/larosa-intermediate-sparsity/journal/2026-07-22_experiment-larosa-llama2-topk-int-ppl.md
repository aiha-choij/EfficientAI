# Experiment: larosa-llama2-topk-int-ppl

Status: PENDING
Date: 2026-07-22

## Hypothesis tested
Per-token magnitude Top-K on the FFN intermediate activation i = u ⊙ g
(original basis, no rotation, FFN-only, attention dense) retains wikitext-2
PPL well at s = 50/70/90% on LLaMA2-7B. First validation point of the
larosa-intermediate-sparsity topic.

## What we're testing over alternatives
Implementation-correctness gates + first s sweep on one model before the
3-model extension. No input-mode comparison arm (pivot Q&A 2026-07-22:
dense-ΔPPL-only baseline; input-vs-intermediate ranking accepted from
arXiv:2509.00454).

## Prior art check
- larosa-repro LLaMA2-7B card (2026-07-22): same eval pipeline reproduced
  dense 5.4736 and paper Table 2 within ±0.1; 26 min on one A100 40GB.
- Dead end "gen_act pass-2 OOM" does not apply — this mode has no gen_act
  stage at all. flash-attn 2.7.4.post1 pin still applies.
- CPU pre-verification (commit 40edf40): s=0 logits bitwise-identical to
  vanilla HF on tiny llama+qwen; measured intermediate sparsity == s.

## Expected outcome
- **Success**: (a) s=0 PPL = 5.47 ± 0.1 (dense-equivalence gate);
  (b) printed "mlp h2" measured sparsity ≈ s while "mlp h1" / "attn h1" /
  "attn h2" ≈ 0 (placement gate); (c) PPL recorded at s=0.5/0.7/0.9.
- **Failure**: s=0 deviates > 0.1 from 5.47 (implementation bug), or measured
  sparsity ≠ s (masking bug). PPL blow-up at high s is a *finding*, not a
  failure of the experiment.

## Reproducibility
- **Git tag**: `exp/2026-07-22_larosa-llama2-topk-int-ppl` (repo
  `aiha-choij/EfficientAI`, commit 6e3c91e; mode code in 40edf40)
- **Job ID**: `20260723-133910-larosa-llama2-topk-int-ppl` (2nd attempt; see Notes)
- **Assigned host/GPU**: a100-40-2 (pinned via -H), GPU 1 (PCI index)
- **Command**: `bash scripts/repro_topk_ppl.sh llama /raid/LLM/llama2-7b`
  (cwd `/home/choij/workspace/repos/EfficientAI/larosa`; qsub `-g 1 -m 30`)
- **Config path**: n/a — parameters passed as script args; mode/sparsity set
  at runtime via `--mode topk_intermediate --sparsity`
- **Key parameters**: sparse_mode=topk_intermediate; sweep s=0.0/0.5/0.7/0.9;
  K = int((1−s)·11008) per token (top_k_new truncation, not round); ctx 2048;
  bf16; flash_attention_2; no calibration, no rotation matrices
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3,
  flash-attn 2.7.4.post1
- **Model**: `/raid/LLM/llama2-7b` (shared read-only meta-llama/Llama-2-7b-hf)

## Notes (submission history)
- Attempt 1 (`20260723-133728`) failed at launch: workdir was passed to qsub as
  a quoted `~/...` path; the tilde stayed literal and the dispatcher's
  `cd "$workdir"` failed silently, falling back to the rundir where
  `scripts/repro_topk_ppl.sh` doesn't exist. Resubmitted with the absolute path.
  Dispatcher-side fix (expand/reject tilde workdirs) flagged as a QCom task.

### Results

### Interpretation
