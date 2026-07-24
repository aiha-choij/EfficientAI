# rsparse-repro

## Status
active (reproduction complete, comparison analysis open)

## Hypothesis
R-Sparse (ICLR 2025, arXiv:2504.19449) claims training-free 50% model-level
sparsity on Llama-2-7B with minimal degradation via rank-aware activation
sparsity (per-module sparse/low-rank split, evolutionary-searched ratio rho).
If reproducible, it becomes a comparison baseline for our LaRoSA line.

## Key Findings
- Full baseline reproduced EXACTLY: 8-task zero-shot avg 65.88 = paper 65.88
  (6/8 tasks identical to the digit) — eval setting validated.
- R-Sparse 50% with self-implemented Algorithm 1 search: avg 64.59 vs paper
  64.06 (+0.53). Paper's Llama-2-7B recipe/search code is unreleased; our
  reimplementation (utils/search_recipe.py in fork) suffices.
- Uniform rho sweep @s=0.5: 0.7 → 64.31, 0.5 → 63.63, 0.3(fp16) → 63.17.
  Monotone in rho; searched alpha mean 0.95 — sparse-heavy is right, matches
  paper Table 3 (pure low-rank fails).
- Search recovers BoolQ: 69.94 (uniform 0.7) → 74.04 (searched), paper 72.84.
- WikiText-2 PPL (seqlen 4096): dense 5.1164; searched@50% +0.045;
  uniform0.7@50% +0.101; uniform0.7@40% +0.050.
- vs LaRoSA repro (seqlen 2048, dense 5.4736): LaRoSA@40% +0.143,
  larosa-topk@50% +0.047. R-Sparse deltas look competitive BUT see caveats.

## Dead Ends
(none — reproduction succeeded)

## Open Questions
- Fairness caveats before claiming R-Sparse > LaRoSA: (1) seqlen 4096 vs
  2048; (2) R-Sparse eval leaves first 10% of tokens dense
  (prefill_ratio=0.1) → effective sparsity ~45% not 50%; (3) rho=0.3 row is
  fp16. Rerun R-Sparse PPL at seqlen 2048 with prefill_ratio→1 decode-only
  protocol for a clean head-to-head?
- Llama-3-8B / Mistral-7B extension (paper Table 1 rows) not yet reproduced
  except piqa sanity.

## Next Experiments
1. Matched-protocol PPL: R-Sparse @s=0.4/0.5, seqlen 2048, quantify the
   prefill-dense advantage (one small job).
2. (optional) Llama-3-8B 8-task with released recipe for full Table 1 row.

## Active Jobs
(none)

## Pointers
- Fork with repro scripts + searched recipe: github.com/aiha-choij/R-Sparse
  (REPRODUCE.md has full tables; config/llama2_sparsity_50_evolutionary_search.npy)
- Exec env: a100-40-1, conda env r_sparse (~/workspace/miniconda3), py3.8.19
  torch 2.4.0+cu124 transformers 4.39.2; low-rank weights
  ~/workspace/repos/low_rank_models/{llama-2-7b,llama-3-8b} (fp32 default,
  RSPARSE_DTYPE=float16 override available)
- Gotchas: evaluation.py accepts ONE task per call; rho<=0.3 fp32 OOMs on
  40GB; dispatcher double-books GPU during CPU warmup (serialize jobs or fix
  GRACE_SEC — separate QCom task filed)
