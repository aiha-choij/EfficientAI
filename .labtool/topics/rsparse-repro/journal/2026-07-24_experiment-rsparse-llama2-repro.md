# experiment: rsparse-llama2-repro (2026-07-24)

Status: DONE
(Note: run outside labtool-experiment — no PENDING card existed; card created
at result time. All numbers below are from gateway job logs.)

### Hypothesis tested
R-Sparse (ICLR 2025) Table 1 Llama-2-7B results (Full 65.88, R-Sparse50%
64.06 8-task avg) are reproducible from the public repo, with the unreleased
per-module recipe replaced by our own Algorithm 1 reimplementation.

### Setup
- Fork aiha-choij/R-Sparse (scripts/repro/*, utils/search_recipe.py,
  utils/ppl_wikitext.py); server a100-40-1; conda env r_sparse (py3.8.19,
  torch 2.4.0+cu124, transformers 4.39.2); fp32 eval, bs=1, bundled lm-eval;
  prefill_ratio=0.1 (repo default).
- Search: DE, pop 32, 5 gens, group 28, C4 16×4096 ppl loss, seed 42.

### Results
(artifacts: gateway ~/workspace/runs/<job>/log; job IDs listed)
- Sanity Llama-3-8B piqa [20260724-055016]: full 79.71 (paper 79.71 exact),
  r_sparse50 79.38 (paper 77.69).
- Full baseline 8-task [20260724-134238]: 69.14/78.07/93.80/31.40/57.14/
  77.74/76.35/43.43, avg 65.88 — paper 65.88, 6/8 tasks digit-identical.
- Search [20260724-065547]: final calib ppl 8.1852, alpha mean 0.952, 3.8 h.
- Searched recipe 8-task [20260724-065557]: 67.80/77.15/92.70/31.40/56.55/
  74.04/75.34/41.72, avg 64.59 (paper R-Sparse50 64.06, +0.53).
- Uniform sweep @s=0.5 [20260724-065554, 20260724-171826]: rho 0.7 → 64.31,
  rho 0.5 → 63.63, rho 0.3 (fp16) → 63.17.
- WikiText-2 PPL seqlen 4096 [20260724-171826]: dense 5.1164; searched@50%
  5.1612 (+0.045); uniform0.7@50% 5.2176 (+0.101); uniform0.7@40% 5.1660
  (+0.050). Reference larosa-repro (seqlen 2048): dense 5.4736, LaRoSA@40%
  +0.143, larosa-topk@50% +0.047.

### Interpretation
(user review pending — proposed reading, to be confirmed/edited by user)
Reproduction succeeded: eval setting validated by exact full-baseline match,
and our search recipe slightly exceeds the paper's reported R-Sparse50 row.
Cross-method PPL comparison vs LaRoSA is NOT yet apples-to-apples (seqlen
4096 vs 2048; R-Sparse leaves first 10% tokens dense → effective sparsity
~45%); matched-protocol rerun needed before any "R-Sparse beats LaRoSA at
40%" claim.

### Reproducibility
- Fork: github.com/aiha-choij/R-Sparse @ main (REPRODUCE.md, searched recipe
  config/llama2_sparsity_50_evolutionary_search.npy committed)
- Jobs (gateway queue IDs): 20260724-055016 (sanity), 20260724-134238 (full),
  20260724-065547 (search), 20260724-065557 (searched eval),
  20260724-065554 (uniform 0.7), 20260724-171826 (uniform 0.5/0.3 + ppl×4)
- Commands: see REPRODUCE.md "재현 절차"; low-rank weights at
  a100-40-1:~/workspace/repos/low_rank_models/
- Known pitfalls: evaluation.py single-task-per-call; rho<=0.3 fp32 OOM on
  40GB (RSPARSE_DTYPE=float16); dispatcher warmup double-booking → serialize
  or -m 38 exclusive GPU.
