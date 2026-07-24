# init: rsparse-repro (2026-07-24)

## Initial hypothesis
R-Sparse (ICLR 2025) 50% training-free sparsity results on Llama-2-7B are
reproducible from the public repo (VITA-Group/R-Sparse) despite the
Llama-2-7B search recipe being unreleased; serves as comparison baseline
against our larosa-repro numbers.

## Starting phase
Running experiments (reproduction pipeline executed same day; see result card).

## Notes
- Fork: aiha-choij/R-Sparse. Repo gaps found: no llama-2-7b config json, no
  Llama-2 recipe npy, no search code → all reimplemented in fork.
- Paper reference points (Table 1, Llama-2-7B): Full avg 65.88,
  R-Sparse40% 65.00, R-Sparse50% 64.06.
