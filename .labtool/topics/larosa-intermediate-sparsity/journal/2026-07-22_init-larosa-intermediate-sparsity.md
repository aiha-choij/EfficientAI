# Init: larosa-intermediate-sparsity

Date: 2026-07-22
Origin: pivot from `larosa-repro` (see that topic's
journal/2026-07-22_pivot-intermediate-topk-sparsity.md for the full rationale,
preserved insights, and Q&A design decisions).

Topic: FFN-only, per-token magnitude Top-K on the intermediate activation
i = u ⊙ g (down-projection input), original basis, s ∈ {50, 70, 90}%,
evaluated with the wikitext-2 PPL pipeline validated in larosa-repro.

First experiment: implement `topk_intermediate` mode in larosa and validate on
LLaMA2-7B (s=0 ≡ dense 5.47 sanity gate, measured-sparsity gate, then s sweep).
