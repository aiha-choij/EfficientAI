# 2026-07-31 — init: groupwise-flocking-tuning

## Initial hypothesis
A group×neuron ℓ2,1 (group-lasso) penalty added to the backprop-free local
FFN reconstruction loss (parent doc §5, "Sparse-BRECQ") can induce
neuron-selection flocking within token groups — transplanting BlockFFN's
CLS-aware objective from from-scratch pretraining to cheap block-local
tuning. Measured motivation: at s=0.9 within-group Top-K overlap is only
~0.19–0.32 (training-free ceiling), so the overlap itself must be learned.

## Starting phase
Method design (design questions 1–2: how ℓ2,1 coexists with the
pure-quadratic solver, and with the downstream-only parameter principle)
→ then a single-layer PoC on LLaMA2-7B.

## Notes
- Kickoff context supplied by the user on 2026-07-31 in the QCom host
  session ("아이디어 D: CLS-aware Sparse-BRECQ 탐구") — its §2 anchors,
  §4 design questions, §5 role boundaries, and §6 infra notes are
  reflected in gist.md verbatim where relevant.
- Success metrics were pre-defined at kickoff: ΔC(δ≤g) vs the 1.2–3.2×
  chance baseline, and group-shared-mask PPL vs the per-token anchor
  (dense 5.4736 / s=0.9 per-token 8.1083, LLaMA2-7B wikitext-2).
- Concurrent-session convention: current.md minimal touch (own topic row +
  focus only).
