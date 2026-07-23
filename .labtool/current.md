# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| oracle-residual-sparsity | 🟢 active | Oracle top-p on FFN intermediate: mean-gate residual r=u⊙(g−ḡ) + rank-r compensation (C0–C6 ladder), lm-eval critical sparsity |
| larosa-intermediate-sparsity | ✅ done | Per-token Top-K on i=u⊙g confirmed on LLaMA2-7B (50% → +0.047 PPL); closed by pivot, 3-model ext in backlog |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |

## This Session
Focus: oracle-residual-sparsity — pivot recorded; scope narrowed to LLaMA2-7B
+ wikitext-2 PPL + C4 r=512 only (user). Phase 1 DONE: oracle mode implemented,
all 4 unit tests pass on CPU. Next: Phase-2 calibration job.

## Active Jobs
- `050-20260723-155254-oracle-llama2-phase0-calib` — Phase 2: calibration
  (wikitext103 + c4) + phase-0 i-vs-r report + M factors r=512, LLaMA2-7B,
  a100-40-2. Card: topics/oracle-residual-sparsity/journal/
  2026-07-22_experiment-oracle-llama2-phase0-calib.md

## Direction
New spec (2026-07-22): in the oracle setting, decompose the FFN via the
calibration gate mean — y = compensation(Mx) + sparse residual r = u⊙(g−ḡ) —
and test whether the residual's concentration (H1), weight-aware scoring (H2),
and rank-r compensation with r ≤ h/8 (H3) push critical sparsity beyond plain
|i| top-p at equal effective compute. Conditions C0–C6, models
Llama-3.2-3B → Llama-3.1-8B → Gemma-3-4b-pt, judged by lm-eval zero-shot
normalized accuracy ≥ 0.99 (critical sparsity), wikitext PPL secondary.
Implementation is standalone in EfficientAI (no R-Sparse fork — user decision);
attention and all linears stay dense, only the MLP forward is wrapped,
compute-then-mask simulation. Full spec: topics/oracle-residual-sparsity/spec.md.

## Next Experiments
1. Phase 2: one GPU job on LLaMA2-7B — 01_calibrate (c4 + wikitext103 stats),
   02_distribution_report (go signal = r-curve above i-curve), 03_build_M r=512.
2. Phase 3: dense + C1 PPL sweep over p grid (oracle_ppl_sweep.sh); sanity
   vs Top-K anchors (5.521/5.730/8.108 at s=50/70/90).
3. Phase 4: C2–C5 sweeps, one job per condition; main PPL-vs-sparsity table.

## Latest
- 2026-07-22: `oracle-llama2-phase0-calib` submitted (050-20260723-155254,
  tag exp/2026-07-22_oracle-llama2-phase0-calib, pinned a100-40-2) — Phase 2:
  two-corpus calibration + phase-0 i-vs-r report (H1 go/no-go) + M r=512.
- 2026-07-22: Phase 1 DONE — `sparse_mode='oracle'` (C0–C5) implemented in
  inference/oracle_mlp.py + modeling hooks; scripts/oracle/01–04 + sweep; all
  4 spec unit tests pass on CPU (identities to ~1e-7). Scope narrowed by user:
  LLaMA2-7B only, wikitext-2 PPL only, C4 r=512 only.
- 2026-07-22: PIVOT — larosa-intermediate-sparsity closed (done; hypothesis
  confirmed on LLaMA2-7B, 3-model ext moved to backlog). New topic
  oracle-residual-sparsity: mean-gate residual decomposition + rank-r
  compensation, oracle top-p, C0–C6, lm-eval critical sparsity. Spec preserved
  verbatim; no R-Sparse fork (implement standalone in EfficientAI).
- 2026-07-22: `larosa-llama2-topk-int-ppl` DONE — gates pass (s=0 ≡ 5.47356,
  sparsity of i ≈ s); PPL 5.521/5.730/8.108 at s=50/70/90. Intermediate 70%
  beats input-mode 50%. Hypothesis confirmed on LLaMA2-7B.
- 2026-07-22: `topk_intermediate` implemented (40edf40) — config.sparse_mode flag,
  MLP Top-K on i, dense attention, no Q loading; CPU tiny-model tests: s=0
  bitwise-identical to vanilla HF (llama+qwen), measured sparsity == s.
- 2026-07-22: PIVOT — larosa-repro closed (done); new topic
  larosa-intermediate-sparsity: FFN intermediate Top-K, no rotation, s=50/70/90.
- 2026-07-22: llama3-8b + qwen25-7b PPL DONE — all 12 points across 3 models
  within ±0.1 of paper Table 2. Reproduction complete.

## If you're starting a new session
- Focus topic: oracle-residual-sparsity. Read its gist.md first, then spec.md
  for full implementation detail (spec is authoritative for the design).
- Immediate next action: Phase 1 — implement OracleSparseMLP in EfficientAI
  (follow the sparse_mode plumbing pattern of commit 40edf40), pass the 4 unit
  tests on a CPU tiny model before touching GPUs.
- Blockers to clear before Phase 2: (1) HF gated access + HF_TOKEN on gateway
  for Llama-3.2-3B / Llama-3.1-8B / gemma-3-4b-pt; (2) lm-eval-harness install
  into conda env `larosa`; (3) decide rank grid (proposed h/32, h/16, h/8) and
  8B version (3.1-8B download vs existing /raid/LLM/llama3-8b) — see gist
  Open Questions.
- Context: trusted dense PPL anchors from larosa-repro: 5.47 / 6.13 / 6.85
  (LLaMA2-7B / LLaMA3-8B / Qwen2.5-7B), pipeline noise ~±0.1. Eval-log
  sparsity labels are SWAPPED in the old pipeline (see closed topic's Key
  Findings) — do not inherit that logging code without fixing it. Dispatcher
  chokes on literal `~` in workdir — absolute paths in qsub. QCom infra
  commits are local-only, not pushed.
