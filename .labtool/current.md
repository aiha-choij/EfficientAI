# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| larosa-intermediate-sparsity | 🟢 active | Per-token Top-K on FFN intermediate activation i=u⊙g (no rotation), s=50/70/90%, wikitext-2 PPL |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |

## This Session
Focus: larosa-intermediate-sparsity — pivoted from input-side (LaRoSa) to
intermediate activation Top-K sparsification; experiment plan ready, code not
yet written.

## Active Jobs
- (none)

## Direction
Sparsification point moves from the FFN input x (LaRoSa's rotated threshold
mode) to the intermediate activation i = u⊙g — the most sparsity-tolerant FFN
component per the universal-properties study (arXiv:2509.00454). Mechanism:
per-token magnitude Top-K (K = round((1−s)·d)), FFN-only, attention dense, no
rotation, no calibration. Validate implementation on LLaMA2-7B against the
trusted dense baseline (5.47), then sweep s=50/70/90 and extend to LLaMA3-8B +
Qwen2.5-7B. Baseline comparison: dense ΔPPL only.

## Next Experiments
1. Implement `topk_intermediate` mode in larosa (mlp.py forward + ppl_test
   `--mode` flag + no-gen_act runner); LLaMA2-7B gates: s=0 ≡ 5.47, measured
   zero-fraction ≈ s; then s=50/70/90 PPL.
2. Extend sweep to LLaMA3-8B (dense 6.13) + Qwen2.5-7B (dense 6.85) → 3×3
   ΔPPL table.

## Latest
- 2026-07-22: PIVOT — larosa-repro closed (done); new topic
  larosa-intermediate-sparsity: FFN intermediate Top-K, no rotation, s=50/70/90.
  Old next-steps: lm_eval packaging deprioritized, RB-Sparse carried as future work.
- 2026-07-22: llama3-8b + qwen25-7b PPL DONE — all 12 points across 3 models within ±0.1 of paper Table 2. Reproduction complete.
- 2026-07-22: `larosa-llama3-8b-ppl` + `larosa-qwen25-7b-ppl` submitted (validated runner, both on a100-40-2).
- 2026-07-22: labtool initialized; conda env `larosa` + Qwen2.5-7B download started on a100-40-2.
- 2026-07-22: found LLaMA2-7B/LLaMA3-8B already on gateway at `/raid/LLM/` (read-only) — no HF token needed.
- 2026-07-22: fixed env: flash-attn pinned to 2.7.4.post1 (2.8.x wheel needs GLIBC≥2.32, gateway has 2.31).

## If you're starting a new session
- Focus topic: larosa-intermediate-sparsity (gist has full plan + notation).
- Immediate next action: implement `topk_intermediate` mode in
  `larosa/inference/mlp.py` + `scripts/ppl_test_larosa_llama.py` (--mode flag),
  new runner without gen_act; then labtool-experiment for the LLaMA2-7B job.
- Context: (1) this mode needs NO rotation matrices, NO histograms, NO gen_act —
  it loads the vanilla HF model; the saved D.pt files stay untouched for future
  RB-Sparse. (2) Trusted dense baselines from larosa-repro: 5.47 / 6.13 / 6.85
  (LLaMA2-7B / LLaMA3-8B / Qwen2.5-7B), pipeline noise ~±0.1. (3) Infra fixes
  from the repro session live in QCom (dispatcher PCI_BUS_ID + subshell placement
  fix) and conda env larosa (flash-attn pinned 2.7.4.post1) — QCom commits are
  local-only, not pushed.
