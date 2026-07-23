# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| larosa-intermediate-sparsity | 🟢 active | Per-token Top-K on FFN intermediate activation i=u⊙g (no rotation), s=50/70/90%, wikitext-2 PPL |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |

## This Session
Focus: larosa-intermediate-sparsity — LLaMA2-7B sweep DONE, hypothesis
confirmed on the first model (50% → +0.047 PPL); next is the 3-model extension.

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
1. 3-model extension: LLaMA3-8B + Qwen2.5-7B, same s sweep via
   repro_topk_ppl.sh, one job per model → full 3×3 ΔPPL table.
2. Weight-aware scoring arm (‖W_d[:,j]‖·|i_j|) at s=0.7/0.9 on LLaMA2-7B to
   attack the s=90% degradation (+2.635).

## Latest
- 2026-07-22: `larosa-llama2-topk-int-ppl` DONE — gates pass (s=0 ≡ 5.47356,
  sparsity of i ≈ s); PPL 5.521/5.730/8.108 at s=50/70/90. Intermediate 70%
  beats input-mode 50%. Hypothesis confirmed on LLaMA2-7B.
- 2026-07-22: `larosa-llama2-topk-int-ppl` submitted (20260723-133910, tag
  exp/2026-07-22_larosa-llama2-topk-int-ppl, pinned a100-40-2).
- 2026-07-22: `topk_intermediate` implemented (40edf40) — config.sparse_mode flag,
  MLP Top-K on i, dense attention, no Q loading; CPU tiny-model tests: s=0
  bitwise-identical to vanilla HF (llama+qwen), measured sparsity == s.
- 2026-07-22: PIVOT — larosa-repro closed (done); new topic
  larosa-intermediate-sparsity: FFN intermediate Top-K, no rotation, s=50/70/90.
  Old next-steps: lm_eval packaging deprioritized, RB-Sparse carried as future work.
- 2026-07-22: llama3-8b + qwen25-7b PPL DONE — all 12 points across 3 models within ±0.1 of paper Table 2. Reproduction complete.

## If you're starting a new session
- Focus topic: larosa-intermediate-sparsity (gist has full plan + notation).
- Immediate next action: labtool-experiment — 3-model extension, two jobs:
  `scripts/repro_topk_ppl.sh llama /raid/LLM/llama3-8b` and
  `scripts/repro_topk_ppl.sh qwen ~/workspace/models/Qwen2.5-7B`
  (pin -H a100-40-2; absolute workdir path — dispatcher chokes on literal `~`).
  Gates per model: s=0 ≡ dense (6.1377 / 6.8497) ±0.1; measured sparsity ≈ s.
- Log-reading caveat: eval prints mlp/attn sparsity labels SWAPPED (upstream
  bug, see gist Key Findings) — "attn h2" is really the intermediate i.
- Context: (1) this mode needs NO rotation matrices, NO histograms, NO gen_act —
  it loads the vanilla HF model; the saved D.pt files stay untouched for future
  RB-Sparse. (2) Trusted dense baselines from larosa-repro: 5.47 / 6.13 / 6.85
  (LLaMA2-7B / LLaMA3-8B / Qwen2.5-7B), pipeline noise ~±0.1. (3) Infra fixes
  from the repro session live in QCom (dispatcher PCI_BUS_ID + subshell placement
  fix) and conda env larosa (flash-attn pinned 2.7.4.post1) — QCom commits are
  local-only, not pushed.
