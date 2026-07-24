# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| oracle-residual-sparsity | 🟢 active | Oracle top-p on FFN intermediate: mean-gate residual r=u⊙(g−ḡ) + rank-r compensation (C0–C6 ladder), lm-eval critical sparsity |
| larosa-intermediate-sparsity | ✅ done | Per-token Top-K on i=u⊙g confirmed on LLaMA2-7B (50% → +0.047 PPL); closed by pivot, 3-model ext in backlog |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |

## This Session
Focus: oracle-residual-sparsity — main table + C4 whitening round DONE.
Whitening/allocation both harmful (Dead Ends); rank is the working lever:
plain uniform r=1024 is the best C4 (7.229 @s=0.9, beats C1, gap to C3
+0.59) at +6.2% compute. Next decision: r=2048 convergence test vs
output-side-weighted objective design.

## Active Jobs
- (none)
- NOTE: a6000-2 execution env stays available (venv ~/workspace/venv-larosa,
  sdpa, model /raid/LLM/llama2-7b, stats/factors under ~/workspace/oracle).

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
- 2026-07-24: C4 whitening round DONE — whitening worsens PPL at every rank
  despite −13% L2 (Dead End); tau-allocation harmful (Dead End); plain
  uniform r=1024 best C4: 5.737/5.915/7.229, beats C1 @s=0.9, gap to C3
  +0.23/+0.29/+0.59 at +6.2% compute. Next: r=2048 or output-side weighting.
- 2026-07-24: PHASE 4 DONE — main table: C3 ΔPPL +0.031/+0.155/+1.164 vs C1
  +0.048/+0.255/+2.636 at s=0.5/0.7/0.9 (H1 confirmed, −56% at 0.9). C2
  worse than C1 (H2 rejected). C4 r=512 below C1 everywhere (H3 partial-go,
  rank problem). C5 < C3. Spec §8: PARTIAL-GO. Next: C4 rank sweep proposal.
- 2026-07-24: Phase 3 GATE PASSED on a6000-2 — dense 5.4738, C1 5.5216/5.7284/
  8.1096 (anchors within 0.0013). Oracle path ≡ topk_intermediate. Phase 4
  (C2-C5) submitted, one job per condition. sdpa-attr fix 36b39ee.
- 2026-07-23: Histograms DONE — |r| shifted toward 0 vs |i| in all 5 sample
  layers (med ratio 0.640 at L7 → 0.924 at L31); report artifact updated.
- 2026-07-23: `oracle-llama2-phase0-calib` DONE — H1 GO at distribution level:
  r above i in 30/32 layers, mean gap +3%p (peak +5%p mid-stack, inverted at
  layers 30-31). ḡ Pearson(c4,wt103) 0.48-0.995 (weak early). r=512 Frobenius
  energy 0.54-0.985. c4 streaming worked; primary stats = c4.
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
