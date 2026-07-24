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
- `20260724-171801-larosa-llama2-topk-overlap` @ a100-40-2 — cross-token
  neuron-selection overlap analysis (topk_intermediate, s=0.5/0.7/0.9),
  submitted from the older larosa-intermediate-sparsity session. Journal:
  topics/larosa-intermediate-sparsity/journal/2026-07-24_experiment-larosa-llama2-topk-overlap.md
- NOTE: a6000-2 execution env stays available (venv ~/workspace/venv-larosa,
  sdpa, model /raid/LLM/llama2-7b, stats/factors under ~/workspace/oracle).

## Direction
Mean-gate residual decomposition on LLaMA2-7B (top-K s={0.5,0.7,0.9},
wikitext-2 PPL): H1 confirmed — exact compensation (C3) cuts C1's degradation
56% at s=0.9. The open front is the DEPLOYABLE compensation (C4): plain
uniform rank is the only working lever (r=1024 → 7.229 @s=0.9, +6.2%
compute); whitening and spectral-energy allocation are proven dead ends
(input-space L2 misaligned with downstream loss). Specs:
topics/oracle-residual-sparsity/spec.md + spec-c4-whitening.md.

## Next Experiments
1. C4 plain uniform r=2048 (+12.4% compute): convergence test toward C3
   (success: s=0.9 gap < 0.2). One small job.
2. Output-side-weighted factorization objective — design discussion first.
3. (later) LLaMA3-8B / other-family generalization once C4 form is settled.

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
- Focus topic: oracle-residual-sparsity. Read gist.md (Key Findings has the
  full C4-variant table); specs: spec.md + spec-c4-whitening.md.
- Immediate next action: user decision between r=2048 arm and output-side
  weighting design (gist Next Experiments).
- Execution env: a6000-2 GPU0 (gateway A100s often occupied) — venv
  ~/workspace/venv-larosa (torch 2.6.0+cu124, transformers 4.46.3, sdpa, NO
  flash-attn), model /raid/LLM/llama2-7b, artifacts ~/workspace/oracle/
  llama2-7b/{stats,factors,results} (mirrored to gateway). Backend/arch
  effects ~1e-3 PPL (phase-3 gate).
- Context: dense anchor 5.4738; C1 anchors 5.5216/5.7284/8.1096; eval-log
  sparsity labels are SWAPPED (upstream bug) — JSONs carry correct values.
  Dispatcher chokes on literal `~` in workdir — absolute paths in qsub.
  `runs` ELAPSED includes queue wait — read timestamps from result files.
  Report artifact: claude.ai/code/artifact/d379b88d-fb61-49ed-aa1e-e1f987ba016a
