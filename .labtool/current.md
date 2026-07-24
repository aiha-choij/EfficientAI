# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| oracle-residual-sparsity | 🟢 active | Oracle top-p on FFN intermediate: mean-gate residual r=u⊙(g−ḡ) + rank-r compensation (C0–C6 ladder), lm-eval critical sparsity |
| coactivation-block-structure | 🟢 active | Neuron permutation via co-activation clustering to make group-shared Top-K masks block-structured (P1 stats → P2 clustering → P3 oracle PPL) |
| larosa-intermediate-sparsity | ✅ done | Per-token Top-K on i=u⊙g confirmed on LLaMA2-7B (50% → +0.047 PPL); closed by pivot, 3-model ext in backlog |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |
| rsparse-repro | 🟢 active | R-Sparse (ICLR25) Llama-2-7B 50% reproduced: 8-task avg 64.59 vs paper 64.06, full baseline exact; matched-protocol PPL vs LaRoSA still open |

## This Session
Also: rsparse-repro topic added and reproduction completed same-day (gateway
pipeline session) — see topic gist for tables and fairness caveats.
Focus: oracle-residual-sparsity — main table + C4 whitening round DONE.
Whitening/allocation both harmful (Dead Ends); rank is the working lever:
plain uniform r=1024 is the best C4 (7.229 @s=0.9, beats C1, gap to C3
+0.59) at +6.2% compute. Next decision: r=2048 convergence test vs
output-side-weighted objective design.

## Active Jobs
- NOTE: a6000-2 execution env stays available (venv ~/workspace/venv-larosa,
  sdpa, model /raid/LLM/llama2-7b, stats/factors under ~/workspace/oracle).
- NOTE: a6000-4 is now also a llama2-capable execution host (venv-larosa +
  repo replicated from a6000-2, model /raid/LLM/llama2-7b, 3 idle 48GB GPUs
  at last check).

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
- 2026-07-25: `coact-llama2-p2-blocks` DONE (3.4 min) — PPMI clustering
  passes the ≥1.3× strong null on within-block mass (13/15) and dyn top-m
  coverage (15/15, 1.5–4.2×; L31 ~4×), but block-union tiling saturates
  (naive touched-blocks sharing dead) and absolute top-m coverage is only
  0.20–0.36 in mid layers. User: proceed to P3 with low expectations.
- 2026-07-25: `coact-llama2-p1-stats` DONE (1.5 min) — union tax quantified
  at s=0.9: 6.0–6.3× @g=16, 9.1–9.4× @g=64 (92–95% of saturation 9.96×) in
  layers 0–24; layer 31 exception (3.9–6.6×). Naive union sharing dead at
  g=64; P2 clustering (strong null ≥1.3× vs random) decides the axis.
- 2026-07-24: `rsparse-llama2-repro` DONE — R-Sparse ICLR25 Llama-2-7B
  reproduced: full baseline exact (65.88), self-searched 50% recipe 64.59 vs
  paper 64.06; PPL@4096 searched +0.045 vs dense 5.1164. LaRoSA head-to-head
  needs matched protocol (seqlen + prefill caveats). Card:
  topics/rsparse-repro/journal/2026-07-24_experiment-rsparse-llama2-repro.md
- 2026-07-24: `larosa-llama2-topk-overlap` DONE — Top-K index sets are
  dominantly token-dependent: adjacent overlap 3.15× chance at s=0.9 but 68%
  disagreement in absolute terms; no always-on set; layer 31 the exception.
  Warning for block-shared masks. Report:
  https://claude.ai/code/artifact/c73a7f23-2ac7-4b27-9857-6c21a60d184f
- 2026-07-24: C4 whitening round DONE — whitening worsens PPL at every rank
  despite −13% L2 (Dead End); tau-allocation harmful (Dead End); plain
  uniform r=1024 best C4: 5.737/5.915/7.229, beats C1 @s=0.9, gap to C3
  +0.23/+0.29/+0.59 at +6.2% compute. Next: r=2048 or output-side weighting.
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
