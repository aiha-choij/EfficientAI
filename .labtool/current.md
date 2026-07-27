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
Focus: oracle-residual-sparsity — 2026-07-27 within-topic steer (user): C4
compensation moves to the SLR line — R-Sparse's sparse/low-rank split
grafted onto Mx (H4). Two variants at matched B_eff budget: S1 static hot
rank-1 neuron terms + SVD(M_cold); S2 dynamic top-|x| input channels of the
SVD residual. Diagnostics-first (E0 offline gates) → E1 PPL sweep vs the
r=1024 anchor (7.229 @ s=0.9). Old proposals (quantized M, loss-aligned
fitting) deprioritized, not abandoned. Implementation plan in gist Next
Experiments; code lands in oracle_mlp.py + scripts/oracle/09_slr_diag.py.

## Active Jobs
- `050-20260728-084743-oracle-llama2-e0-slr-diag` — oracle-residual-sparsity
  E0 SLR diagnostics (a6000-2 pinned, 1 GPU ≥30GiB, PENDING). Card:
  topics/oracle-residual-sparsity/journal/2026-07-27_experiment-oracle-llama2-e0-slr-diag.md
- NOTE: a6000-2 execution env stays available (venv ~/workspace/venv-larosa,
  sdpa, model /raid/LLM/llama2-7b, stats/factors under ~/workspace/oracle).
- NOTE: a6000-4 is now also a llama2-capable execution host (venv-larosa +
  repo replicated from a6000-2, model /raid/LLM/llama2-7b, 3 idle 48GB GPUs
  at last check).

## Direction
Mean-gate residual decomposition on LLaMA2-7B (top-K s={0.5,0.7,0.9},
wikitext-2 PPL): H1 confirmed — exact compensation (C3) cuts C1's degradation
56% at s=0.9. The open front is the DEPLOYABLE compensation (C4), now on the
SLR line (H4, 2026-07-27 steer): comp(x) ≈ Mx as sparse + low-rank at
matched MAC budget — S1 static hot rank-1 neurons + SVD(M_cold), S2 dynamic
top-|x| input channels of the SVD residual (R-Sparse template; M's heavy
mid-stack singular tail is the motivation). Anchor to beat: plain r=1024
(7.229 @ s=0.9, +6.2% compute). Whitening and spectral-energy allocation
remain dead ends. Specs: topics/oracle-residual-sparsity/spec.md +
spec-c4-whitening.md; steer card
journal/2026-07-27_pivot-c4-slr-compensation.md.

## Next Experiments (SLR line — details in gist)
1. E0 offline SLR diagnostics (minutes, 1 GPU): S1 hot-set-removal spectra
   (r90, energy@r) + S2 x-channel concentration and matched-budget approx
   error (screening only). Needs a small x-capture pass + new
   scripts/oracle/09_slr_diag.py.
2. E1 SLR PPL sweep: c4 comp_mode {slr_neuron, slr_input}, B_eff=1024
   splits, s={0.5,0.7,0.9}. Gate: beat r=1024 by ≥0.05 PPL @ s=0.9 →
   B_eff=2048 round (subsumes old r=2048 arm).
3. [deprioritized, kept] Quantized full-rank M W4/W8; loss-aligned A,B
   fitting (deferred, unchanged).

## Latest
- 2026-07-27: `oracle-llama2-e0-slr-diag` SUBMITTED
  (050-20260728-084743, a6000-2) — E0 offline SLR gates: hot-set-removal
  spectra + x-channel concentration + matched-budget rel-err @ B_eff=1024.
  SLR code (slr_neuron/slr_input c4 modes) landed at tag
  exp/2026-07-27_oracle-llama2-e0-slr-diag; unit tests + tiny smoke pass.
- 2026-07-27: STEER (within-topic, oracle-residual-sparsity): C4
  compensation → SLR hybrid (R-Sparse template on Mx). H4 added; E0
  diagnostics + E1 PPL sweep planned; quantized-M/loss-aligned proposals
  deprioritized (kept in gist). Card:
  topics/oracle-residual-sparsity/journal/2026-07-27_pivot-c4-slr-compensation.md
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
