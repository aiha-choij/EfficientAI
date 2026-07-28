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
Focus: oracle-residual-sparsity — steer to SLR line (H4), implementation
(slr_neuron/slr_input c4 modes, tag exp/2026-07-27_oracle-llama2-e0-slr-diag),
and E0 diagnostics DONE same day: S1 refuted (hot removal RAISES r90,
1270→1470 — dead end), S2 passes decisively (all arms beat lr_r1024;
best pure-sparse −40% mid-stack rel-err; abs score suffices). User approved
E1 = S2-only 9 runs {r512:k1024, r256:k1536, r0:k2048} × s{0.5,0.7,0.9}.
Input-L2 screening caveat stands — E1 PPL is the referee.

## Active Jobs
- `050-20260728-155727-oracle-llama2-e1-s2-ppl` — oracle-residual-sparsity
  E1 S2 PPL sweep (a6000-2 pinned, 1 GPU ≥30GiB, PENDING, est. 2.5-4 h).
  User pre-approved follow-through: gate pass → B_eff=2048 + rho search;
  gate miss → input-side family dead, fall back to quantized-M. Card:
  topics/oracle-residual-sparsity/journal/2026-07-27_experiment-oracle-llama2-e1-s2-ppl.md
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
matched MAC budget. E0 verdict: S2 (dynamic top-|x| input channels of the
SVD residual) is the surviving variant — all arms beat lr_r1024 on
screening, monotone toward pure-sparse (−40% mid-stack rel-err); S1
(neuron hot set) is a DEAD END (hot removal raises r90 — hot terms ≈ M's
top subspace). Anchor to beat in PPL: plain r=1024 (7.229 @ s=0.9, +6.2%
compute). Whitening and spectral-energy allocation remain dead ends. Specs:
topics/oracle-residual-sparsity/spec.md + spec-c4-whitening.md; steer card
journal/2026-07-27_pivot-c4-slr-compensation.md.

## Next Experiments (from E0 result — details in gist)
1. E1 S2 PPL sweep (user-approved): c4 comp_mode=slr_input, abs score,
   B_eff=1024 arms {r512:k1024, r256:k1536, r0:k2048} × s={0.5,0.7,0.9} =
   9 runs. Gate: beat lr_r1024 by ≥0.05 PPL @ s=0.9 → B_eff=2048 round +
   per-layer rho search; miss → input-side family dead, fall back to
   quantized-M.
2. [deprioritized, kept] Quantized full-rank M W4/W8; loss-aligned A,B
   fitting (deferred, unchanged).

## Latest
- 2026-07-27: `oracle-llama2-e1-s2-ppl` SUBMITTED (050-20260728-155727,
  a6000-2) — 9 PPL runs: slr_input abs, arms {r512:k1024, r256:k1536,
  r0:k2048} × s{0.5,0.7,0.9}. Gate: any arm ≤ 7.179 @ s=0.9 (lr_r1024
  − 0.05) → B_eff=2048 + rho search; miss → input-side family dead.
- 2026-07-27: `oracle-llama2-e0-slr-diag` DONE (17 min) — S1 refuted (hot
  removal raises r90 1270→1470; all arms lose to lr_r1024 → Dead End); S2
  passes (all arms beat lr_r1024, best s2_r0_k2048 mid-stack rel-err 0.198
  vs 0.330, −40%; abs ≈ wnorm). User: S1 dead, E1 = S2-only 9 runs. Card:
  topics/oracle-residual-sparsity/journal/2026-07-27_experiment-oracle-llama2-e0-slr-diag.md
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
## If you're starting a new session
- Focus topic: oracle-residual-sparsity. Read gist.md (Key Findings has the
  full C4-variant table); specs: spec.md + spec-c4-whitening.md.
- Immediate next action: E1 S2 PPL sweep is RUNNING
  (050-20260728-155727, a6000-2) — on finish, record via labtool-result and
  apply the pre-agreed gate (≤7.179 @ s=0.9 → B_eff=2048 + rho search;
  miss → input-side family dead, quantized-M next). User pre-approved the
  follow-through 2026-07-27.
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
