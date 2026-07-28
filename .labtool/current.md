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
Focus: oracle-residual-sparsity — E1 S2 PPL sweep DONE: GATE PASSED with
margin. Best arm s2 r256:k1536 = 5.5961/5.7526/6.9417 vs lr_r1024
5.7365/5.9152/7.2294; gap to C3 halved (+0.591 → +0.304 @ s=0.9) at equal
budget. PPL optimum is the MIXED split (rank share ~25%), not E0's
pure-sparse pick — validate allocation on PPL only. Escalating per
pre-approved plan: E2 = B_eff=2048 round (lr_r2048 + 3 s2 splits × 3s).

## Active Jobs
- (none)
- NOTE: a6000-2 execution env stays available (venv ~/workspace/venv-larosa,
  sdpa, model /raid/LLM/llama2-7b, stats/factors under ~/workspace/oracle).
- NOTE: a6000-4 is now also a llama2-capable execution host (venv-larosa +
  repo replicated from a6000-2, model /raid/LLM/llama2-7b, 3 idle 48GB GPUs
  at last check).

## Direction
Mean-gate residual decomposition on LLaMA2-7B (top-K s={0.5,0.7,0.9},
wikitext-2 PPL): H1 confirmed — exact compensation (C3) cuts C1's degradation
56% at s=0.9. The open front is the DEPLOYABLE compensation (C4), now on the
SLR line (H4): CONFIRMED on PPL by E1 — s2 slr_input (top-|x| channels
through R = M − BA) is the best deployable C4: r256:k1536 = 6.9417 @ s=0.9
(lr_r1024 7.2294; exact C3 6.6381) at +6.2% compute. Optimum is a mixed
split (rank share ~25%); offline-L2 screening picks arms but NOT the fine
ordering. S1 (neuron hot set) dead; whitening/allocation remain dead ends.
Next front: does the edge persist at B_eff=2048, and per-layer allocation. Specs:
topics/oracle-residual-sparsity/spec.md + spec-c4-whitening.md; steer card
journal/2026-07-27_pivot-c4-slr-compensation.md.

## Next Experiments (from E2 result — details in gist)
1. E3 cheap-end sweep, B_eff ∈ {512, 256} (awaiting user go). Success: an
   SLR arm at B_eff=512 (+3.1%) beats lr_r1024 (7.2294 @ s=0.9) — half the
   compute, better quality than the previous best LR.
2. E4 per-layer (r_l, k_l) allocation at the best budget from E3.
3. [deprioritized, kept] Quantized full-rank M W4/W8; loss-aligned A,B
   fitting (deferred, unchanged).

## Latest
- 2026-07-28: REPORT published — baseline→MGR→LR→SLR with plots and compute
  accounting: results/reports/mgr-slr-report-2026-07-28.html
  (claude.ai/code/artifact/48a40f99-6da8-4d34-ac09-729a506604f9)
- 2026-07-28: `oracle-llama2-e2-s2-b2048` DONE (33 min) — gate passed: best
  s2 r256:k3584 6.6344 @ s=0.9 beats lr_r2048 6.7098 and ties exact C3
  6.6381. But 2hr = h² at r=2048, so this budget EQUALS the cost of exact
  Mx — saturation, not a deployable point. Next front is DOWNWARD
  (B_eff 512/256). Card:
  topics/oracle-residual-sparsity/journal/2026-07-28_experiment-oracle-llama2-e2-s2-b2048.md
- 2026-07-28: `oracle-llama2-e1-s2-ppl` DONE — GATE PASSED: all 3 s2 arms
  beat lr_r1024 at every s; best r256:k1536 5.5961/5.7526/6.9417 (−0.288
  @ s=0.9 vs anchor), gap to C3 halved. Mixed split beats pure sparse (E0
  fine-ordering did not transfer). Escalating to E2 B_eff=2048. Card:
  topics/oracle-residual-sparsity/journal/2026-07-27_experiment-oracle-llama2-e1-s2-ppl.md
- 2026-07-27: `oracle-llama2-e0-slr-diag` DONE (17 min) — S1 refuted (hot
  removal raises r90 1270→1470; all arms lose to lr_r1024 → Dead End); S2
  passes (all arms beat lr_r1024, best s2_r0_k2048 mid-stack rel-err 0.198
  vs 0.330, −40%; abs ≈ wnorm). User: S1 dead, E1 = S2-only 9 runs. Card:
  topics/oracle-residual-sparsity/journal/2026-07-27_experiment-oracle-llama2-e0-slr-diag.md
- 2026-07-25: `coact-llama2-p2-blocks` DONE (3.4 min) — PPMI clustering
  passes the ≥1.3× strong null on within-block mass (13/15) and dyn top-m
  coverage (15/15, 1.5–4.2×; L31 ~4×), but block-union tiling saturates
  (naive touched-blocks sharing dead) and absolute top-m coverage is only
  0.20–0.36 in mid layers. User: proceed to P3 with low expectations.
## If you're starting a new session
- Focus topic: oracle-residual-sparsity. Read gist.md (Key Findings has the
  full C4-variant table); specs: spec.md + spec-c4-whitening.md.
- Immediate next action: user ruling on the E2 saturation reading (gist Key
  Findings, E2 card Interpretation), then E3 cheap-end sweep B_eff 512/256.
  Report artifact for the whole line:
  claude.ai/code/artifact/48a40f99-6da8-4d34-ac09-729a506604f9
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
