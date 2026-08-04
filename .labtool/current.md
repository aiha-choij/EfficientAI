# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| oracle-residual-sparsity | 🟢 reopened | Mean-gate residual + Mx compensation: SLR best fixed-s form (6.94 @ s=0.9, +6.2%), but frontier-dominated by TIS at lower s — wrapped 2026-07-28, reopen criteria in gist |
| coactivation-block-structure | 🟢 active | Neuron permutation via co-activation clustering to make group-shared Top-K masks block-structured (P1 stats → P2 clustering → P3 oracle PPL) |
| larosa-intermediate-sparsity | ✅ done | Per-token Top-K on i=u⊙g confirmed on LLaMA2-7B (50% → +0.047 PPL); closed by pivot, 3-model ext in backlog |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |
| rsparse-repro | 🟢 active | R-Sparse (ICLR25) Llama-2-7B 50% reproduced: 8-task avg 64.59 vs paper 64.06, full baseline exact; matched-protocol PPL vs LaRoSA still open |
| groupwise-flocking-tuning | 🟢 active | Idea D (CLS-aware Sparse-BRECQ): group×neuron ℓ2,1 in the local reconstruction loss to LEARN within-group neuron-selection overlap; design → single-layer PoC (LLaMA2-7B) |
| group-conditional-refit | 🟢 active | Mask-conditional refit W(μ)=W⁰+Σ δ_b Δ_b (closed-form, group-ℓ2,1); E0 heterogeneity gate running as gateway request — spec in research-wiki plans/group-conditional-refit-spec.md |

## This Session
Focus (2026-07-31, QCom host session): groupwise-flocking-tuning — NEW
topic (Idea D kickoff). Resolving design Q1–2, then single-layer PoC.
Previous focus below kept for context.

Focus: oracle-residual-sparsity — WRAPPED (user steer). E-W0 double
negative: (1) diagonal loss-aligned metric failed its gate (Spearman
identical to plain L2; cannot fix whitening inversion); (2) TIS frontier
fill-in (5.883/6.155/6.709 at s=0.75/0.8/0.85) strictly dominates every
compensated arm at s=0.9 incl. exact C3, both accountings. Surviving claim
is fixed-sparsity only. E-W1/W2/E3/E4 cancelled unrun. Report updated with
frontier postscript (same artifact URL). Topic paused with reopen criteria;
next focus = user pick from backlog.

## Active Jobs
- Gateway request `20260804-171529-gc-refit-e0` (group-conditional-refit,
  a100-40-2, agent-driven) — E0 mask-cluster refit cross-evaluation +
  oracle block-mean compensation diagnostic; card:
  topics/group-conditional-refit/journal/2026-08-04_experiment-gc-refit-e0.md
- `050-20260731-171738-flocking-poc-l16` (groupwise-flocking-tuning,
  a6000-4, PENDING) — single-layer ℓ2,1 IRLS PoC, LLaMA2-7B layer 16;
  card: topics/groupwise-flocking-tuning/journal/2026-07-31_experiment-flocking-poc-l16.md
- NOTE (2026-07-29 sync): three finished jobs on the gateway not tracked by
  this repo's labtool — `20260731-1553*-refit-harness-{l0,l1}-{s05,s09}-8b`
  (a100-40-2, 완료). Presumed another project/session; confirm with user
  before recording or cleaning.
- NOTE: a6000-2 execution env stays available (venv ~/workspace/venv-larosa,
  sdpa, model /raid/LLM/llama2-7b, stats/factors under ~/workspace/oracle).
- NOTE: a6000-4 is now also a llama2-capable execution host (venv-larosa +
  repo replicated from a6000-2, model /raid/LLM/llama2-7b, 3 idle 48GB GPUs
  at last check).

## Direction
oracle-residual-sparsity wrapped (paused): the mean-gate compensation line
produced a strong WITHIN-FAMILY result (SLR r256:k1536 = 6.9417 @ s=0.9,
79% of headroom to exact C3 at +6.2% compute; ties C3 at +12.4%) but the
E-W0 frontier test showed the whole family — C3 included — is strictly
dominated at matched total compute by plain TIS at lower sparsity. Headline
negative recorded; reopen criteria: extreme corner (s≥0.95, B≤512),
kernel wall-clock, pinned-sparsity application. Wrap-up card:
topics/oracle-residual-sparsity/journal/2026-07-28_pivot-wrapup-compensation-line.md

## Next Experiments (focus topic needed — backlog candidates)
1. rsparse-repro: matched-protocol PPL head-to-head vs LaRoSA (seqlen 2048,
   prefill-dense removed) — small job, closes the fairness caveat.
2. coactivation-block-structure: P3 result recorded as catastrophic with
   interpretation still pending — close it out (result card a0c5cb2).
3. larosa-intermediate-sparsity backlog: 3-model Top-K PPL extension.

## Latest
- 2026-08-04: NEW TOPIC group-conditional-refit + E0 gate SUBMITTED as
  gateway request 20260804-171529-gc-refit-e0 — mask-cluster refit
  cross-evaluation (is the optimal refit mask-dependent?) + oracle
  block-mean compensation diagnostic. Spec (with derivation & critique):
  research-wiki plans/group-conditional-refit-spec.md. Pre-registered:
  E0 negative blocks E1.
- 2026-08-04: fusion first result seen (oracle-residual-sparsity, retry job
  ...-130635-fusion-mgr-refit-llama2b after OOM of the original): joint
  anchored ridge s=0.9 PPL 6.195 — beats all three thresholds (SLR 6.94 /
  frontier 6.71 / exact 6.64). fusion-r3 follow-up running. Result card
  recording left to the owning session (labtool-result).
- 2026-08-04: oracle-residual-sparsity REOPENED — MGR x refit fusion
  (zero-cost lever changes the frontier accounting); job
  050-20260804-121113 submitted, card journal/2026-08-04_reopen-refit-fusion.md
- 2026-07-31: flocking-poc-l16 SUBMITTED (050-20260731-171738, a6000-4,
  tag exp/2026-07-31_flocking-poc-l16) — first groupwise-flocking-tuning
  experiment; design Q1-2 resolved (IRLS + limited W_up update, design A).
- 2026-07-29: session sync — wrap of oracle-residual-sparsity confirmed and
  committed (05ce2f7); no active jobs; next session picks the focus from
  the backlog candidates below. Untracked refit-harness jobs spotted on the
  gateway (see Active Jobs note).
## If you're starting a new session
- oracle-residual-sparsity is PAUSED (wrapped 2026-07-28): read the wrap-up
  card first — journal/2026-07-28_pivot-wrapup-compensation-line.md has the
  surviving claim, the frontier-dominance evidence, and reopen criteria.
  Full report: results/reports/mgr-slr-report-2026-07-28.html
  (claude.ai/code/artifact/48a40f99-6da8-4d34-ac09-729a506604f9)
- Immediate next action: pick the next focus with the user from Next
  Experiments backlog candidates (rsparse matched-protocol PPL / coact P3
  closeout / larosa 3-model extension).
- Execution env unchanged: a6000-2 (venv-larosa, sdpa, /raid/LLM/llama2-7b,
  artifacts ~/workspace/oracle/llama2-7b); a6000-4 also llama2-capable.
  Dispatcher: absolute paths in qsub; `runs` ELAPSED includes queue wait.
