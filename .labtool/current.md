# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| oracle-residual-sparsity | ⏸ paused | Mean-gate residual + Mx compensation: SLR best fixed-s form (6.94 @ s=0.9, +6.2%), but frontier-dominated by TIS at lower s — wrapped 2026-07-28, reopen criteria in gist |
| coactivation-block-structure | 🟢 active | Neuron permutation via co-activation clustering to make group-shared Top-K masks block-structured (P1 stats → P2 clustering → P3 oracle PPL) |
| larosa-intermediate-sparsity | ✅ done | Per-token Top-K on i=u⊙g confirmed on LLaMA2-7B (50% → +0.047 PPL); closed by pivot, 3-model ext in backlog |
| larosa-repro | ✅ done | Reproduced LaRoSa Table 2 PPL on LLaMA2/3 + Qwen2.5 (12/12 ±0.1) — trusted baseline |
| rsparse-repro | 🟢 active | R-Sparse (ICLR25) Llama-2-7B 50% reproduced: 8-task avg 64.59 vs paper 64.06, full baseline exact; matched-protocol PPL vs LaRoSA still open |
| local-loss-refit | 🟢 active | Isolated effect of refitting down_proj (closed-form ridge) to a frozen C2-score mask, no other repair — L0/L1/L2 ladder x s x token-block g. Picked up oracle-residual-sparsity's paused C2 line as a narrower follow-up. |

## This Session
Focus: local-loss-refit — new topic (2026-07-31), implementation in
progress on branch auto/local-loss-refit (refit_mlp.py, unit tests,
build/eval scripts). No GPU job submitted yet. This is a separate agent
session from the one that wrapped oracle-residual-sparsity below — that
topic's own state is left as that session recorded it.

Prior focus (unchanged, recorded by a different session): oracle-residual-sparsity — WRAPPED (user steer). E-W0 double
negative: (1) diagonal loss-aligned metric failed its gate (Spearman
identical to plain L2; cannot fix whitening inversion); (2) TIS frontier
fill-in (5.883/6.155/6.709 at s=0.75/0.8/0.85) strictly dominates every
compensated arm at s=0.9 incl. exact C3, both accountings. Surviving claim
is fixed-sparsity only. E-W1/W2/E3/E4 cancelled unrun. Report updated with
frontier postscript (same artifact URL). Topic paused with reopen criteria;
next focus = user pick from backlog.

## Active Jobs
- (none)
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
- 2026-07-28: TOPIC WRAPPED (steer) — oracle-residual-sparsity paused.
  E-W0: metric gate failed + TIS frontier dominance (TIS@0.85 6.709 beats
  SLR-B1024@0.9 6.942; TIS@0.75 5.883 beats exact C3 6.638, both
  accountings). Report postscript published. Wrap-up card:
  topics/oracle-residual-sparsity/journal/2026-07-28_pivot-wrapup-compensation-line.md
- 2026-07-28: critique session — 3 critical findings recorded: (1) SLR trails
  TIS at s=0.7 too (advantage confined to s=0.9; report+gist corrected);
  (2) iso-compute TIS-vs-SLR never measured (frontier runs added to E-W0);
  (3) remaining SLR-approx headroom is only 0.30 PPL vs 1.16 in the
  mean-gate model itself -> loss-aligned direction prioritized.
- 2026-07-28: `oracle-llama2-ew0-gradstats` SUBMITTED (050-20260729-062542,
  a6000-2, tag exp/2026-07-28_oracle-llama2-ew0-gradstats) — grad
  sensitivity w, metric validation vs 11 known-PPL variants, TIS frontier.
- 2026-07-28: REPORT published — baseline→MGR→LR→SLR with plots and compute
  accounting: results/reports/mgr-slr-report-2026-07-28.html
  (claude.ai/code/artifact/48a40f99-6da8-4d34-ac09-729a506604f9)
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
