# Experiment: block-comp-phase2-3b-round1 (Phase 2 — sharing-tax curve, first pass)

Status: PENDING
Date: 2026-07-31

## Hypothesis tested
Spec Phase 2 gate: quantify the sharing tax of a block-shared mask (C7a, no
compensation) vs the g=1 per-token anchor (oracle C2), on the PPL axis, and
check consistency with the coactivation topic's overlap measurement
(C(1)=0.316 at s=0.9 — adjacent tokens already disagree on ~68% of their
top-K neurons, so sharing tax should be substantial even at g=16).

## What we're testing over alternatives
- C7a (block-shared mask, residual score, no compensation) vs C2 (per-token
  mask, |i|*col_norm score, g=1) is not an apples-to-apples score
  comparison (C7a uses the C3/C4/C5 residual score per this topic's
  documented interpretation call, not C2's own score) — this round is
  about the SHARING TAX (mask granularity), not the score-family choice.
  A cleaner within-score-family anchor (residual score at g=1, i.e. oracle
  C3) is a candidate addition for the next round if C7a vs C2 looks
  confounded by the score difference.
- p is the top-p knob (cumulative mass), not exact sparsity s — achieved
  sparsity is the reported axis, per spec.

## Prior art check
- coactivation-block-structure P1/P3: union-tax 6-9.4x at g=16-64 (naive,
  no budget); P3's budgeted block mask (no compensation, LLaMA2-7B,
  clustered/random neuron blocks) hits PPL 4.5k-24k at s=0.9 vs 8.11
  anchor. This experiment is the token-block analog (no neuron
  permutation) on a different model (3B) — expect a real but likely less
  extreme tax, since C7a's neurons are NOT permuted/blocked, only the
  TOKEN mask is block-shared.
- oracle-residual-sparsity: C2/C4 g=1 numbers for llama3.2-3b-instruct
  don't exist yet (checked `~/workspace/oracle/` — only llama2-7b has
  calibration). This round's C2 g=1 jobs establish that anchor for the
  first time on this dev model.

## Expected outcome
C7a PPL at g=16/64, p=0.9 should be substantially worse than the C2 g=1
anchor at the same p (some sharing tax), but plausibly far less
catastrophic than coactivation P3's 4.5k-24k (no neuron permutation
involved here, and the residual-based selection may behave differently
from P3's |i|-based one). No success/failure gate for this round — it's a
measurement, not a go/no-go test; Phase 2's own gate is just "produces a
curve consistent with the coactivation overlap measurement," which needs
more p/g points than this first pass to assess.

## Reproducibility
- **Git tag**: none yet (branch `auto/block-sparse-compensation` at
  commit 0f8cdc7 / PR #2, no code changes for this experiment — eval-only)
- **Job IDs**: calibration `050-20260731-164842-block-comp-calib-3b`
  (STATUS=ok); eval `050-20260731-165225-bc-c2-g1-p0.9`,
  `050-20260731-165229-bc-c2-g1-p0.7`, `050-20260731-165233-bc-c7a-g16-p0.9`,
  `050-20260731-165237-bc-c7a-g64-p0.9` (all queued)
- **Assigned host/GPU**: a100-40-2 (pinned via -H)
- **Commands**: see job cmd files under `~/workspace/jobs/<id>/cmd`;
  calibration reused `scripts/oracle/01_calibrate.py` unmodified, eval
  used `scripts/oracle/04_eval_ppl.py` (c2 anchor) and
  `scripts/block_comp/01_eval_ppl.py` (c7a)
- **Config path**: n/a — parameters as script args
- **Key parameters**: model `/raid/LLM/llama3.2-3b-instruct` (plain
  pretrained 3B not available locally, dev-model substitution per
  established convention); calibration wikitext103, n=512, seqlen=2048;
  eval wikitext-2 test, select=topp
- **Key deps**: conda env `larosa` (gateway a100-40-2)
- **Model**: `/raid/LLM/llama3.2-3b-instruct`; outputs
  `~/workspace/oracle/llama3.2-3b-instruct/stats/wikitext103` (calib),
  `~/workspace/block_comp/llama3.2-3b-instruct/results/*.json` (eval)
- **Sync**: n/a (single-repo gateway session, no cross-host sync needed)

## Notes
Narrow first pass (4 jobs: 2 anchor p-points, 2 C7a (g,p) points) —
queue-submission rule caps a single batch at 4. Round 2 will fill in
p=0.7 for C7a and extend to more p-grid points per
`oracle-residual-sparsity/spec.md` §5's `[0.5,0.6,0.7,0.75,0.8,0.85,0.9,
0.93,0.95,0.97,0.99]` grid, prioritizing the region where the curve is
informative once round 1's shape is known.
