# Experiment: block-comp-phase2-3b-round1 (Phase 2 — sharing-tax curve, first pass)

Status: CONFIRMED (Phase 2 gate met; see Round 4 for the final table)
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

## Results
All 4 jobs STATUS=ok, llama3.2-3b-instruct, wikitext-2 test PPL:

| condition | g | p | achieved sparsity/s_block | PPL |
|---|---|---|---|---|
| C2 (per-token, \|i\|·col_norm) | 1 | 0.9 | 0.4592 | 11.1624 |
| C2 (per-token) | 1 | 0.7 | 0.7038 | 12.0568 |
| C7a (block, residual score, no comp) | 16 | 0.9 | 0.2375 | 15.6911 |
| C7a (block, residual score, no comp) | 64 | 0.9 | 0.2012 | 15.8210 |

Sanity checks passed: C2 g=1 p↓ → sparsity↑, PPL↑ (both directions correct).
C7a g=16→64 at fixed p=0.9: sparsity drops slightly (0.2375→0.2012, larger
blocks average the residual score flatter, keeping marginally more
neurons) and PPL rises slightly (15.69→15.82) — directionally consistent
with "larger token block = more sharing tax."

## Interpretation
- C7a's absolute PPL (15.7-15.8) is far better than coactivation P3's
  bare block-mask catastrophe (4.5k-24k) — but that comparison is not
  apples-to-apples: P3 additionally block-shares the MLP's *neuron* axis
  (via clustering/random neuron partitions on LLaMA2-7B), while C7a here
  only block-shares the *token* mask (no neuron permutation, no rank
  reduction, still one mask value per neuron per block, on 3B). The
  token-only sharing tax at g=16/64 looks real but survivable (worse than
  dense/per-token, not catastrophic) — consistent with the expectation
  that neuron-axis sharing is the more damaging axis, and motivates
  Phase 4 (P3') combining both axes.
- **The C2-vs-C7a comparison in this round is confounded by score family**
  (flagged pre-registered in this card's "What we're testing over
  alternatives"): C2 selects by |i|*col_norm, C7a by the C3/C4/C5 residual
  score |u*(g-g_bar)|*col_norm. At the same p=0.9 these produce very
  different achieved sparsity (0.459 vs 0.2375) even before g enters the
  picture, so the PPL gap (11.16 vs 15.69) mixes "sharing tax" with "score
  family difference." Round 2 adds oracle C3 at g=1 (same residual score,
  no block sharing) as the proper in-family anchor, isolating g as the
  only variable.
- Not yet a Phase 2 gate verdict: need the g=1 residual-score anchor (C3)
  plus at least one more p point to see whether the curve's shape is
  consistent with the coactivation overlap measurement (C(1)=0.316 at
  s=0.9) per the Phase 2 completion gate. Round 2 queued for both.

## Round 2 results (in-family anchor, oracle C3 g=1)
| condition | g | p | achieved sparsity | PPL |
|---|---|---|---|---|
| C3 (residual score, in-family anchor) | 1 | 0.9 | 0.5064 | 11.0900 |
| C3 (residual score, in-family anchor) | 1 | 0.7 | 0.7555 | 11.4875 |
| C7a (block, residual score) | 16 | 0.9 | 0.2375 | 15.6911 |
| C7a (block, residual score) | 64 | 0.9 | 0.2012 | 15.8210 |

**New confound found, not yet resolved**: even within the SAME score
family (residual), achieved sparsity is not matched across g at fixed p —
g=1,p=0.9 gives sparsity 0.506, but g=16,p=0.9 gives only 0.2375 (larger
blocks flatten the aggregated score across more diverse per-token peaks,
so the same cumulative-mass fraction keeps more neurons). This means the
naive PPL gap at matched p (11.09 vs 15.69) still mixes "sharing tax" with
"different achieved sparsity," exactly the trap spec section 8 pitfall 5
warns about ("달성 sparsity는 블록 단위 정의로 보고 — per-token 정의와 섞으면
회수율이 왜곡"). A fair reading needs PPL compared at MATCHED achieved
sparsity, which requires interpolating a p-sweep within each g to a common
sparsity axis (essentially the spec's own critical-sparsity machinery).
Queued 2 more C3 g=1 points (p=0.95, p=0.97) to bracket the ~0.20-0.24
sparsity region C7a occupies at p=0.9, so the next update can interpolate
a same-sparsity anchor value instead of comparing raw PPL at matched p.

## Round 3 results + first honest sharing-tax estimate
| condition | g | p | achieved sparsity | PPL |
|---|---|---|---|---|
| C3 (in-family anchor) | 1 | 0.97 | 0.3111 | 11.0504 |
| C3 (in-family anchor) | 1 | 0.95 | — | FAILED (CUDA OOM, transient — see Notes) |
| C7a | 16 | 0.7 | 0.5202 | 33.9006 |
| C7a | 64 | 0.7 | 0.4727 | 32.3566 |

`bc-c3-g1-p095` (050-20260731-175727) failed: `CUDA OutOfMemoryError`
during `attach_col_norms`, "Process 1227241 has 32.93 GiB memory in use"
on the assigned GPU — far more than any of this topic's own jobs ever
allocate (a 3B model's own footprint is ~6GB, confirmed by every
succeeding job's own memory lines). This is a different, much larger
process that was not present a few seconds earlier at dispatch time (the
probe-based dispatcher has an inherent race: it checks free memory once,
then the job's actual model load happens later) — read as transient
external GPU contention on the shared cluster, not a bug in this topic's
code or job spec. `nvidia-smi` re-checked immediately after: all GPUs back
to expected free memory, the large process was gone. No fix needed beyond
retry.
**Retry plan changed, not blindly repeated**: p=0.97's result (sparsity
0.3111) revealed p=0.95 would land BETWEEN the already-measured p=0.9
(0.5064) and p=0.97 (0.3111) points — not useful for bracketing the
target ~0.20-0.24 sparsity region, which needs p even higher than 0.97.
Queued p=0.99 instead of re-submitting the failed p=0.95 job verbatim.

**First interpolated/extrapolated sharing-tax estimate** (C3 g=1 anchor
PPL at matched achieved sparsity, linear interpolation between adjacent
measured anchor points; extrapolation flagged where sparsity falls
outside the anchor's measured range):

| g | p | C7a sparsity | C7a PPL | anchor PPL (matched sparsity) | ΔPPL (sharing tax) |
|---|---|---|---|---|---|
| 16 | 0.9 | 0.2375 | 15.6911 | ~11.03 (extrapolated beyond p=0.97 point) | ~4.66 |
| 64 | 0.9 | 0.2012 | 15.8210 | ~11.03 (extrapolated) | ~4.79 |
| 16 | 0.7 | 0.5202 | 33.9006 | ~11.11 (interpolated, in-range) | ~22.79 |
| 64 | 0.7 | 0.4727 | 32.3566 | ~11.08 (interpolated, in-range) | ~21.28 |

**Interpretation**: the sharing tax is strongly nonlinear in achieved
sparsity — roughly ΔPPL≈4.7 at sparsity≈0.20-0.24 but ΔPPL≈21-23 at
sparsity≈0.47-0.52 (a ~4.5-5x jump in absolute PPL cost for roughly a
2x increase in sparsity). This is consistent with the coactivation
topic's finding that per-token neuron-set overlap collapses fast as
sparsity rises (adjacent-token overlap only 3.15x chance at s=0.9) — the
higher the sparsity, the more a token's individually-important neurons
diverge from its block-mates', so a block-shared mask increasingly misses
each token's real needs. The two p=0.9 anchor values are extrapolated
(target sparsity is below the lowest directly-measured anchor point,
0.3111 at p=0.97) so are less certain than the p=0.7 pair, which fall
inside the measured anchor range. Round 4 (`bc-c3-g1-p099`, queued) will
convert the p=0.9 row from extrapolated to interpolated.
This is the first quantitative signal for H4 (block-wise compensation
recovery target): C7/C8 need to close a gap that itself grows sharply
with target sparsity — Phase 3's C7/C8 sweep should treat the ΔPPL≈21-23
(higher-sparsity) regime as the harder, more important test of the
compensation hypothesis, not just the ΔPPL≈4.7 regime.

## Round 4 (final anchor point) + tightened sharing-tax numbers
`bc-c3-g1-p099` (050-20260731-180037): C3 g=1, p=0.99 → sparsity 0.1949,
PPL 11.0508. Together with p=0.97 (sparsity 0.3111, PPL 11.0504), this
brackets BOTH C7a p=0.9 points (0.2375, 0.2012) from inside the measured
range — the p=0.9 row of the sharing-tax table is now interpolated, not
extrapolated. The anchor curve is very flat here (PPL 11.0504-11.0508
across sparsity 0.19-0.31), so the correction versus the earlier
extrapolate-based estimate is negligible:

| g | p | C7a sparsity | C7a PPL | anchor PPL (interpolated, in-range) | ΔPPL (sharing tax) |
|---|---|---|---|---|---|
| 16 | 0.9 | 0.2375 | 15.6911 | 11.0507 | **4.64** |
| 64 | 0.9 | 0.2012 | 15.8210 | 11.0508 | **4.77** |
| 16 | 0.7 | 0.5202 | 33.9006 | ~11.11 | **~22.79** |
| 64 | 0.7 | 0.4727 | 32.3566 | ~11.08 | **~21.28** |

**Phase 2 gate met.** The curve is fully interpolated (no more
extrapolation caveats), confirms the nonlinear-in-sparsity sharing tax
(4.5-5x PPL cost for ~2x sparsity), and is directionally consistent with
the coactivation topic's overlap-collapse measurement. Status: CONFIRMED.

## Phase 3 kickoff (same session)
First C7/C8a/C8 data point queued at the harder regime this round
identified (g=16, p=0.7, sparsity~0.52, sharing tax ΔPPL≈22.79 — the
bigger test of H4/H5 than the milder p=0.9 regime): rank=512 (C7's
comp_lr, oracle-C4-style) and r_sk=256=d/32 (C8/C8a's gate/up/down
sketch, smallest of the spec's sweep {d/32,d/16,d/8}). Jobs:
`bc-c7-g16-p07-r512`, `bc-c8a-g16-p07-rsk256`, `bc-c8-g16-p07-rsk256`
(050-20260731-1804{03,04,07} area). Recovery rate (H4 gate: >=50%) will
be computable once these land: recovery = (C7a_PPL - condition_PPL) /
(C7a_PPL - anchor_PPL) = (33.90 - condition_PPL) / (33.90 - 11.11).

## Phase 3 round 1: bug found + fixed, first result striking
Two of the three Phase 3 jobs failed with CUDA OOM
(`bc-c7-g16-p07-r512`, `bc-c8-g16-p07-rsk256`); the third
(`bc-c8a-g16-p07-rsk256`) succeeded. **Root-caused as two real bugs, not
external contention** (unlike the earlier `bc-c3-g1-p095` OOM):
1. `_svd_factors` ran `torch.linalg.svd` directly on GPU for the full
   gate/up/down projection weight matrices (8192x3072 on this dev model) —
   much larger than oracle_mlp's own SVD target M ([h,h]=[3072,3072]).
   Doing this across 28 layers x 3 projections caused unbounded CUDA
   memory growth (process hit ~19.8GB vs the model's own ~6GB footprint)
   — never caught by the CPU-only tiny-model unit tests.
2. `attach_block_factors_inplace` unconditionally built BOTH C7's
   comp_lr AND C8/C8a's sketch regardless of which one the requested
   condition actually uses — the c7 job crashed inside sketch-building it
   never needed.
Fixed: `_svd_factors` now runs on CPU (only the tiny rank-r result moves
to GPU); `attach_block_factors_inplace` takes a `condition` argument and
skips the unused half. All unit tests re-verified (no regression — the
CPU tiny-model tests never exercised this code path at real scale in the
first place, which is exactly why it shipped un-caught). Commit
`80942a6`. Both failed jobs resubmitted (`bc-c7-g16-p07-r512b`,
`bc-c8-g16-p07-rsk256b`).

**C8a result is striking**: g=16, p=0.7, r_sk=256=d/32 →
achieved sparsity 0.5397 (vs C7a's 0.5202 at the same nominal p, g — see
Open Questions for the small discrepancy), **PPL 11.3815**. Compared to
C7a's PPL 33.9006 at this regime and the ~11.08-11.11 in-family anchor,
C8a recovers almost the ENTIRE sharing tax:
recovery = (33.9006 - 11.3815) / (33.9006 - 11.11) ≈ **99%**. This is
H4's diagnostic upper bound (gate estimated via a small sketch, but u and
W_down still exact) — strong evidence that most of the sharing tax lives
in the "which neurons deviate from ḡ" signal a low-rank gate estimate can
already capture. C7 and C8 (the two resubmitted jobs) will show whether
the *deployable* forms (7's mean-gate-only compensation, 8's fully
sketched compensation) come anywhere close to this ceiling.

**C7 (resubmitted, fixed code) result — H4 Partial-go signal**: g=16,
p=0.7, rank=512 → sparsity 0.5317, PPL 28.2506.
recovery = (33.9006-28.2506)/(33.9006-11.11) ≈ **25%** — well under the
spec's 50% Go threshold, and far below C8a's ~99%. This is exactly the
spec section 5 "Partial-go" signature (C8a recovers, C7/plain mean-gate
does not): the static mean-gate ḡ compensation (oracle C4's own
mechanism, generalized to blocks) captures only a quarter of the tax at
this regime, while a per-token gate estimate (even from a small rank-256
sketch, C8a) captures nearly all of it. This is strong, direct evidence
for H5 (sharing-tax neurons are the ones whose gate deviates most from
ḡ — a mean-gate compensation structurally cannot chase that deviation,
a per-token gate estimate can). The open question is whether C8
(deployable, u also sketched) keeps C8a's ~99% or regresses toward C7's
~25% once u is no longer exact — still running.

## Notes
Narrow first pass (4 jobs: 2 anchor p-points, 2 C7a (g,p) points) —
queue-submission rule caps a single batch at 4. Round 2 will fill in
p=0.7 for C7a and extend to more p-grid points per
`oracle-residual-sparsity/spec.md` §5's `[0.5,0.6,0.7,0.75,0.8,0.85,0.9,
0.93,0.95,0.97,0.99]` grid, prioritizing the region where the curve is
informative once round 1's shape is known.
