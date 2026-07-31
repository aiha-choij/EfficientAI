# Experiment: block-comp-phase3-8b (Phase 3 — C7/C8a/C8 sweep, 8B, formal Go gate)

Status: PENDING (p-probe done, main sweep running)
Date: 2026-07-31

## Hypothesis tested
Spec's actual Go gate (section 5): "8B, g=16, s≈0.9 구간에서 C8 회수율 ≥
50%". The 3B leg (journal
`2026-07-31_experiment-block-comp-phase2-3b-round1.md`) confirmed a
Go-crossing recovery pattern at sparsity~0.52 (p=0.7), not s≈0.9 — this
experiment finds the p that lands near sparsity 0.9 on the MAIN model
and reruns the same C7a/C7/C8a/C8 protocol there, which is what the
spec's own gate actually requires.

## What we're testing over alternatives
- Llama-3.1-8B, not the 3B dev model — the spec gate is defined for 8B.
- p-probe first (2 cheap C3 jobs) rather than guessing p from the 3B
  curve directly: 3B and 8B have different weight distributions, so the
  p-to-achieved-sparsity mapping isn't guaranteed to transfer even though
  both use the same oracle-format residual score.
- Same score/mask logic as 3B (C3/C4/C5-style residual score,
  block-aggregated) — no new code, this experiment only exercises
  Phase 1's existing block_comp_mlp.py conditions on a new model.

## Prior art check
- 3B leg (Phase 2+3, CONFIRMED): sharing tax nonlinear in sparsity
  (ΔPPL~4.7 at sparsity~0.2-0.24, ~21-23 at ~0.47-0.52); C7 recovers only
  ~18-25%, C8a (diagnostic) recovers ~98-99%, C8 (deployable) crosses 50%
  Go at r_sk=d/8 (~66-67.5%) at both g=16 and g=64 tested. This
  experiment checks whether that pattern holds at the spec's actual
  target sparsity (~0.9) on the main model.
- 8B oracle calibration already done (`bc-calib-8b`, no anomaly).

## Expected outcome
Given the 3B pattern was consistent across g=16/64 and across sparsity
levels (tax grows with sparsity, C8 recovery grows with rank), the
prior is that C8 at a large-enough r_sk will again cross 50% at s≈0.9 on
8B too -- but sparsity 0.9 is HIGHER than anything tested on 3B (max was
~0.52), and the 3B curve showed tax growing steeply with sparsity, so
it's also plausible recovery is harder to achieve at this more extreme
point. No strong prior on which; report whichever happens.

## Reproducibility
- **Git tag**: none yet (branch `auto/block-sparse-compensation`, commit
  `6b80dcf` as of the no_grad fix)
- **Job IDs**: probes `050-20260731-200313-bc-c3-g1-8b-p05` (p=0.5,
  STATUS=ok), `050-20260731-200318-bc-c3-g1-8b-p03` (p=0.3, STATUS=ok);
  g=1 anchor `050-20260731-201235-bc-c3-g1-8b-p07` (STATUS=ok); C7a
  `050-20260731-200648-bc-c7a-g16-8b-p05` (STATUS=ok);
  pre-fix (killed, superseded): `bc-c7-g16-8b-p05-r896` (OOM, failed),
  `bc-c7-g16-8b-p05-r896b`/`bc-c8a-g16-8b-p05-rsk448`/`bc-c8-g16-8b-p05-rsk1792`
  (all killed via `runs -k` after 35-49min stall, see Notes below);
  resubmitted post-fix as `bc-c7-g16-8b-p05-r896c`,
  `bc-c8a-g16-8b-p05-rsk448b`, `bc-c8-g16-8b-p05-rsk1792b` (all queued
  as of this writing).
- **Assigned host/GPU**: a100-40-2 (pinned via -H)
- **Commands**: probes via `scripts/oracle/04_eval_ppl.py --condition c3`
  (oracle-format, reusing 8B calibration); sweep via
  `scripts/block_comp/01_eval_ppl.py` (same as the 3B sweep)
- **Config path**: n/a — parameters as script args
- **Key parameters**: p=0.5 chosen from the probe (see Results below);
  g=16 (matches the 3B sweep's primary block size); rank=896=d/16 for C7
  (d=14336 for 8B, vs 3B's rank=512=d/16 with d=8192 -- scaled to keep
  the same *fraction* of d, not the same absolute rank); r_sk=448=d/32
  for C8a, r_sk=1792=d/8 for C8 (the best-performing fraction found on
  3B's r_sk sweep)
- **Key deps**: conda env `larosa` (gateway a100-40-2)
- **Model**: `/raid/LLM/Llama-3.1-8B`; stats
  `~/workspace/oracle/Llama-3.1-8B/stats/wikitext103`; outputs
  `~/workspace/block_comp/Llama-3.1-8B/results/*.json`
- **Sync**: n/a (single-repo gateway session)

## Notes
p-to-sparsity direction correction (recorded because an earlier gist
note had it backwards): LOWER p means HIGHER achieved sparsity (less
cumulative mass kept = more aggressive masking), not the reverse.

### Results: p-probe
| p | achieved sparsity | PPL (C3, g=1, diagnostic anchor only) |
|---|---|---|
| 0.5 | 0.8814 | 7.8314 |
| 0.3 | 0.9565 | 15.7063 |

p=0.5 (sparsity 0.8814) is close enough to the spec's "s≈0.9 구간" to use
directly -- didn't spend a third job narrowing further (linear
interpolation between these two points would suggest p~0.45 for exactly
0.90, but 0.8814 is within the spec's own approximate "≈0.9" framing).
Proceeded straight to the main sweep at p=0.5, g=16.

### Main sweep round 1: C7a lands, C7 OOM's on a NEW bug (fixed)
`bc-c7a-g16-8b-p05`: g=16, p=0.5 -> **s_block=0.7337, PPL 49.4909**
(block aggregation flattens sparsity below the g=1 anchor's 0.8814, same
qualitative pattern as 3B).

`bc-c7-g16-8b-p05-r896` FAILED with CUDA OOM inside
`compute_M`/`build_M_factors`'s SVD -- a NEW instance of the same class
of bug Phase 3 already hit once (see the earlier journal card): the
comp_lr factor path (used only by C7, imported from `oracle_mlp.py`)
still ran its M=[h,h] SVD on GPU, once per layer. At 8B scale (h=4096,
32 layers -- bigger and more layers than 3B's h=3072/28, which never
tripped this) it OOM'd despite the model itself needing only ~16GB.
**Root-caused and fixed**: added a local `_build_comp_lr_factors` (CPU
SVD, same math) in `block_comp_mlp.py`, NOT a change to `oracle_mlp.py`
itself (kept untouched per this topic's convention -- that file is
shared with the paused `oracle-residual-sparsity` topic). Unit tests
(both block-comp and oracle) re-verified, no regression. Resubmitted as
`bc-c7-g16-8b-p05-r896b`.

Also queued `bc-c3-g1-8b-p07` (in-family g=1 anchor at p=0.7) to bracket
C7a's s_block=0.7337 for interpolation -- p=0.5's g=1 anchor (0.8814) is
too high (need a LOWER-sparsity g=1 point, i.e. a HIGHER p, to bracket
from below).

### Second bug in the same class, found by a stall not a crash (fixed properly this time)
The `bc-c7-g16-8b-p05-r896b` resubmit (CPU-SVD workaround) and the
still-running `bc-c8a-g16-8b-p05-rsk448` / `bc-c8-g16-8b-p05-rsk1792`
were all found stalled at a periodic check: 35-49 minutes elapsed, log
showing only checkpoint-load lines, zero eval-loop progress (verified
via `runs -l` on all three). Root-caused properly this time (superseding
the CPU-SVD "fix" explanation from the previous section): the actual bug
in both this incident and the earlier `bc-c7-g16-8b-p05-r896` OOM was
that `attach_block_factors_inplace` (and the SVD helpers it calls) were
never wrapped in `torch.no_grad()`. Model weights have
`requires_grad=True` by default (`.eval()` doesn't change that), so every
per-layer SVD call built and retained an autograd graph that nothing
ever freed -- unbounded memory growth regardless of device. The CPU-SVD
move only traded GPU OOM for CPU compute; at 8B scale (matrices up to
14336x4096) plus 3 concurrent CPU-heavy jobs contending for the same
host's cores, that CPU path was so slow it looked like a hang.

Fix: reverted `_svd_factors`/`_build_comp_lr_factors` to GPU SVD and
added `@torch.no_grad()` to `attach_block_factors_inplace` (commit
`6b80dcf`, pushed to `auto/block-sparse-compensation`). Re-verified both
`test_block_comp_units.py` and `test_oracle_units.py` -- no regression.
Killed all 3 stalled pre-fix jobs via `runs -k` (this command exists and
works -- corrects an earlier assumption in this topic that running jobs
couldn't be safely stopped) and resubmitted identically as
`bc-c7-g16-8b-p05-r896c`, `bc-c8a-g16-8b-p05-rsk448b`,
`bc-c8-g16-8b-p05-rsk1792b` (all queued, a100-40-2 pinned, 3 jobs in this
batch).

### p=0.5 round lands clean -- strong recovery, but sparsity target and metric formula both need a caveat
All 3 resubmitted jobs completed cleanly (git_commit `7524fad`, the
no_grad-fixed code):
- C7 (rank=896): s_block=0.7500, **PPL 16.1227**
- C8a (r_sk=448=d/32): s_block=0.7500, **PPL 7.7136**
- C8 (r_sk=1792=d/8): s_block=0.7370, **PPL 11.5803**

Using the g=1 anchor from `bc-c3-g1-8b-p07` (PPL 6.7521, sparsity 0.7541
-- reasonably close to all three conditions' achieved sparsity here,
unlike some earlier rounds' mismatches) and C7a (PPL 49.4909,
s_block=0.7337) as the tax baseline, recovery = (C7a_PPL −
condition_PPL)/(C7a_PPL − anchor_PPL):

| condition | PPL | recovery |
|---|---|---|
| C7a (no comp) | 49.4909 | 0% |
| C7 (mean-gate) | 16.1227 | **~78%** |
| C8 (deployable, r_sk=d/8) | 11.5803 | **~89%** |
| C8a (diagnostic) | 7.7136 | **~98%** |
| g=1 anchor | ~6.75 | 100% |

All three cross the spec's 50% Go bar by a wide margin -- a much
stronger result than the 3B leg, where C7 recovered only ~18-25% and C8
needed r_sk=d/8 just to cross 50%. Not accepting this at face value as a
final verdict, for two reasons:

1. **Achieved sparsity (~0.74-0.75) is well short of the spec's literal
   "s≈0.9" gate target.** Block aggregation flattens s_block below the
   g=1 anchor's nominal sparsity (same pattern noted for C7a earlier),
   and at p=0.5 that flattening is large enough (g=1 anchor 0.8814 at
   this p → s_block~0.74-0.75) that this round is really testing a
   ~0.75 regime, not ~0.9. Given the 3B finding that the sharing tax
   grows steeply and *nonlinearly* with sparsity, recovery at the actual
   ~0.9 target is not guaranteed to look like this -- could be
   meaningfully lower. Submitted a p=0.3 round (full C7a/C7/C8a/C8,
   4 jobs) to land closer to s_block≈0.9 before declaring a formal
   verdict: `bc-c7a-g16-8b-p03`, `bc-c7-g16-8b-p03-r896`,
   `bc-c8a-g16-8b-p03-rsk448`, `bc-c8-g16-8b-p03-rsk1792`.
2. **Recovery here is computed as a PPL-ratio proxy, not the spec's
   literal formula.** Re-reading spec.md §5's "지표" section during this
   check: the spec defines recovery rate as a ratio of *critical
   sparsities* (`normalized accuracy ≥ 0.99` max sparsity per oracle
   spec's definition), not PPL differences at one matched sparsity point
   -- i.e. `[critical_sparsity(condition) − critical_sparsity(C7a)] /
   [critical_sparsity(g=1 anchor) − critical_sparsity(C7a)]`. Every
   number in this topic since Phase 2 (sharing tax curve, Phase 3 3B,
   this round) has used the PPL-ratio proxy instead, without previously
   flagging the discrepancy explicitly -- an interpretation gap, not
   caught until now. Computing true critical-sparsity would require a
   full accuracy-sweep per condition (much more expensive, mirrors
   oracle spec §5's harness), which hasn't been done anywhere in this
   topic. Flagging this now as an unconfirmed interpretation call (same
   category as the block-score interpretation call already in Open
   Questions) rather than re-deriving every existing result under a
   different metric this late -- the PPL-ratio proxy is a reasonable
   stand-in for "how much of the tax gap is closed" and preserves
   comparability with everything measured so far, but the formal
   Go/Partial-go/No-go declaration should note this is under the proxy
   metric, not the spec's literal formula.

### g=1 anchor result: close single-point estimate near C7a's sparsity
`bc-c3-g1-8b-p07`: p=0.7 -> sparsity 0.7541, PPL 6.7521 -- close to
(slightly above) C7a's s_block=0.7337, not an exact bracket. Given the
3B experiment's anchor curve was quite flat over comparable sparsity
ranges (log-PPL nearly constant across a 0.3+ sparsity span), using this
single nearby point as an approximate anchor for sp~0.73 is reasonable
rather than spending another job for a tighter bracket -- flagged as an
approximation, not treated as exact. Preliminary sharing-tax read at
this point: C7a PPL 49.4909 vs anchor ~6.75 -> a very large tax at this
near-s=0.9 8B operating point (much bigger, in absolute PPL terms, than
anything measured on 3B, consistent with the 3B finding that tax grows
steeply with sparsity -- this is now a much higher sparsity than the 3B
sweep ever reached). C7/C8a/C8 results (still running) will show whether
compensation can recover a useful fraction of this much larger gap.
