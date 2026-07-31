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
  `7cbb446` at time of the p-probe)
- **Job IDs**: probes `050-20260731-200313-bc-c3-g1-8b-p05` (p=0.5,
  STATUS=ok), `050-20260731-200318-bc-c3-g1-8b-p03` (p=0.3, STATUS=ok);
  sweep `050-20260731-2006{48,54,59}`, `050-20260731-200704`
  (`bc-c7a-g16-8b-p05`, `bc-c7-g16-8b-p05-r896`,
  `bc-c8a-g16-8b-p05-rsk448`, `bc-c8-g16-8b-p05-rsk1792`) -- all queued,
  none landed yet.
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
