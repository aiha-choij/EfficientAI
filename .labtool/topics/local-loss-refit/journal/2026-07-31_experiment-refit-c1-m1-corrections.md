# Experiment: refit-c1-m1-corrections (section 4 honesty corrections: C1 ridge anchor, M1 log-PPL + dense anchor)

Status: CONFIRMED — C1 was a real, substantial bug, not a minor artifact
Date: 2026-07-31

## Hypothesis tested
From requesting agent request `20260731-162935-block-compensation` section 4:
- **C1**: `solve_refit`'s ridge is a 0-shrinkage prior. Candidate alternate
  explanation for the s=0.5/0.7 "refit hurts" finding: if masking at low
  sparsity leaves little true bias to correct, the 0-anchor ridge could be
  adding shrinkage-toward-zero noise (not just failing to find real signal)
  on weak-evidence columns. Fix: anchor to the original W_down instead.
  Question: does the s=0.5/0.7 conclusion survive this fix, or was it (in
  part) a ridge-prior artifact?
- **M1**: the "3B 37% / 8B 60%" sharing-tax absorption headline was computed
  on raw PPL, which exponentially distorts a multiplicative quantity.
  Recompute on log-PPL (nats). Also: no dense (s=0) PPL anchor exists yet
  to frame the result table in absolute terms.

## What we're testing over alternatives
- This is a correction pass on an ALREADY-CLOSED topic (the core Go/No-go
  question was answered and the topic marked topic-closing in the prior
  request). Not reopening the full matrix — narrow, targeted fixes.
- C1: only s=0.9 and s=0.5 at g=1 on 3B (per the request's own scope: "3B
  에서 판별 후 뒤집히는 경우에만 8B 확장") — s=0.7 and 8B are conditional
  follow-ups, not run yet.
- M1's log-PPL recomputation uses the EXISTING result JSONs verbatim (no
  re-estimation) — this is pure arithmetic, verified by hand before writing
  it into the gist (3B: raw absorb 36.66%->log 12.35%; 8B: raw 60.37%->log
  21.52%, matching the request's expected replacement figures almost
  exactly).

## Prior art check
- local-loss-refit gist Key Findings (topic-closing entry, 2026-07-31):
  s=0.9 Go (PPL -20.1%/-7.6%, accuracy +3.97%p/+... ), s=0.5 hurts (PPL
  +13.7%/+18.3%, accuracy -1.19%p), both confirmed on 2 models x 2 metrics.
  This experiment does NOT dispute that confirmation on its own terms
  (same ridge, same everything) -- it asks whether a DIFFERENT ridge
  formulation (anchored, not 0-shrinkage) changes the picture.
- Dead Ends: L2 (sequential refit) confirmed worse, not touched by this
  correction (C1's fix is applied to L2's call site for code consistency
  since solve_refit's signature changed, but L2 is not being re-litigated
  here -- out of this request's scope per its own text).

## Expected outcome
Three possibilities for C1, no priors on which: (a) s=0.5 result flips to
neutral/positive -> the original conclusion was partly a ridge-prior
artifact, topic-closing entry needs a real correction, and L2 becomes a
candidate for re-examination (M3, explicitly out of THIS request's scope
though); (b) s=0.5 stays negative but the magnitude shrinks meaningfully ->
partial artifact, headline direction survives but the effect size claim
needs updating; (c) s=0.5 is essentially unchanged -> the anchored ridge
was not the explanation, bias-variance (or another mechanism) stands as
the leading hypothesis, recorded with more confidence for having ruled out
this alternative. Report whichever happens plainly, no thumb on the scale.

## Reproducibility
- **Git tag**: none yet (branch `auto/refit-honesty-corrections`, off
  `auto/local-loss-refit` at commit `46e29d6`, this correction's own commit
  `1680b1f` — not proposed via `agent-pr` yet, pending these results)
- **Job IDs**: `050-20260731-191040-refit-dense-anchor-3b`,
  `050-20260731-191044-refit-dense-anchor-8b`,
  `050-20260731-191049-refit-c1-build-3b-s09`,
  `050-20260731-191054-refit-c1-build-3b-s05` (all queued, a100-40-2)
- **Assigned host/GPU**: a100-40-2 (pinned via -H)
- **Commands**: dense anchors use `scripts/refit/02_eval_ppl.py --mode l0
  --s 0.0 --g 1` (existing code, no changes); C1 recalibration uses
  `scripts/refit/01_build_l1.py --s {0.9,0.5} --g 1 --lambdas 0.01
  --out_dir .../weights/l1c1_s{s}_g1 --stats_out .../stats/l1c1_s{s}_g1`
  (new `--stats_out` flag, new anchored `solve_refit` call site) — eval
  (L0 vs new-L1 PPL) to follow once builds land, not yet submitted.
- **Config path**: n/a — parameters as script args
- **Key parameters**: llama3.2-3b-instruct, wikitext103 calibration
  (n=512, seqlen=2048, reusing the EXISTING saved `calib_tokens.pt` --
  same calibration set as the original s=0.5/0.9 runs, so this is a fair
  like-for-like re-solve, not a different calibration sample), lambda=0.01
  (same as original, C2's sweep comes after C1's verdict)
- **Key deps**: conda env `larosa` (gateway a100-40-2)
- **Model**: `/raid/LLM/llama3.2-3b-instruct`, `/raid/LLM/Llama-3.1-8B`
  (dense anchor only); new weights land under
  `~/workspace/refit/llama3.2-3b-instruct/weights/l1c1_s{0.9,0.5}_g1_lam0.01`
  (kept separate from the original buggy `l1_s{s}_g1_lam0.01` artifacts for
  audit/comparison), raw stats under `.../stats/l1c1_s{s}_g1/`
- **Sync**: n/a (single-repo gateway session)

## C1 recalibration builds: both done, both eval jobs queued
`refit-c1-build-3b-s09` and `refit-c1-build-3b-s05` both completed
cleanly, reusing the EXACT saved calibration tokens from the original
(buggy-ridge) runs (`calib_tokens.pt`, confirmed in the log) so this is a
fair like-for-like re-solve. Raw (G,C,n) saved to
`~/workspace/refit/llama3.2-3b-instruct/stats/l1c1_s{0.9,0.5}_g1/` per
layer -- future lambda/prior changes need no recalibration going forward.
New anchored-ridge weights saved to
`.../weights/l1c1_s{0.9,0.5}_g1_lam0.01/` (kept separate from the
original `l1_s{s}_g1_lam0.01` artifacts for audit). Eval jobs
(`refit-c1-eval-3b-s09`, `refit-c1-eval-3b-s05`) queued immediately after
-- PPL numbers not yet in, no verdict yet on whether s=0.5's "refit
hurts" finding survives the anchored ridge.

## C1 eval results — the s=0.5 "hurts" conclusion was substantially a ridge-prior artifact

Both eval jobs landed. llama3.2-3b-instruct, g=1, wikitext-2 PPL:

| s | L0 (mask only) | old L1 (0-anchor ridge) | new L1 (W_down-anchor ridge, C1 fix) | old ΔL1 | new ΔL1 |
|---|---|---|---|---|---|
| 0.9 | 21.5859 | 19.9510 | **16.1622** | -7.57% | **-25.13%** |
| 0.5 | 11.2104 | 13.26 (~+18%) | **11.2074** | +18.3% | **-0.03%** |

**Two findings, both large:**
1. **s=0.5: the "refit hurts" headline does NOT survive the C1 fix.**
   New L1 PPL (11.2074) is statistically indistinguishable from L0
   (11.2104, -0.03%) and nearly exactly the dense anchor (11.0489). The
   original +18% "hurts" conclusion was, in large part, an artifact of
   the 0-shrinkage ridge prior actively pulling weak-evidence columns
   toward zero at low sparsity (exactly the mechanism C1 flagged as a
   candidate explanation) -- not a real bias-variance property of refit
   itself. **The topic-closing headline ("refit hurts at s=0.5, s=0.7 is
   a crossover zone") is now WRONG as stated and needs correction.**
2. **s=0.9: the Go result gets substantially STRONGER, not just
   unaffected.** New ΔL1 is -25.13%, more than 3x the old -7.57%. C1 was
   not a low-sparsity-only bug -- the 0-anchor prior was leaving real
   performance on the table at s=0.9 too.

**Per the request's own conditional instruction ("3B에서 판별 후 뒤집히는
경우에만 8B 확장")**: the s=0.5 result DID flip (hurts -> neutral), so
8B verification is now in scope, not merely optional.

**Explicitly out of this request's scope (per its own text), recorded as
an observation only**: this reopens L2 (sequential refit, Dead End) as a
candidate for re-examination (M3) -- L2 also used the buggy 0-anchor
ridge, and if error compounding was partly masked by an unrelated
ridge-prior handicap on L1's side, the L1-vs-L2 comparison might look
different with both using the anchored ridge. NOT investigating this now
-- flagged for a future request only.

**Also flagged, not yet acted on**: the harness (accuracy-axis)
confirmation of "s=0.5 hurts" (Δacc -1.19%p average) was run against the
OLD buggy-ridge L1 weights too. Whether that accuracy-axis finding also
reverses with the fixed ridge is an open question this experiment does
not answer (would need a new harness run against the new weights) --
not fabricating an assumption either way.

## M1 results: dense anchors (both landed, no anomaly)
- 3B (llama3.2-3b-instruct): dense PPL = **11.0489**. Against s=0.9,g=1
  L0 (mask-only, no refit) = 21.5859: masking alone nearly doubles PPL
  (+95.4%). L1 (refit) = 19.9510, still +80.6% over dense -- refit
  recovers only part of masking's own damage, not the whole gap to dense.
- 8B (Llama-3.1-8B): dense PPL = **6.2394**. Against s=0.9,g=1 L0 =
  12.9347: masking alone +107.3% over dense. L1 = 10.3305, +65.5% over
  dense.
- Both jobs sparsity=0.0 confirmed (dense passthrough, matches the L0
  s=0 unit-test identity). This completes the M1 absolute-frame request:
  the result table can now report L0/L1 PPL relative to a real dense
  anchor, not just relative to each other.

## Notes
- The C1 code fix is mathematically guaranteed to never make in-sample fit
  worse than the anchor (W_down) at any lambda -- see the docstring in
  `refit_mlp.solve_refit` for the one-line proof (substitute V=W-W_anchor,
  ridge always beats V=0 on the transformed problem). This does NOT
  guarantee held-out (eval) improvement, which is the actual open question
  here.
- Dense anchor jobs cost nothing extra in code (reuse of `mode=l0, s=0.0`,
  which the existing unit tests already show is an exact dense-forward
  identity) -- purely a "run it once and record the number" gap-fill.

## PR #3 opened
https://github.com/aiha-choij/EfficientAI/pull/3 (stacked on `auto/local-
loss-refit` PR #1, unmerged) -- includes the C1 code fix, M1 corrections,
and the 3B recalibration results above.

## Follow-on jobs queued (8B extension + C2 lambda sweep)
Per the request's rule ("3B에서 판별 후 뒤집히는 경우에만 8B 확장") the
s=0.5 flip puts 8B verification in scope. Queued:
- `refit-c1-build-8b-s09`, `refit-c1-build-8b-s05` (Llama-3.1-8B, same
  anchored-ridge recalibration as 3B, g=1) -- confirms/refutes whether
  the 3B correction generalizes to the main model.
- `refit-c2-sweep-3b-s09`, `refit-c2-sweep-3b-s05` (llama3.2-3b-instruct,
  `--lambdas 0.001 0.01 0.1` in one job each, using the already-supported
  multi-lambda loop in `01_build_l1.py` -- one recalibration sweep,
  three solved weight sets, no new code) -- C2's lambda sweep on top of
  the now-fixed anchored ridge.

## C2 build jobs landed (both s=0.9 and s=0.5), eval jobs queued
`refit-c2-sweep-3b-s09` and `refit-c2-sweep-3b-s05` both completed
cleanly, reusing the same saved calibration tokens, each producing 3
weight sets (lambda in {0.001, 0.01, 0.1}) in a single job (the
already-supported multi-lambda loop needs one calibration sweep, not
three). Submitted the 3 eval jobs for s=0.9's sweep
(`refit-c2-eval-3b-s09-lam{001,01,1}`); s=0.5's 3 eval jobs queued for
next round (kept this submission batch to <=4 new jobs per the queue
rule). No PPL-vs-lambda numbers yet.
`refit-c1-build-8b-s09`/`-s05` (8B extension) still running.

## C2 (lambda sweep) results: mild sensitivity at s=0.9, flat at s=0.5
| s | lambda | L1 PPL |
|---|---|---|
| 0.9 | 0.001 | 16.1553 |
| 0.9 | 0.01 | 16.1622 |
| 0.9 | 0.1 | **16.0359** (best of the three) |
| 0.5 | 0.001 | 11.2132 |
| 0.5 | 0.01 | 11.2074 |
| 0.5 | 0.1 | pending |

At s=0.9, lambda=0.1 is mildly better than 0.001/0.01 (16.04 vs ~16.16,
~0.8% better) -- not perfectly monotonic (0.01 is slightly worse than
0.001, likely within run-to-run/optimization noise at this scale) but
the higher-lambda arm is the clear best of the three tested. At s=0.5,
both lambda values land within 0.05% of each other and within ~0.3% of
L0/dense -- confirms the "essentially neutral, not lambda-sensitive"
reading from the C1 recalibration holds across this sweep too, not just
at the one lambda originally tested. `l1c2_s0.5_g1_lam0.1` eval still
running -- expect it to land in the same tight neutral band.

**Practical implication for C2's own question**: at s=0.9, a slightly
higher lambda than the original default (0.01) may be worth adopting
going forward (0.1 beat both smaller values here) -- not conclusive from
3 points, but directionally suggestive. At s=0.5, lambda choice doesn't
matter within this range; the anchor (C1), not the regularization
strength, is what drove the correction.
