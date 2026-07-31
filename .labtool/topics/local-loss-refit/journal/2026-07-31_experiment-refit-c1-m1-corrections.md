# Experiment: refit-c1-m1-corrections (section 4 honesty corrections: C1 ridge anchor, M1 log-PPL + dense anchor)

Status: PENDING (code done, GPU jobs in flight)
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
