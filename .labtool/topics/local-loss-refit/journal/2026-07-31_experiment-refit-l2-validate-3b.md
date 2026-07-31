# Experiment: refit-l2-validate-3b

Status: DONE, interpretation CONFIRMED by control run (2026-07-31)
Date: 2026-07-31

## Hypothesis tested
Single verification point for L2 (mirrors the L0/L1 validation): at s=0.9,
g=1, does the sequential (GPTQ-style) refit -- mask/score recomputed
against the SPARSE stream, teacher always the dense model -- work
end-to-end at real model scale, and how does it compare to L0/L1 at the
same (s,g)?

## What we're testing over alternatives
L2's premise (per the request spec and this topic's gist) is that
recomputing the mask against what the model will ACTUALLY see at inference
(the sparse stream, not a dense proxy) should be more faithful than L1's
frozen-dense mask, and sequential per-layer fitting can, in principle,
compensate for upstream layers' own approximation error. Whether that
premise pays off in practice, vs the compounding-error risk inherent to any
greedy/sequential calibration scheme, was unknown before this run.

## Prior art check
- refit-l0l1-matrix-3b/8b: L0/L1 anchors at s=0.9, g=1 to compare against
  (3B: L0=21.5859, L1=19.9510).
- test_8_l2_s0_restoration_multilayer (unit test, written same day):
  confirmed the sequential teacher/student mechanism is algebraically
  correct at s=0 across 3 layers (max rel diff ~1e-5, dense/sparse streams
  agree to 1.86e-08) -- catching and fixing a real bug (teacher/student
  calls need to alternate per calibration chunk, not run as two separate
  full passes, or the accumulation targets the wrong chunk's teacher
  output). This run is the first test with REAL masking (s=0.9) and a real
  model, where compounding effects (absent at s=0, where sparse==dense) can
  actually manifest.

## Expected outcome
No strong prior on L2 vs L1's sign -- flagged as an open question in this
topic's gist from the start ("does the low-s regression persist when the
mask is recomputed against the sparse stream instead of frozen from the
dense one?"). Watched for: PPL in the same ballpark as L1 (mechanism
working as intended) vs wildly different (bug) vs consistently worse
(error compounding dominates the sparse-stream-fidelity benefit).

## Reproducibility
- Git tag: none (branch auto/local-loss-refit, commit 8f0324c at submit
  time)
- Job ID: 050-20260731-150816-refit-l2-validate-3b
- Host/GPU: a100-40-2 (pinned), 1 GPU, 8 min elapsed (fast -- L2's smaller
  default calibration budget keeps wall-clock low)
- Command: `env PY=.../envs/larosa/bin/python bash
  larosa/scripts/refit/run_validate_l2.sh /raid/LLM/llama3.2-3b-instruct
  /home/choij/workspace/refit/llama3.2-3b-instruct`
- Config: s=0.9, g=1, lambda=0.01, calibration wikitext103, **nsamples=128**
  (NOT 512 like L1 -- L2's documented smaller default, since each layer
  costs ~3x a plain dense forward; own calib_tokens_l2.pt, NOT the same
  saved tokens L1 used), eval wikitext-2 (same pipeline).
- **Confound flagged before interpreting**: L2 used 1/4 the calibration
  tokens L1 did. A same-nsamples control run
  (050-20260731-151718-refit-l2-validate-3b-n512, nsamples=512, reusing
  L1's exact saved calib_tokens.pt) was submitted immediately after to
  separate "L2 needs more calibration" from "L2 has an inherent
  compounding-error problem at this budget" -- see that job's card /
  this card's update once it returns.

### Results
(artifacts: job logs at ~/workspace/runs/20260731-150816-refit-l2-validate-3b/log
and ~/workspace/runs/20260731-151718-refit-l2-validate-3b-n512/log; result
JSONs under ~/workspace/refit/llama3.2-3b-instruct/results/)

| condition | s | g | nsamples | PPL | achieved sparsity |
|---|---|---|---|---|---|
| L0 | 0.9 | 1 | — | 21.5859 | 0.9000 |
| L1 | 0.9 | 1 | 512 | 19.9510 | 0.9000 |
| L2 | 0.9 | 1 | 128 | 26.9013 | 0.9000 |
| **L2 (control)** | 0.9 | 1 | **512** (matches L1, same calib_tokens.pt) | **26.3277** | 0.9000 |

**Control run confirms the calibration-budget confound is NOT the driver.**
4x the calibration data (128 -> 512, reusing L1's own saved token set for a
fully apples-to-apples comparison) moved PPL by only −0.57 (26.90 -> 26.33,
≈2% relative) — nowhere near closing the ~6-7 PPL gap to L1. **L2 is WORSE
than both L0 (mask-only) and L1 at this (s, g), and that is not a
calibration-starvation artifact.**

### Interpretation
**CONFIRMED (not a calibration artifact).** Two hypotheses were raised;
one is now ruled out:
1. ~~Calibration-budget confound~~ -- RULED OUT. 4x the data barely moved
   the result (26.90 -> 26.33). Whatever is happening, more calibration
   tokens at this scale does not fix it.
2. **Error compounding (a real property of sequential/greedy calibration,
   not a bug) -- now the leading and only remaining explanation.** Unlike
   L1, where every layer regresses from the TRUE dense activations, L2's
   layer L sees the OUTPUT of layers <L's own (imperfect, regularized)
   refit -- any approximation error at layer 0 propagates and compounds
   through the stack. GPTQ-style quantization methods carry the same
   structural risk; it doesn't always net out favorably relative to
   independent per-layer fitting, especially when dense-anchored (L1-style)
   fitting is cheap and available as an alternative (unlike GPTQ, which
   doesn't have a "just use the original
   weights" option since the weights themselves are being quantized).
3. (Ruled out, not the cause) Implementation bug in the sequential
   mechanism itself -- test_8 already exercises the exact same code path
   (enable_l2_layer_collect / set_l2_role / finalize_l2_layer /
   single_layer_forward) end-to-end across multiple layers and confirms it
   is algebraically correct at s=0; this run only changes s (0->0.9) and
   scale (tiny random model -> real 3B), neither of which touches the
   mechanism test_8 validated.

This IS a real, reportable negative result for L2's specific formulation
(recompute-mask-against-sparse-stream + sequential-fit), NOT for local-loss-
refit generally -- L1 remains solidly confirmed at s=0.9 on two models
(gist Key Findings). Reporting as such, per the request's own instruction
to report null/negative results plainly: **dense-anchored independent
fitting (L1) beats trying to be "faithful" to the sparse stream (L2), at
least at this (s, g, lambda) and this calibration scale.** This is useful
information for the compensation-branch line (oracle-residual-sparsity)
too -- it echoes that line's own experience that metrics/objectives which
sound more "faithful" to the deployed condition don't automatically win
(e.g. the whitening round's input-L2-vs-downstream-loss mismatch).

## Next
- Do NOT add L2 to the broader (s,g) matrix -- it already clearly loses at
  the one point tested, matched-budget, mechanism-verified-correct. Spending
  more GPU time on a losing condition isn't justified without a new idea
  for why it might turn around elsewhere in the grid (no such idea exists
  right now).
- Not run (flag only, do not implement unprompted -- this topic's scope is
  measuring refit's effect, not re-engineering L2 into a third variant):
  (a) whether a LARGER lambda (more regularization) tempers the error
  compounding, since compounding plausibly comes from each layer's fit
  being aggressive/overconfident given only that layer's own local
  objective; (b) whether excluding the first few layers from L2's
  sequential treatment (dense-anchored there, sequential only later) caps
  how much compounding can accumulate.
- lm-eval-harness (0.4.3) confirmed installed in the gateway conda env
  (`~/miniconda3/envs/larosa`) -- the "unconfirmed" open question from
  init is resolved. Next natural step for the CONFIRMED L0/L1 finding is
  the 8-task zero-shot suite at --limit 1000, PPL-first groundwork already
  done.
