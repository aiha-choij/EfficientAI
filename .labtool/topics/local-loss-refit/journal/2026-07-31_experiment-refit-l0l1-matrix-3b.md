# Experiment: refit-l0l1-matrix-3b

Status: DONE (2026-07-31)
Date: 2026-07-31

## Hypothesis tested
Reduced-cost L0/L1 matrix (cost-reduction ladder rung 1) on the dev model:
does ΔL1 = PPL(L1) − PPL(L0) stay favorable across the full (s, g) grid, and
does it grow with g (does refit absorb more of the "sharing tax" as more
tokens are forced to share one mask)?

## What we're testing over alternatives
The single verification point (2026-07-31, prior journal card) only checked
one (s,g) = (0.9, 1). This job maps the full reduced grid: g ∈ {1, 32} at
all s ∈ {0.5, 0.7, 0.9}, g ∈ {8, 128} at s=0.9 only.

## Prior art check
Only prior local-loss-refit result: refit-l0l1-validate-3b (s=0.9, g=1,
GO, ΔL1=−1.635). No other journal history for this topic.

## Expected outcome
Per the request spec's own framing: ΔL1(g) should grow (more negative) as
g increases at fixed s=0.9, since larger blocks create a coarser/less
token-precise mask (more sharing tax) that a linear refit has more
systematic bias to correct. No specific expectation was set for s=0.5/0.7
beyond "matched-(s,g) comparison" — the single verification point only
exercised s=0.9.

## Reproducibility
- Git tag: none (branch auto/local-loss-refit, commit ec2403d — see
  RESULT_JSON `git_commit` in every result file)
- Job ID: 050-20260731-104517-refit-l0l1-matrix-3b
- Host/GPU: a100-40-2 (pinned), 1 GPU, 85 min elapsed
- Command: `env PY=.../envs/larosa/bin/python bash
  larosa/scripts/refit/run_matrix.sh /raid/LLM/llama3.2-3b-instruct
  /home/choij/workspace/refit/llama3.2-3b-instruct`
- Config: lambda=0.01 fixed, calibration wikitext103 (reused the exact
  calib_tokens.pt saved by the validation job — same 512x2048 tokens,
  seed=42), eval wikitext-2 (same pipeline as all prior anchors)
- Same model deviation as the validation job: `llama3.2-3b-instruct`
  (instruct-tuned) stands in for the spec's plain pretrained Llama-3.2-3B
  dev checkpoint, not present locally.

### Results
(artifact: job log at ~/workspace/runs/20260731-104517-refit-l0l1-matrix-3b/log;
result JSONs under ~/workspace/refit/llama3.2-3b-instruct/results/)

| s | g | L0 PPL | L1 PPL | ΔL1 = L1−L0 | ΔL1 (relative) |
|---|---|---|---|---|---|
| 0.5 | 1 | 11.2104 | 13.2622 | **+2.052** | +18.3% |
| 0.7 | 1 | 12.0017 | 14.2477 | **+2.246** | +18.7% |
| 0.9 | 1 | 21.5859 | 19.9510 | **−1.635** | −7.6% |
| 0.5 | 32 | 14.0646 | 17.7605 | **+3.696** | +26.3% |
| 0.7 | 32 | 21.1946 | 25.1633 | **+3.969** | +18.7% |
| 0.9 | 32 | 169.601 | 113.334 | **−56.267** | −33.2% |
| 0.9 | 8 | 75.438 | 49.685 | **−25.752** | −34.1% |
| 0.9 | 128 | 371.941 | 241.851 | **−130.090** | −35.0% |

Two clearly different regimes:
1. **s=0.9 (all g): L1 clearly beats L0, and the ABSOLUTE gap grows fast
   with g** (−1.6 at g=1 -> −130.1 at g=128), tracking the g-growth of L0
   itself (L0's own PPL explodes with g: 21.6 -> 371.9, the "sharing tax"
   made visible). L1 also grows with g but less steeply (19.95 -> 241.85).
   Refit absorbs roughly (350.35−221.90)/350.35 ≈ **37% of the ADDITIONAL
   sharing-tax PPL** incurred going from g=1 to g=128 at s=0.9. The
   RELATIVE ΔL1 is roughly flat once g>1 (−33% to −35%), vs only −7.6% at
   g=1 -- refit's relative value jumps once blocking is introduced at all,
   then holds roughly steady as blocks get coarser.
2. **s=0.5 and s=0.7 (g=1 and g=32): L1 is WORSE than L0, consistently**
   (+18% to +26% relative). This was NOT tested by the single verification
   point (which only covered s=0.9) and is a genuine surprise.

## Interpretation
(provisional — dev model only, `Llama-3.1-8B` matrix job
(050-20260731-105247) still running; do not treat as final until the main
model confirms or refutes the low-s regression)

**Leading hypothesis for the s=0.5/0.7 regression**: bias-variance
tradeoff. At low sparsity the mask removes little, so L0 (original weight,
frozen mask) is already close to optimal -- there is very little true
systematic bias for a linear refit to correct. The closed-form fit still
strictly reduces IN-SAMPLE (calibration) loss by construction (per unit
test 3), but with little real bias to remove, what it mostly captures is
calibration-corpus-specific noise, which does not transfer to the
wikitext-2 eval set (note: calibration corpus is wikitext103, NOT
wikitext-2 -- a real, if related, distribution shift). At s=0.9 the
systematic masking bias is large enough that this correction dominates the
variance cost, so refit nets a clear win; at s=0.5/0.7 it doesn't.
Alternative/confounding explanations not yet ruled out: (a) the
calibration/eval corpus mismatch (wikitext103 vs wikitext-2) specifically,
independent of sparsity; (b) something particular to the instruct-tuned
dev checkpoint. Neither is ruled in either.

**This does NOT change the request's Go decision** (which was scoped to
s=0.9, g=1 and passed clearly) but it substantially changes the picture:
refit's benefit looks confined to the high-sparsity regime, not universal
across s. Report as observed, per the request's own instruction to report
null/negative results plainly.

## Next
- Do not resolve the s=0.5/0.7 question on the dev/instruct model alone --
  wait for `Llama-3.1-8B` (base, matched calibration setup) matrix results.
- If the regression replicates on the base model too: consider a same-
  corpus calibration/eval ablation (e.g. calibrate AND eval on wikitext-2
  splits) to isolate the corpus-mismatch hypothesis from the bias-variance
  one -- flag for user decision, not in scope to add unprompted.
- Proceed to L2 implementation regardless (per plan) -- L2's sequential
  refit uses a fresh mask recomputed against the sparse stream, which
  could plausibly behave differently at low s.
