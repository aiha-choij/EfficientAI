# Experiment: refit-harness-*-8b (lm-eval zero-shot, L0/L1 x s in {0.9,0.5})

Status: IN PROGRESS (2026-07-31)
Date: 2026-07-31

## Hypothesis tested
Does the PPL-based finding (L1 refit clearly helps at s=0.9, clearly hurts
at s=0.5) hold on the accuracy axis the request spec's own judgment
criteria are framed around (normalized zero-shot accuracy, 8-task suite,
--limit 1000)?

## Prior art check
- refit-l0l1-matrix-8b: PPL anchors, s=0.9 g=1: L0=12.9347, L1=10.3305
  (ΔL1 -20.1%); s=0.5 g=1: L0=6.3869, L1=7.2615 (ΔL1 +13.7%, hurts).
- refit-lmeval-smoke-3b / smoke2-3b / check-archallenge (same day): HFLM
  wrapping of our already-configured model works correctly; `piqa` alone
  fails on this gateway env (datasets==5.0.0 dropped legacy loading-script
  support that lm_eval 0.4.3's piqa task definition relies on) -- suite
  scoped to 7 tasks (arc_easy, arc_challenge, boolq, hellaswag, winogrande,
  sciq, lambada_openai), documented deviation from the request's 8-task
  list.

## Reproducibility
- Git tag: none (branch auto/local-loss-refit, commit 053a814 at submit)
- Jobs: 050-20260731-155329 (L0 s=0.9), 050-20260731-155334 (L1 s=0.9),
  050-20260731-155342 (L0 s=0.5), 050-20260731-155351 (L1 s=0.5) -- all
  Llama-3.1-8B, g=1, a100-40-2, 7 tasks, --limit 1000, seed=42.
- L1 weights reused as-is from refit-l0l1-matrix-8b (no rebuild):
  weights/l1_s{0.9,0.5}_g1_lam0.01.
- Script: scripts/refit/03_lm_eval.py (HFLM wraps the already-masked/
  refit-loaded model instance directly, not a fresh reload).

### Results so far
(artifact: ~/workspace/runs/20260731-155334-refit-harness-l1-s09-8b/log;
result JSON: ~/workspace/refit/Llama-3.1-8B/results/harness_l1_s0.9_g1.json)

**L1, s=0.9, g=1** (achieved sparsity 0.9000):

| task | acc | acc_norm |
|---|---|---|
| arc_easy | 0.698 | 0.670 |
| arc_challenge | 0.365 | 0.403 |
| boolq | 0.751 | — |
| hellaswag | 0.457 | 0.591 |
| winogrande | 0.642 | — |
| sciq | 0.932 | 0.897 |
| lambada_openai | 0.599 (ppl 6.841) | — |

L0 s=0.9 (the comparison point for the s=0.9 Go check), L0 s=0.5, and L1
s=0.5 are still running -- do not compute or interpret deltas from this
one condition alone. Two harmless log lines each run ("Repo id must be a
string..." from HFLM's HF-Hub metadata lookup on an already-loaded model
instance; "fatal: not a git repository" from lm_eval's own provenance
logging) do not affect results, confirmed during the smoke tests.

## Next
- Await the remaining 3 jobs, then build the full 2x2 (L0/L1 x s=0.9/0.5)
  table and check whether the accuracy-axis pattern matches PPL's
  (s=0.9 helps, s=0.5 hurts). Update this card's Interpretation once all 4
  are in -- do not interpret from a single condition.
