# Experiment: refit-harness-*-8b (lm-eval zero-shot, L0/L1 x s in {0.9,0.5})

Status: DONE, hypothesis CONFIRMED (2026-07-31)
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

### Results: s=0.9 (both conditions in)
(artifacts: ~/workspace/runs/20260731-15532[9|34]-refit-harness-l[01]-s09-8b/log;
result JSONs: ~/workspace/refit/Llama-3.1-8B/results/harness_l[01]_s0.9_g1.json)

| task | L0 acc | L1 acc | Δacc | L0 acc_norm | L1 acc_norm | Δacc_norm |
|---|---|---|---|---|---|---|
| arc_easy | 0.638 | 0.698 | **+6.0%p** | 0.612 | 0.670 | +5.8%p |
| arc_challenge | 0.377 | 0.365 | -1.2%p | 0.386 | 0.403 | +1.7%p |
| boolq | 0.626 | 0.751 | **+12.5%p** | — | — | — |
| hellaswag | 0.462 | 0.457 | -0.5%p | 0.612 | 0.591 | -2.1%p |
| winogrande | 0.609 | 0.642 | +3.3%p | — | — | — |
| sciq | 0.895 | 0.932 | +3.7%p | 0.853 | 0.897 | +4.4%p |
| lambada_openai | 0.559 | 0.599 | +4.0%p | (ppl 10.56 -> 6.84, also improved) | | |

**L1 beats L0 on 5/7 tasks by acc, ties/slightly-worse on 2 (arc_challenge
-1.2%p, hellaswag -0.5%p acc / -2.1%p acc_norm)**; average Δacc across the
7 tasks = **+3.97%p**, boolq the standout (+12.5%p). This CONFIRMS the
PPL-based s=0.9 finding on the accuracy axis too, using the request spec's
own primary judgment metric (normalized zero-shot accuracy) rather than
the secondary PPL signal -- the Go decision now has independent support
from both metrics, on both models (3B PPL, 8B PPL, 8B accuracy).

Two harmless log lines each run ("Repo id must be a string..." from
HFLM's HF-Hub metadata lookup on an already-loaded model instance; "fatal:
not a git repository" from lm_eval's own provenance logging) do not affect
results, confirmed during the smoke tests and again here.

### Results: s=0.5 (contrast, both conditions in)
(artifacts: ~/workspace/runs/20260731-1553[42|51]-refit-harness-l[01]-s05-8b/log;
result JSONs: ~/workspace/refit/Llama-3.1-8B/results/harness_l[01]_s0.5_g1.json)

| task | L0 acc | L1 acc | Δacc | L0 acc_norm | L1 acc_norm | Δacc_norm |
|---|---|---|---|---|---|---|
| arc_easy | 0.809 | 0.795 | -1.4%p | 0.795 | 0.771 | -2.4%p |
| arc_challenge | 0.509 | 0.481 | -2.8%p | 0.527 | 0.500 | -2.7%p |
| boolq | 0.804 | 0.820 | +1.6%p | — | — | — |
| hellaswag | 0.518 | 0.491 | -2.7%p | 0.683 | 0.660 | -2.3%p |
| winogrande | 0.728 | 0.717 | -1.1%p | — | — | — |
| sciq | 0.962 | 0.960 | -0.2%p | 0.942 | 0.941 | -0.1%p |
| lambada_openai | 0.759 | 0.742 | -1.7%p | (ppl 3.18 -> 3.24, also worse) | | |

**L1 loses to L0 on 6/7 tasks at s=0.5**, average Δacc = **-1.19%p**
(only boolq improves, +1.6%p -- the same task that showed the single
biggest s=0.9 gain, +12.5%p, an interesting asymmetry not investigated
further here). This CONFIRMS the PPL-based s=0.5 finding (+13.7% relative,
hurts) on the accuracy axis too.

## Interpretation
**Both halves of the request's core question are now answered with two
independent metrics, cross-validated:**
- s=0.9, g=1: L1 clearly beats L0 on BOTH PPL (-20.1% relative) and
  zero-shot accuracy (+3.97%p average, 5/7 tasks) -- **Go**, per the
  request's own stated example threshold ("accuracy +3%p"), met almost
  exactly on the nose.
- s=0.5, g=1: L1 clearly loses to L0 on BOTH PPL (+13.7% relative) and
  zero-shot accuracy (-1.19%p average, 6/7 tasks) -- confirms refit is not
  a universal fix, consistent with the bias-variance framing from the PPL
  matrix cards (little true bias to correct at low s -> the fit mostly
  captures calibration noise that doesn't transfer).

This closes the loop the PPL-only matrix results (3B + 8B, 8 (s,g) points
each) could not close alone: the request spec's PRIMARY judgment metric is
accuracy, with PPL as a secondary/continuous signal, and both metrics now
tell the same story on the main model. No further harness runs are
strictly required to answer the request's stated Go/No-go question --
remaining extensions (full g-sweep on the harness, a true dense s=0
baseline for framing, matching piqa via a different lm_eval version) are
backlog items, not blockers.
