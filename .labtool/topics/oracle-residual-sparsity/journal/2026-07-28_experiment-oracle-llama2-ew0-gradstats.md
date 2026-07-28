# Experiment: oracle-llama2-ew0-gradstats

Status: PENDING
Date: 2026-07-28

## Hypothesis tested
Loss-aligned (output-side) sensitivity — the direction the whitening round
prescribed and the critique (2026-07-28) elevated: w = sqrt(E[(dL/dy)^2]) at
the MLP block output, estimated by calibration backprop, yields an offline
error metric E‖w ⊙ (comp(x) − Mx)‖ that ranks factor variants in their
MEASURED PPL order — in particular it must fix the whitening inversion that
plain input-L2 provably gets backwards (whitened had 13% LOWER L2 yet HIGHER
PPL). If validated, w becomes the basis for weighted SVD factors (E-W1) and
weighted selection scores (E-W2), and a reusable screening metric for the
cheap-budget round (E3) and per-layer allocation (E4).

## What we're testing over alternatives
Metric-validation-first: we have 11 factor variants with known PPL@s=0.9
(plain512/1024/2048, wht512/1024, alloc256/512/1024, three E1 slr splits) —
a free test set no new PPL run can improve on. If the weighted metric fails
here, the whole W-direction is falsified for a few GPU-minutes; if it
passes, every later round inherits a validated screen. Also bundled (from
critique Critical 1&2): TIS frontier fill-in c1 @ s={0.75, 0.8, 0.85} —
locates the TIS-vs-SLR crossover and starts the iso-compute frontier.

## Prior art check
- Whitening card (2026-07-24): "any future compensation objective should
  weight the OUTPUT side (downstream sensitivity), not the input
  distribution" — deferred since; this is the first follow-up. Known
  numbers the metric must reproduce/fix: plain trunc_err 2.239 vs whitened
  1.941 (inversion vs PPL 7.2294/7.3974 at r1024, 8.7638/9.7606 at r512).
- E0/E1 (2026-07-27/28): offline input-L2 picked pure-sparse; PPL picked
  the mixed split — second known misranking (soft check; slr PPLs within
  0.05 so rank noise possible).
- Dead Ends checked: input whitening (dead), spectral alloc (dead), S1
  (dead) — none is this; this is the output-side mirror of whitening.
- H2 rejection (Phase 4): score-only changes did nothing under zero-fill —
  tempers expectations for the E-W2 selection-score arm, noted in critique.

## Expected outcome
- Positive control: our plain metric must reproduce the known inversion
  (wht error BELOW plain) — if not, the harness is broken; do not interpret.
- GATE (E-W1/W2 go): weighted metric puts wht512 > plain512 AND wht1024 >
  plain1024 in error (matching PPL). Spearman(weighted, PPL) should
  clearly exceed Spearman(plain, PPL) across the 11 variants.
- Stability: per-layer corr of log w between c4 and wikitext103 grads —
  if early layers are weak (like ḡ was, 0.48–0.53), restrict W-methods to
  mid/late layers.
- TIS frontier: c1 PPL at s=0.75/0.8/0.85 — expect the crossover vs SLR
  B=1024 (6.9417 @ s=0.9) to appear in this band; these three numbers also
  seed the iso-compute comparison from critique Critical 2.
- Failure: weighted metric does not fix the inversion → this w (diagonal
  second moment) is insufficient; report and decide with user whether to
  try full gradient covariance or drop the direction.

## Reproducibility
- Git tag: exp/2026-07-28_oracle-llama2-ew0-gradstats (ab6bcc2; local =
  github = gateway = a6000-2 all synced)
- Job ID: 050-20260729-062542-oracle-llama2-ew0-gradstats
- Assigned host/GPU: a6000-2 (pinned via -H), 1 GPU >=30GiB [pending dispatch]
- Command: `bash -c "export PY=/home/choij/workspace/venv-larosa/bin/python;
  scripts/oracle/run_ew0.sh /raid/LLM/llama2-7b
  /home/choij/workspace/oracle/llama2-7b"`
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a6000-2)
- Config path: n/a — parameters in run_ew0.sh
- Key parameters: grad stats — frozen params, graph rooted at embeddings,
  E[(dL/dy)^2] at down_proj output, fp64 CPU accumulation; c4 arm reuses
  stats/c4/calib_tokens.pt (128 seqs x 2048), wt103 arm builds its own
  (seed 42) -> grad_stats/{c4,wt103}. Metric check — 11 variants (dirs
  r512, plain_r1024, plain_r2048, wht_r512, wht_r1024, wht_alloc{256,512,
  1024}, slr_r512_k1024, slr_r256_k1536, slr_r0_k2048), PPLs passed on CLI
  from prior cards, 8 capture seqs / 16384 tokens per layer, w floor
  1e-4*mean -> results/metric_check.json. TIS — c1 topk s {0.75,0.8,0.85}
  -> results/c1_topk_s*.json.
- Key deps: torch 2.6.0+cu124, transformers 4.46.3, venv
  /home/choij/workspace/venv-larosa, sdpa, RTX A6000
- Estimated cost/time: 1 GPU, ~1-1.5 h (2 grad passes ~10-20 min, metric
  check ~20 min, 3 PPL runs ~10 min)
- Tiny-model smoke: 10 and 11 run end-to-end on CPU/MPS fixture (fp64-on-
  MPS pitfall fixed by cpu-first accumulation).

### Results

### Interpretation
