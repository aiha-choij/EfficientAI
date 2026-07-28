# Experiment: oracle-llama2-e0-slr-diag

Status: DONE (2026-07-27)
Date: 2026-07-27

## Hypothesis tested
H4 screening (E0, offline — no PPL): the SLR hybrid premises are measurable
before any eval job. (a) S1 gate: removing the top-hot_n static rank-1 neuron
terms (c_j = ḡ_j‖W_d[:,j]‖‖W_u[j,:]‖) leaves an M_cold that is meaningfully
lower-rank than M mid-stack. (b) S2 gate: MLP-input x has concentrated
channel energy, and spending MACs on exact top-|x| channels of the SVD
residual R = M − BA reduces compensation error more than spending the same
MACs on additional rank.

## What we're testing over alternatives
Both H4 variants are gated offline for the cost of one short single-GPU job,
instead of paying ~6 PPL sweeps blind. The matched-budget accounting
(rank 1 ≡ 1 hot neuron ≡ 2 input channels; B_eff = 1024 ≡ plain r=1024,
+6.2% compute) makes every arm directly comparable to the best known C4.

## Prior art check
- Steer card 2026-07-27_pivot-c4-slr-compensation.md: direction rationale;
  only direct journal hit for sparse+low-rank / hot-set.
- C4 whitening round (2026-07-24): anchors (plain r1024 = 5.737/5.915/7.229);
  METRIC CAVEAT — input-L2 rel-err ranks within a geometry but inverted
  across geometries (whitened lower L2, worse PPL). This job's rel-err is
  screening only; PPL (E1) is the referee.
- Phase-0 (2026-07-22): M r90 ≈ 1270 mean (1400–1500 mid-stack); energy@512
  0.54–0.60 mid-stack — the deficit S1 tries to remove.
- rsparse-repro gist: searched alpha mean 0.95 (sparse-heavy wins; pure
  low-rank fails) — motivates sparse-leaning splits.
- No Dead Ends conflicts (whitening/alloc are ON HOLD, untouched here).

## Expected outcome
- S1 go-signal: M_cold mid-stack energy@1024 clearly above plain M
  (≳ +0.1 absolute) or mean r90 drops materially with hot_n ≤ 1376; split
  arms s1_r*_h* beat lr_r1024 on mid-stack mean rel-err.
- S2 go-signal: top-2048-channel energy share high per token AND s2 arms
  beat lr_r1024 mid-stack rel-err (abs vs wnorm score ordering recorded).
- Decision: best-of-family arms that beat lr_r1024 proceed to E1 PPL sweep;
  if NO arm beats the lr baseline, H4 is weakened at B_eff=1024 — reassess
  with user (raise budget vs switch to quantized-M line) before any PPL job.
  Given the metric caveat, a near-tie does NOT kill an arm by itself.
- Failure mode to watch: x-channel concentration high but s2 error not
  better → M's heavy tail lives off the outlier channels (premise (b) false).

## Reproducibility
- Git tag: exp/2026-07-27_oracle-llama2-e0-slr-diag (d827aaf; local = github
  = gateway = a6000-2 clone all synced to this commit)
- Job ID: 050-20260728-084743-oracle-llama2-e0-slr-diag
- Assigned host/GPU: a6000-2 (pinned via -H), 1 GPU >=30GiB [pending dispatch]
- Command: `bash -c "/home/choij/workspace/venv-larosa/bin/python
  scripts/oracle/09_slr_diag.py --model_name /raid/LLM/llama2-7b
  --stats_dir /home/choij/workspace/oracle/llama2-7b/stats/c4
  --nsamples 8 --capture_tokens 16384 --budget 1024
  --out_json /home/choij/workspace/oracle/llama2-7b/results/slr_diag_b1024.json"`
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a6000-2)
- Config path: n/a — parameters as script args
- Key parameters: budget B_eff=1024; hot_grid {0,256,512,768,1024,1376};
  s1_splits (r:hot) {768:256, 512:512, 256:768, 0:1024}; s2_splits (r:k)
  {768:512, 512:1024, 256:1536, 0:2048}; x scores abs AND wnorm both
  measured; conc_ks {41,256,512,1024,1536,2048}; energy_ranks
  {256,512,1024,2048}; calibration = stats/c4 calib_tokens (8 seqs x 2048,
  16384 captured tokens/layer, fp16 CPU buffer).
- Parameter deviations from gist E0 sketch: hot grid merged with the D2
  split values + d/8 top end (gist sketch said {344,688,1376}); capture
  16k tokens not ~64k (CPU-memory bound; mean rel-err statistics converge
  well below that).
- Key deps: torch 2.6.0+cu124, transformers 4.46.3, venv
  /home/choij/workspace/venv-larosa, sdpa backend (no flash-attn), RTX A6000
- Unit tests at tag: test_oracle_units.py all pass incl. new test_7_slr
  (slr full-budget ≡ C3, degenerate ≡ lr, save/load bitwise); 09 script
  end-to-end smoke on tiny CPU model OK.
- Note: a6000-2 had an uncommitted edit (larosa/scripts/analyze_topk_overlap.py)
  preserved as stash@{0} `pre-slr-sync-20260727` before fast-forward.

### Results
(artifacts: job log summary of 050-20260728-084743 + slr_diag_b1024.json on
a6000-2 ~/workspace/oracle/llama2-7b/results/; JSON git_commit d827aaf,
16384 tokens/layer; job elapsed 17 min)

D1 — hot-set removal makes M HARDER to approximate (S1 premise inverted):

| hot_n | mean r90 | max r90 | mean energy@2048 |
|-------|----------|---------|------------------|
| 0     | 1270.1   | 1524    | 0.971 |
| 256   | 1437.5   | 1518    | 0.965 |
| 512   | 1461.2   | 1552    | 0.963 |
| 768   | 1469.5   | 1568    | 0.963 |
| 1024  | 1471.5   | 1575    | 0.963 |
| 1376  | 1469.1   | 1577    | 0.963 |

D2 — matched-budget rel err E‖Mx−comp‖/E‖Mx‖ (mean / mid-stack 4–17 / worst):

| arm | mean | mid-stack | worst |
|-----|------|-----------|-------|
| lr_r1024 (baseline)  | 0.2701 | 0.3302 | 0.3545 (L7) |
| s1_r768_h256         | 0.3063 | 0.3728 | 0.3992 (L7) |
| s1_r512_h512         | 0.3504 | 0.4261 | 0.4569 (L7) |
| s1_r256_h768         | 0.4084 | 0.4974 | 0.5329 (L7) |
| s1_r0_h1024          | 0.5921 | 0.7152 | 0.7613 (L7) |
| s2_r768_k512_abs     | 0.2306 | 0.2798 | 0.3013 (L18) |
| s2_r512_k1024_abs    | 0.2096 | 0.2532 | 0.2720 (L18) |
| s2_r256_k1536_abs    | 0.1877 | 0.2248 | 0.2413 (L18) |
| s2_r0_k2048_abs      | 0.1836 | 0.1975 | 0.2205 (L18) |

- wnorm ≈ abs everywhere (Δ ≤ 0.0006, abs equal or marginally better).
- Every S1 arm loses to lr_r1024; monotonically worse with hot_n.
- Every S2 arm beats lr_r1024; monotonically better as the split gets
  sparse-heavier; best arm s2_r0_k2048 cuts mid-stack error −40%
  (0.330 → 0.198).
- x channel-energy concentration (per token, mean over layers): top-41
  (1%) 0.136, top-512 0.536, top-1024 0.743, top-2048 0.934 — moderate,
  no extreme outlier regime.

### Interpretation
(user, 2026-07-27 — adopted the proposed reading "S1 폐기, S2 진출")

- S1 (neuron hot set) is a DEAD END: its premise inverted in measurement —
  removing the top static rank-1 terms makes M_cold HARDER to approximate
  (mean r90 1270 → ~1470), i.e. the hot terms are aligned with M's low-rank
  structure, so trading rank for them wastes budget; every S1 arm lost to
  plain lr_r1024, monotonically in hot_n.
- S2 proceeds to the E1 PPL sweep: 3 arms r512:k1024 / r256:k1536 /
  r0:k2048, abs score (wnorm ≈ abs, dropped), s = {0.5, 0.7, 0.9} — 9 runs,
  anchor lr_r1024 (7.229 @ s=0.9). The weakest split r768:k512 is omitted
  (monotone trend).
- Caveat carried forward: this is an input-L2 screening win; the whitening
  precedent shows cross-family orderings can invert in PPL. E1 is the
  referee.
