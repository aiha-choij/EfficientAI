# Experiment: oracle-llama2-e1-s2-ppl

Status: PENDING
Date: 2026-07-27

## Hypothesis tested
H4/S2 on the PPL axis: the input-channel sparse + low-rank hybrid
compensation (comp = B(Ax) + R(m_x ⊙ x), R = M − BA, m_x = top-k |x|)
beats plain rank-r compensation at matched MAC budget B_eff = 1024 in
wikitext-2 PPL — i.e., the E0 screening win survives the cross-family
metric caveat (whitening precedent: lower input-L2 does NOT guarantee
lower PPL).

## What we're testing over alternatives
The three E0-surviving arms only (user-approved 9 runs; weakest split
r768:k512 dropped on the monotone trend; S1/slr_neuron dead per E0; wnorm
score dropped, abs ≈ wnorm). Same eval pipeline as every prior C4 row, so
deltas ≥0.01 PPL are signal (phase-3 gate).

## Prior art check
- E0 card (2026-07-27, same topic): S2 arms' matched-budget rel-err
  0.253/0.225/0.198 (mid-stack) vs lr_r1024 0.330; monotone toward pure
  sparse. S1 refuted. THIS is the direct gate for these arms.
- C4 whitening round (2026-07-24): plain r1024 PPL anchors
  5.7365/5.9152/7.2294 (s=0.5/0.7/0.9); cross-family input-L2 inversion
  precedent (wht1024 lower L2, worse PPL) — the risk this run adjudicates.
- Phase 4 (2026-07-24): C1 5.5216/5.7284/8.1096; C3 5.5051/5.6283/6.6381;
  dense 5.4738.
- No Dead End conflicts (S1 recorded dead 2026-07-27; not run here).

## Expected outcome
- Success (gate, pre-agreed with user at E0 close): any arm beats plain
  lr_r1024 by ≥ 0.05 PPL at s=0.9 (i.e. ≤ 7.179) → proceed to B_eff=2048
  round (subsumes the old r=2048 reference arm) + per-layer rho search.
  Stretch: close the gap to C3 (6.638 @ s=0.9) meaningfully.
- Informative orderings: does PPL preserve E0's monotone sparse-heavy
  ordering (r0:k2048 best), or invert like whitening did? Inversion =
  direction-bias mechanism (S2 concentrates error on low-|x| channels,
  analogous to input-magnitude weighting).
- Failure: all arms ≥ lr_r1024 − 0.05 @ s=0.9 → input-side approximation
  family (whitening + S2) recorded dead; fall back to quantized-M line.

## Reproducibility
- Git tag: exp/2026-07-27_oracle-llama2-e1-s2-ppl (4efbb4e; local = github
  = gateway = a6000-2 clone all synced)
- Job ID: 050-20260728-155727-oracle-llama2-e1-s2-ppl
- Assigned host/GPU: a6000-2 (pinned via -H), 1 GPU >=30GiB [pending dispatch]
- Command: `bash -c "export PY=/home/choij/workspace/venv-larosa/bin/python;
  scripts/oracle/run_e1_s2.sh /raid/LLM/llama2-7b
  /home/choij/workspace/oracle/llama2-7b"`
- Workdir: /home/choij/workspace/repos/EfficientAI/larosa (a6000-2)
- Config path: n/a — parameters in run_e1_s2.sh
- Key parameters: condition c4, comp_mode=slr_input, x_score=abs, select
  topk; arms (r:k) {512:1024, 256:1536, 0:2048} = B_eff 1024; s grid
  {0.5, 0.7, 0.9}; stats_dir stats/c4 (unchanged calibration); factors to
  factors/slr_r{R}_k{K}; results to results/c4_slr_r{R}_k{K}_topk_s{S}.json.
  Pipeline: 03_build_M x3 (spectra pass skipped — new conditional) →
  04_eval_ppl x9 (wikitext-2, eval_ppl_wikitext_with_inference_sparsity).
- Key deps: torch 2.6.0+cu124, transformers 4.46.3, venv
  /home/choij/workspace/venv-larosa, sdpa backend, RTX A6000
- Estimated cost/time: 1 GPU, ~2.5-4 h (3 factor builds + 9 PPL evals)

### Results

### Interpretation
