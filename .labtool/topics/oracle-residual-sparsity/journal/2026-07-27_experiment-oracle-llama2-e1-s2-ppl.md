# Experiment: oracle-llama2-e1-s2-ppl

Status: DONE (2026-07-28)
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
(artifacts: 9 result JSONs c4_slr_r*_topk_s*.json under a6000-2
~/workspace/oracle/llama2-7b/results/, all git_commit 4efbb4e, achieved
sparsity 0.5000/0.7000/0.9001 as targeted; job log tail "E1 S2 sweep
complete"; anchors from phase-4 / whitening cards)

wikitext-2 PPL (dense 5.4738):

| arm (B_eff=1024)   | s=0.5  | s=0.7  | s=0.9  |
|--------------------|--------|--------|--------|
| C1 (no comp)       | 5.5216 | 5.7284 | 8.1096 |
| C3 (exact, upper)  | 5.5051 | 5.6283 | 6.6381 |
| c4 lr_r1024 anchor | 5.7365 | 5.9152 | 7.2294 |
| s2 r512:k1024      | 5.6200 | 5.7844 | 6.9962 |
| s2 r256:k1536      | **5.5961** | **5.7526** | **6.9417** |
| s2 r0:k2048        | 5.6261 | 5.7805 | 6.9526 |

- GATE PASSED by all three arms: best r256:k1536 @ s=0.9 = 6.9417, beats
  lr_r1024 by 0.288 (gate required ≥ 0.05, i.e. ≤ 7.179).
- Gap to C3 @ s=0.9 shrinks 7.2294−6.6381=+0.591 (lr) → +0.304 (s2) —
  roughly halved at identical MAC budget (+6.2% compute).
- PPL optimum is the MIXED split r256:k1536, not pure sparse: E0's monotone
  sparse-heavy ordering does NOT fully transfer (E0 predicted r0:k2048
  best; PPL says r256:k1536 < r0:k2048 < r512:k1024). Mild, benign
  instance of the input-L2 caveat — cross-family transfer held (all arms
  beat LR), within-family fine ordering did not.
- At s=0.5, s2 arms (5.596–5.626) still trail plain C1 (5.5216) — the
  compensation trade only pays at high sparsity, consistent with the C3
  pattern.

### Interpretation
(user pre-approval 2026-07-27 "E1와 후속 다 진행해줘" — pre-agreed gate
applied; flagged for user review: the mixed-split optimum nuance below)

- H4/S2 CONFIRMED on the PPL axis: sparse+low-rank hybrid compensation
  beats plain rank at matched budget everywhere, decisively at s=0.9. The
  whitening-style cross-family inversion did NOT materialize.
- Follow-through per pre-agreed rule: proceed to B_eff=2048 round (E2)
  with the LR reference arm r2048 and mixed-leaning splits bracketing the
  observed optimum (rank share 12.5–50%), then per-layer split allocation.
- Nuance to carry: screening picked the pure-sparse end, PPL picked a
  mixed split — future allocation work must be validated on PPL, not on
  the offline metric alone.
