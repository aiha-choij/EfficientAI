# 2026-08-04 — reopen: MGR x refit fusion

## Why reopen
The topic was wrapped 2026-07-28 on frontier dominance: at matched compute,
every compensated arm (incl. exact C3) lost to plain TIS at lower sparsity.
The wrap-up card's reopen criteria included "a new lever changes the
accounting." That lever arrived: local-loss-refit's anchored closed-form
refit (C1-fixed, PR #3) costs ZERO runtime compute, was confirmed
harmless-to-strongly-positive on both 3B/8B, and composes linearly with
this topic's deployable form:

  MGR-SLR output  y = W_d (m*r) + B (A x) + h(x)   is LINEAR in (W_d, B)
  -> re-solve [W_d, B] jointly against the dense teacher with the
  W-anchored ridge = master doc section-5's theta = {W_down, U} with
  V0 = A frozen. First actual fusion of the two verified halves.

## Targets (all measured in-family, LLaMA2-7B, wikitext-2)
- beat R0/SLR 6.9417 @ s=0.9  -> fusion works at all
- beat TIS@0.85 6.7088        -> frontier verdict contested again
- beat exact C3 6.6381        -> refit absorbs truncation bias exact comp
                                 cannot (C3 uses the original W_d)

## Experiment: fusion-mgr-refit-llama2 (Status: PENDING)

### Hypothesis tested
Anchored joint refit of (W_d, B) on top of the deployable SLR arm
(r256:k1536, fixed-s residual-score selection) recovers a meaningful part
of (a) the SVD approximation gap SLR->C3 (regression reweights toward
loss-relevant directions -- the whitening round's lesson applied
constructively) and (b) the mean-gate truncation bias C3->dense, at zero
runtime cost. Anchored prior bounds the downside at ~neutral (refit C1
result).

### Prior art check
- E1 (2026-07-27): SLR r256:k1536 = 5.5961/5.7526/6.9417 at s=0.5/0.7/0.9
  (topk select, achieved as targeted) — the R0 sanity target.
- Phase 4 + E-W0: C3 = 6.638 @ s=0.9; TIS@0.85 = 6.709; dense = 5.4738.
- local-loss-refit C1 fix: anchored ridge neutral at low s, -25~-29% at
  s=0.9; 0-shrinkage prior is a known confound — anchor used from day one.
- Whitening round: never judge on offline L2; PPL is the referee (this
  design's B-refit is exactly the "weight the OUTPUT side" recommendation).

### Expected outcome
- Success: R2 < 6.94 at s=0.9 with s=0.5/0.7 no worse than R0.
  Stretch thresholds: 6.71 (frontier), 6.64 (beat exact comp).
- Failure: R2 ~ R0 everywhere (headroom already eaten by compensation),
  or improvement only at the cost of low-s regressions (would contradict
  the anchored-prior result and demand a look at the Gram conditioning).

### Reproducibility
- Git tag: `exp/2026-08-04_oracle-refit-fusion` (commit 091e2e6, main)
- Job ID: 050-20260804-121113-fusion-mgr-refit-llama2
- Assigned host/GPU: a100-40-2 (pinned), GPU [pending dispatch]
- Command: `bash -c ".../envs/larosa/bin/python
  scripts/oracle/06_refit_fusion.py --model_name /raid/LLM/llama2-7b
  --rank 256 --input_k 1536 --s_list 0.5,0.7,0.9 --lambdas 0.01,0.1
  --calib_seqs 128 --layers_per_pass 4 --arms dense,r0,r1,r2
  --out /home/choij/workspace/fusion/llama2-7b"`
  (qsub -H a100-40-2 -g 1 -m 40 -d ~/workspace/repos/EfficientAI-verify/larosa)
- Config: standalone script (vanilla HF + monkeypatched LlamaMLP.forward);
  calib wikitext-103 train 128x2048, eval wikitext-2 test 166x2048;
  residual score |u*(g-g_bar)|*colnorm with ORIGINAL W_d norms (gauge
  fixed, anti-circularity); g_bar from in-script pass 0 (same calib);
  sparse correction h(x) kept at ORIGINAL factors in all arms.
- Arms: dense sanity; R0 (original W_d,B — must land near E1's numbers);
  R1 (W_d only, Schur sub-solve from the same joint Gram); R2 (joint),
  lambda in {0.01, 0.1}.
- Key deps: gateway conda env `larosa` (torch 2.6.0+cu124, transformers
  4.46.3); runs from the verify clone (~/workspace/repos/EfficientAI-verify)
  — the main clone's worktree belongs to the active block-compensation
  agent session and is deliberately untouched.
- Selftest (CPU) passed 4/4 pre-submit: exact M-split identity
  (BAx + Rres@full-k == Mx); anchored recovery at lam->inf; in-sample
  error strictly drops at lam=0.01; R1 Schur sub-solve == direct solve.
  One real bug caught in review pre-submit: Gram hook was registered on
  the decoder layer (pre-attention hidden state) instead of mlp — fixed
  before any GPU time was spent.

### Results
- Round 1 (050-20260804-121113): FAILED — CUDA OOM during Gram
  accumulation (log tail in runs dir). Three compounding memory design
  errors, all fixed in commit 3cc64e6: (1) `G + phi.T@phi` reassignment
  double-buffered each 508MB accumulator -> now in-place `add_`;
  (2) all-layer SLR factors (Rres h x h, ~2.1GB total) were GPU-resident
  during build -> now CPU-resident, moved per hooked layer; (3) solved
  weights (32 layers x 3 s x 2 lam x ~184MB, worst case ~69GB) accumulated
  on GPU -> now stored on CPU, moved back per-layer at eval. Also caught
  before wasting the queue slot: qsub -m 40 never matches A100-40GB
  (usable ~39.6GiB) — all three 08-04 jobs sat unplaced until MINFREE was
  edited to 32 in the pending meta.
- Round 2 resubmitted: 050-20260804-130635-fusion-mgr-refit-llama2b,
  same protocol + layers_per_pass=2 + expandable_segments. PENDING.
- **Round 2 (050-20260804-130635): STATUS=ok.** Source:
  a100-40-2:~/workspace/fusion/llama2-7b/fusion_results.json (git 3cc64e6).
  Dense sanity 5.4735 (trusted anchor 5.4736 — protocol match to 4
  decimals). wikitext-2 PPL, achieved sparsity exact:

  | s | R0 (SLR in-pipeline) | R1 lam=.1 (W_d only) | R2 lam=.1 (joint) |
  |---|---|---|---|
  | 0.5 | 5.5989 | 5.5622 | 5.5596 |
  | 0.7 | 5.7498 | 5.6551 | 5.6495 |
  | 0.9 | 6.7800 | 6.2075 | **6.1954** |

  - All three pre-registered s=0.9 thresholds passed by R2 (and by R1):
    6.94 SLR / 6.7088 TIS@0.85 / 6.6381 exact C3.
  - No low-s regression (R2 improves R0 at every s) — anchored-prior
    harmlessness reconfirmed.
  - lam=0.1 slightly > lam=0.01 everywhere (consistent with the refit C2
    sweep); lam=0.01 rows omitted above, in the JSON.
  - R1 vs R2 nearly tied (6.2075 vs 6.1954): the gain is dominated by the
    W_d refit; joint B refit adds ~0.01 PPL.
  - Caveat: R0 lands at 6.78, below E1's 6.9417 (different g_bar
    calibration sample suspected) — cross-arm deltas within this pipeline
    are the reliable readout; cross-referencing E1 absolute numbers needs
    that caveat.
  - Frontier note: beating 6.7088 contests, not settles, the E-W0 verdict
    — the fair control (anchored refit applied to plain TIS at s=0.85/0.9,
    same zero-cost lever) is unmeasured; proposed as the next job.

### Interpretation
(user-confirmed 2026-08-04) Fusion is valid and beats the exact-compensation
ceiling: R2 6.1954 passes all three pre-registered thresholds (SLR 6.94,
TIS@0.85 6.7088, exact C3 6.6381) with no low-s regression. However the
gain is DOMINATED by the W_d refit (R1 6.2075 ~ R2 6.1954; joint B-refit
adds only ~0.01 PPL) — the accurate statement is "anchored refit works on
top of compensation," not "the section-5 joint formulation is what won."
The frontier verdict is CONTESTED, not overturned: the fair control
(the same zero-cost refit applied to plain TIS at s=0.85/0.9) is
unmeasured and is the next arm (R5). Follow-up round (approved same day)
attacks why the joint part is weak: frozen SVD subspace (R3
regression-first), missing token-specific gate signal (R4 sketch-tail
features = refit x C8 fusion), plus the R5 control.
