# 2026-08-04 — experiment: fusion-r3-amplify

Status: PENDING

## Hypothesis tested
Round 2's fusion gain was dominated by the W_d refit (joint B-refit added
~0.01 PPL). Three structural limits were identified; each gets an arm:
1. (r3full/r3trunc) The low-rank basis A is frozen at the SVD of M —
   a Frobenius-optimal, loss-agnostic subspace (whitening-round lesson).
   Regression-FIRST: solve [W_d, T] with T a full h x h linear
   compensation (features [m*r ; x], target y*, T anchored at M), then
   truncate T to rank-256 + top-1536-channel sparse residual for the
   deployable form. r3full is the diagnostic ceiling for ANY linear-in-x
   compensation; r3trunc is its SLR-cost projection.
2. (r4) Refit's features cannot see token-idiosyncratic gate deviation —
   the signal that made C8 >> C7 in block-sparse-compensation. Add a
   sketch-tail feature block (1-m)*(ghat*uhat) (r_sk = d//8 SVD sketches
   of W_gate/W_up) with its own output map anchored at W_d. This is the
   first "refit x C8" unification (handoff section-3 priority-1, tested
   at g=1 before the block version).
3. (r5) Fair frontier control: the E-W0 verdict compared UNREFIT plain
   TIS to compensated arms; refit is free for both sides. r5 = plain
   |i| top-K mask + anchored W_d refit at s in {0.85, 0.9}. The frontier
   contest is r2/r4-family @ s=0.9 vs r5 @ s=0.85 (+the compute delta of
   the compensation path).

## Prior art check
- Round 2 (same card series): R2 6.1954 @ s=0.9 (thresholds all passed);
  R1~R2 => W_d-dominance motivating these arms.
- block-sparse-compensation: C8a>>C7 at every setting (token-wise gate
  estimate is the key lever) — r4 imports exactly that signal.
- Whitening round: input-distribution-optimal factorizations can hurt
  downstream — r3's regression-first design is the constructive use.
- local-loss-refit L2 Dead End: sequential/deployed-stream calibration
  compounds errors — deliberately NOT attempted here (x-tilde asymmetry
  deferred).

## Expected outcome
- r3full vs r2 (6.1954): if ~equal, linear-in-x compensation is exhausted
  (strong negative worth recording); if clearly better, the basis was the
  bottleneck and r3trunc tells how much survives deployment budget.
- r4 < r2 by a visible margin would confirm token-wise gate signal helps
  INSIDE the closed form; r4 ~ r2 pushes the C8-fusion hope to the block
  setting where the tax is larger.
- r5 @ 0.85 vs best fusion arm @ 0.9 settles the reopened frontier
  question fairly (both sides refit).
- Failure mode: any arm regressing below R0 at matched s (anchored prior
  should prevent it; if seen, inspect Gram conditioning of the 22k-dim
  famA solve).

## Reproducibility
- Git tag: `exp/2026-08-04_oracle-refit-fusion-r3` (commit 4fe359e, main)
- Job ID: 050-20260804-165432-fusion-r3-llama2
- Assigned host/GPU: a100-40-2 (pinned), GPU [pending dispatch]
- Command: `bash -c "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  .../envs/larosa/bin/python scripts/oracle/06_refit_fusion.py
  --model_name /raid/LLM/llama2-7b --rank 256 --input_k 1536
  --s_list 0.7,0.9 --r5_s 0.85,0.9 --lambdas 0.1 --calib_seqs 128
  --layers_per_pass 1 --arms r3full,r3trunc,r4,r5
  --out /home/choij/workspace/fusion/llama2-7b"`
  (qsub -H a100-40-2 -g 1 -m 32 -d ~/workspace/repos/EfficientAI-verify/larosa)
- Config: same protocol as round 2 (calib wt103 128x2048, eval wt2
  166x2048, anchored ridge, gauge-fixed masks); s=0.5 dropped
  (harmlessness established); lam=0.1 only (round-2 winner);
  out file fusion3_results.json (round-2's fusion_results.json preserved).
- Key deps: gateway conda `larosa`; verify clone (agent worktree untouched).
- Selftest (CPU) 5/5 pre-submit: M-split identity; famA prefix-sub-block
  solve == direct r2 solve; full-rank sketch tail == exact dropped
  intermediate; r3trunc == r3full at full rank; in-sample error drop +
  anchor recovery at huge lambda.
- Memory budget (post round-2 OOM discipline): lpp=1; famA Gram 1.98GB,
  famB 0.91GB, famC 0.49GB per (layer, s); peak est ~27GB build / ~30GB
  eval on 39.6GiB A100.

### Results
- Round A (050-20260804-165432): STATUS=fail — r3full/r3trunc completed,
  crashed at r4 set_arm with CUDA OOM. Cause: eval closures held fp32
  clones of W_gate/W_up per layer (~11.5GB across 32 layers), on top of
  r4's solved+sketch tensors -> ~42GB demand. Fix (commit after 4fe359e):
  eval bodies reuse the module's own bf16 gate/up projections (matches
  deployed numerics; ~1e-3 protocol shift vs the fp32-hook path, below
  the Phase-3 noise gate). r4/r5 resubmitted as
  050-20260804-190813-fusion-r3b-r4r5 (out dir llama2-7b-r4r5).
- **r3 results (valid, from round A; fp32-hook eval path, comparable to
  round 2):** wikitext-2 PPL, lam=0.1, achieved sparsity exact:

  | arm | s=0.7 | s=0.9 |
  |---|---|---|
  | r3full (linear-in-x ceiling, h x h regression) | 5.5798 | **6.1501** |
  | r3trunc (rank256+k1536 deployable projection) | 5.6647 | 6.2055 |
  | (round-2 reference R2) | 5.6495 | 6.1954 |

  Reading: the FULL linear compensation ceiling (6.1501) sits only ~0.045
  PPL below R2 (6.1954) at s=0.9 — **linear-in-x compensation is
  essentially exhausted**; the frozen SVD basis was NOT a meaningful
  bottleneck (r3trunc ~ R2: the small full-T gain does not survive
  rank+sparse truncation). Whatever headroom remains toward dense (5.47)
  requires signals nonlinear in x — exactly what r4's token-wise gate
  sketch features test next.

### Interpretation
