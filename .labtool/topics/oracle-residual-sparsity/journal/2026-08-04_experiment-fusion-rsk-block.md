# 2026-08-04 — experiment: fusion-rsk-sweep + fusion-block-g16

Status: PENDING

## Hypothesis tested
Two follow-ups to the r4 result (5.946 @ s=0.9, broke the 6.150 linear
ceiling, but ~0.80x-of-dense-FFN measured compute):
1. (rsk-sweep, g=1) The token-wise gate/up sketch keeps most of its value
   at much smaller rank — block-sparse-compensation saw C8a work at
   r_sk=d/32. Arms r4/r4trunc at r_sk in {344 (d/32), 688 (d/16)};
   r4trunc additionally truncates the LEARNED tail-output map to rank r_sk
   (deployable compute: sketch paths 3*r_sk*(h+d)/(3hd) instead of the
   dense W_tail's +1/3).
2. (block-g16) The validated refit + sketch-tail combination ports to the
   token-block-shared mask setting (g=16, the research line's real target)
   and recovers a large share of the sharing tax. In-family ladder at
   g=16, s=0.9: m0 (mask-only control, = block-comp's C7a semantics) /
   r0 (SLR, no refit) / r2 (SLR+refit) / r4, r4trunc (sketch-tail+refit,
   r_sk=1376) / r5 (plain-|i| block mask + refit). g=1 anchors from
   rounds 2-3: r2 6.195, r4 5.946.

## Prior art check
- Round 3 (2026-08-04): r4 breaks the linear-in-x ceiling only at s=0.9;
  compute accounting corrected to ~0.80 (dense W_tail) — this experiment
  is the mandated efficiency + porting step.
- block-sparse-compensation topic: token-block masks without compensation
  are catastrophic (8B C7a ~1243); token-wise sketch (C8) recovers ~99% of
  the collapse but sits 2.6x above the per-token anchor — the gap this
  port aims at. C8a at r_sk=d/32 stayed strong -> small-rank optimism.
- local-loss-refit: refit's g>1 rows were measured with the buggy
  0-shrinkage prior; the r5/r2 arms here re-measure block refit with the
  anchored prior (partial fill of that gap, llama2-7b).

## Expected outcome
- rsk-sweep: success = r4(d/16) within ~0.05 PPL of 5.946 and r4trunc
  close behind (deploy compute ~0.10+0.062+3*688*15104/1.35e8 ~ 0.39);
  failure = sharp degradation at both small ranks (sketch quality is the
  binding constraint; would argue for asymmetric ranks next).
- block-g16: m0 expected catastrophic (sanity vs block-comp); key readout
  = r4 vs r0/r2 recovery and r4(g=16) vs r4(g=1)=5.946 sharing-tax gap.
  Success (provisional): r4 g=16 lands within ~2x of the g=1 anchor PPL
  and clearly below r2 g=16; finalize after first numbers.

## Reproducibility
- Git tag: `exp/2026-08-04_fusion-rsk-block` (commit after 4646a17 md5
  4646a174..., main)
- Job IDs: 050-20260804-214822-fusion-rsk-sweep,
  050-20260804-214824-fusion-block-g16 (both -H a100-40-2 -g 1 -m 32,
  workdir ~/workspace/repos/EfficientAI-verify/larosa)
- Commands: 06_refit_fusion.py with
  (1) `--r_sk 344,688 --s_list 0.9 --group_size 1 --arms r4,r4trunc
      --out ~/workspace/fusion/llama2-7b-rsk`
  (2) `--r_sk 1376 --s_list 0.9 --r5_s 0.9 --group_size 16
      --arms m0,r0,r2,r4,r4trunc,r5 --out ~/workspace/fusion/llama2-7b-g16`
  common: rank 256, input_k 1536, lam 0.1, calib wt103 128x2048, lpp=1,
  expandable_segments.
- Block rule: score summed over g=16 consecutive tokens within one
  2048-token sequence (never crosses sequences), top-K per block,
  broadcast (block_topk_mask; g=1 bitwise-reduces to per-token).
- Key deps: gateway conda `larosa`; bf16-projection eval path (as round B).
- Selftest (CPU) 6/6 pre-submit: adds g=1 reduction identity, within-block
  mask uniformity, per-token K preservation under sharing.

### Results

### Interpretation
