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
Both jobs STATUS=ok (runs 20260804-214822 / -214824; artifacts
~/workspace/fusion/llama2-7b-{rsk,g16}/fusion3_results.json).

**(1) rsk-sweep (g=1, s=0.9; references: r4@d/8=5.946, R2=6.195,
linear ceiling=6.150, dense=5.474):**

| arm | r_sk=344 (d/32) | r_sk=688 (d/16) | r_sk=1376 (d/8, prior round) |
|---|---|---|---|
| r4 (full learned tail map) | 6.168 | 6.099 | 5.946 |
| r4trunc (tail map truncated to r_sk) | 6.532 | 6.434 | (not run) |

- r4 degrades gracefully with rank (5.946 -> 6.099 -> 6.168); at d/16 it
  still sits below the linear ceiling (6.099 < 6.150) — the nonlinear
  gain survives half-rank sketches.
- r4trunc is the bad news: post-hoc truncation of the LEARNED tail
  output map destroys most of the gain (6.43-6.53, worse than R2 6.195
  at ~2.4x R2's compute). The nonlinear benefit does not live in a
  low-rank subspace of W_tail that plain SVD can find after the fact.
  Deployable-form design is now the open engineering problem (candidates:
  factored/bilinear learning of the tail map, C8-style shared down-sketch
  with learned scaling).

**(2) block-g16 (token-block-shared masks, g=16, s=0.9, LLaMA2-7B;
per-token anchors: r4 g=1 = 5.946, R2 g=1 = 6.195):**

| arm | PPL | log-PPL share of the sharing tax recovered* |
|---|---|---|
| m0 (block mask only, no comp/refit) | 84.47 | 0% (control) |
| r5 (plain-magnitude mask + refit) | 14.44 | 66.6% |
| r0 (SLR comp, no refit) | 11.55 | 75.1% |
| r2 (SLR + refit) | 11.40 | 75.6% |
| r4trunc (sketch-tail, truncated map, r_sk=d/8) | 7.238 | 92.6% |
| **r4 (sketch-tail features, full map)** | **6.266** | **98.0%** |

*recovery = 1 - [ln(PPL_arm) - ln(5.946)] / [ln(84.47) - ln(5.946)],
i.e. the share of the m0-to-per-token-anchor log-PPL gap closed.

- Headline: in the block regime REFIT ALONE BARELY HELPS (r0 11.55 ->
  r2 11.40, -1.3%, vs -8.6% at g=1) but the token-wise sketch features
  are decisive (11.40 -> 6.266, -45%): with them, 16-token-shared masks
  land within +5.4% PPL of the per-token r4 anchor. This is the
  sharing-tax mechanism confirmed at the fusion level — the tax IS
  token-idiosyncratic gate information, which only a per-token
  (nonlinear) estimate can restore; static/linear repairs saturate near
  75%.
- Even the deployable truncated form (7.238, ~0.39 compute) beats every
  linear block arm by a wide margin.
- r5 < r0: in the block regime, plain-mask+refit loses to unrefit
  compensation — compensation grows in importance as the tax grows.

### Interpretation
(user-confirmed 2026-08-04) In the block-shared regime refit alone is
nearly inert, but the token-wise estimate features recover 98% of the
sharing tax in log-PPL (6.266 at g=16, within +5.4% of the per-token
anchor) — the quality side of the group-sparsity problem is effectively
solved by the refit x sketch fusion. The single remaining problem is the
deployable form: post-hoc SVD truncation of the learned tail map loses
most of the gain, so the tail map must be learned low-rank from the
start (alternating least squares on the factored form), which is the
next experiment (approved in the same message).
