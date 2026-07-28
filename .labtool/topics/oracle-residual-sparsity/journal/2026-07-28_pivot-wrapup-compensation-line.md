# Steer: compensation line wrapped (paused) — frontier dominance accepted

- Date: 2026-07-28
- Type: line wrap-up within oracle-residual-sparsity; topic -> paused
- Trigger: E-W0 (050-20260729-062542) delivered two negatives — the
  pre-registered metric gate failed (diagonal loss-aligned weighting cannot
  fix the whitening inversion; Spearman identical to plain L2) and the
  bundled TIS frontier fill-in showed EVERY compensated arm at s=0.9,
  including the exact ceiling C3, is strictly dominated on the MAC-PPL
  frontier by plain TIS at lower sparsity, under both accountings
  (oracle 2hd+hK and predictor hd+2hK). User decision in-conversation:
  wrap the line rather than test the extreme corner or retry with a
  Fisher/full-covariance metric.

## Previous direction (pre-steer)

Post-critique plan: E-W0 metric validation -> E-W1 weighted SVD factors
(main bet) / E-W2 weighted selection score (side bet) -> E3 cheap-end sweep
(B_eff 512/256) -> E4 per-layer allocation. All of it presumed the
compensation line could reach the frontier; the E-W0 frontier measurement
falsified that premise at the measured grid, making E-W1/W2/E3/E4 moot as
planned.

## What survives (the honest claim)

Fixed-sparsity, within-family: if the FFN core is pinned at s=0.9,
MGR-SLR (r256:k1536, B_eff=1024, +6.2% compute) is the best known
deployable compensation — 6.9417 vs plain-rank 7.2294, 79% of the headroom
to the exact ceiling (6.6381). At B_eff=2048 SLR ties the exact rule. What
does NOT survive: any claim that compensated s=0.9 beats plain TIS at
matched total compute — TIS@0.85 = 6.7088 is cheaper and better than
SLR-B1024@0.9; TIS@0.75 = 5.8827 is cheaper and better than C3 itself.

## Reopen criteria

1. Extreme-corner frontier: s in {0.92, 0.95, 0.97} x comp {none, B256,
   B512} — the only unmeasured region where compensation can reach the
   frontier (TIS curve blows up; hK headroom < comp cost).
2. Kernel-level evaluation: dense-GEMM comp branch vs sparse-gather
   neurons — wall-clock, not MACs (outside oracle scope).
3. An application where FFN sparsity is externally pinned (memory-bandwidth
   or latency contract), making the fixed-s claim directly usable.

## Status of experiments

- No PENDING cards, no running jobs. E-W1/E-W2/E3/E4 cancelled unrun.
- Deferred items closed with the topic: Fisher/full-cov metric (untried by
  choice), quantized-M W4/W8, loss-aligned A,B fitting, per-layer
  allocation, C6 group mask, multi-model extension (spec program).

## Insights at time of pivot (gist Key Findings, preserved verbatim)

- **E2 B_eff=2048 round (2026-07-28): SLR saturates onto the exact rule;
  the budget axis is exhausted upward.** All three s2 arms reach C3 within
  −0.004…+0.029 PPL @ s=0.9 (best r256:k3584 = 6.6344 vs C3 6.6381 — a tie
  at the noise floor), while plain lr_r2048 still trails at 6.7098. The
  rank-share ordering from E1 holds but the spread collapses from 0.29 PPL
  (B=1024) to 0.075 (B=2048). ARITHMETIC NOTE (derived): 2hr = h² at
  r = h/2 = 2048, so every arm in this round costs exactly what computing
  Mx costs — C3 is available at the same budget.
  When relevant: (a) the deployable frontier is B_eff ≤ 1024 — probe
  DOWNWARD (512, 256), not upward; (b) never quote a B_eff=2048 arm as a
  compute win over exact compensation; (c) old "r=2048 reference arm"
  proposal is now closed.
  Journal: 2026-07-28_experiment-oracle-llama2-e2-s2-b2048.md
- **[MAIN] E1 S2 PPL sweep (2026-07-28): H4/S2 confirmed — best deployable
  C4 to date.** LLaMA2-7B, B_eff=1024 (+6.2% compute), wikitext-2 PPL:
  all three slr_input arms beat plain lr_r1024 at every s; best arm
  r256:k1536 = 5.5961/5.7526/6.9417 vs lr_r1024 5.7365/5.9152/7.2294 —
  gate passed with margin (−0.288 @ s=0.9 vs required −0.05); gap to exact
  C3 roughly HALVED (+0.591 → +0.304 @ s=0.9). PPL optimum is the MIXED
  split (rank share ~25%), not the pure-sparse end E0's screening picked
  (r0:k2048 second, r512:k1024 third) — cross-family transfer held, fine
  ordering didn't. At s=0.5 AND s=0.7 all approximated comp arms still
  trail plain C1 (best SLR 5.7526 vs C1 5.7284 at s=0.7; noted in critique
  2026-07-28 — the SLR advantage is confined to s=0.9 on the measured grid,
  crossover between 0.7 and 0.9 unlocated).
  When relevant: (a) E2 splits should bracket rank share 12.5–50%;
  (b) validate any allocation scheme on PPL, never on offline L2 alone;
  (c) s2 r256:k1536 is the new reference deployable arm.
  Journal: 2026-07-27_experiment-oracle-llama2-e1-s2-ppl.md
- **E0 SLR diagnostics (2026-07-27): S2 (input-channel sparse) passes
  screening decisively; S1 (neuron hot set) refuted.** Matched-budget
  B_eff=1024 rel-err E‖Mx−comp‖/E‖Mx‖ on 16k calibration tokens: every S2
  arm beats lr_r1024 (mid-stack 0.330), monotone in sparse-heaviness — best
  pure-sparse s2_r0_k2048 = 0.198 mid-stack (−40%); score |x| ≈
  |x|·‖R[:,c]‖ (Δ≤0.0006, use abs). x channel concentration is only
  moderate (top-2048 = 93.4% energy) — S2 wins structurally (zero error on
  selected channels), not via extreme outliers. S1 inverted its premise:
  removing hot rank-1 terms RAISES r90 (1270 → ~1470) — the static hot
  terms are aligned with M's top singular directions.
  When relevant: (a) E1 arms are S2-only {r512:k1024, r256:k1536, r0:k2048},
  abs score; (b) any future "pull static structure out of M" idea must first
  check it isn't just re-deriving M's top subspace; (c) input-L2 caveat
  still applies cross-family (whitening precedent) — PPL is the referee.
  Journal: 2026-07-27_experiment-oracle-llama2-e0-slr-diag.md
- **C4 whitening round (2026-07-24): rank is the lever; whitening and
  tau-allocation are both harmful.** Full 2x2 at matched budgets: plain
  uniform r=1024 is the best C4 (5.737/5.915/7.229; beats C1 at s=0.9, gap
  to C3 down to +0.23/+0.29/+0.59) at +6.2% compute. Whitening reduces its
  own L2 objective by 13% on real inputs yet worsens PPL at every rank
  (+1.00 at r512 s=0.9) — input-distribution L2 is misaligned with
  downstream loss — decile analysis: whitening halves error on top-variance
  directions (D10 ×0.47-0.57) but raises it +10-22% on low/mid-variance ones
  (D1-D5), so loss-relevant signal lives in LOW-variance directions. Energy
  allocation starves LATE layers (alloc1024 worst trunc/tail = layer 31,
  0.96; budget 256 collapses, PPL 17.4).
  When relevant: (a) scaling r is the safe C4 lever (r=2048 = +12.4% is the
  next test); (b) any future compensation objective should weight the
  OUTPUT side (downstream sensitivity), not the input distribution.
  Journal: 2026-07-24_experiment-oracle-llama2-c4-whitening.md
- **[MAIN] Phase 4 table (2026-07-24): H1 confirmed, H2 rejected, H3
  partial-go.** LLaMA2-7B, top-K matched s, wikitext-2 PPL (dense 5.4738):
  C3 (residual score + exact mean-gate compensation) cuts C1's degradation by
  34/39/56% at s=0.5/0.7/0.9 — s=0.9 PPL 8.110 → 6.638 (ΔPPL +2.636 →
  +1.164). C2 (col_norm score only) ≈ C1 at 0.5, worse at 0.7/0.9 → the gain
  is ALL from compensation, not the score. C4 (rank-512) collapses below C1
  everywhere (+0.78 PPL comp error already at s=0.5) — the mid-stack
  Frobenius deficit bites, exactly as Phase 0 flagged. C5 (ḡ*) consistently
  a bit worse than C3's plain ḡ.
  When relevant: (a) the deployable form needs bigger r — r=1024 costs ~6.2%
  compute (2r/3d), r=2048 ~12.4%; (b) don't bother with col_norm-only or
  ḡ* variants going forward.
  Journal: 2026-07-24_experiment-oracle-llama2-phase4-c2c5.md
- **Phase 3 gate passed (2026-07-24)**: oracle path under top-K reproduces the
  topk_intermediate anchors to 4 decimals on a DIFFERENT GPU (A6000) and
  attention backend (sdpa) — dense 5.4738 vs 5.4736; C1 5.5216/5.7284/8.1096
  vs 5.5210/5.7296/8.1083 (max Δ 0.0013). When relevant: backend/arch effects
  are ~1e-3, so any C2–C5 delta above ~0.01 PPL is real signal; also the C1
  row of the main table is done. Journal: 2026-07-23_experiment-...phase3.md
- **|r| histograms confirm the zero-shift (2026-07-23)**: on 5 sample layers
  the |r| distribution sits left of |i| — med|r|/med|i| = 0.813/0.640/0.659/
  0.841/0.924 at layers 0/7/16/24/31; shift largest exactly where the induced-
  sparsity gap peaks, gone by layer 31. When relevant: layer-selective
  application (exclude late layers) is the natural refinement if C3/C4 gains
  are diluted. Data: a6000-2 ~/workspace/oracle/llama2-7b/phase0/histograms.json
- **Phase-0 distribution report, LLaMA2-7B (2026-07-23): H1 go with caveats.**
  Residual r = u⊙(g−ḡ) is more top-p-concentrated than i in 30/32 layers;
  mean induced-sparsity gap ≈ +3%p (p=0.7→0.9), peaking +5%p mid-stack
  (layers 6–18) and INVERTING in layers 30–31 (−0.2/−2.6%p). ḡ corpus
  stability is weak early (c4↔wikitext103 Pearson 0.48–0.53 at layers 2–4,
  ~0.9 mid-stack). M is not strongly low-rank mid-stack: r=512 retains only
  0.54–0.60 of Frobenius energy (layers 4–17), vs 0.94–0.985 at layers 30–31.
  When relevant: (a) if C3/C4 underperform, try exclude_layers=[30,31];
  (b) C4-vs-C3 gap is the H3 signal — the heavy singular tail predicts risk;
  (c) early-layer C3/C5 results carry calibration-corpus noise.
  Journal: 2026-07-22_experiment-oracle-llama2-phase0-calib.md
- **Phase 1 complete (2026-07-22)**: oracle conditions implemented as
  `sparse_mode='oracle'` in the existing modeling file
  (`inference/oracle_mlp.py` + small hooks in modeling_llama_larosa.py);
  scripts 01–04 + sweep runner under `scripts/oracle/`. All 4 spec-§2 unit
  tests pass on CPU fp32 tiny model: p=1 identity 9e-8, C4 full-rank ≡ C3
  3e-7, C4 p=1 error exactly (M̂−M)x (residual 1e-9), mask-vs-slice 2e-9;
  plus save/load round-trip bitwise-identical and topk_intermediate
  regression smoke OK.
  When relevant: trusting the c3/c4 algebra — the compensation identities
  were verified numerically, so PPL differences in later phases are signal,
  not implementation bugs.

