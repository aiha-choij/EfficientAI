# Steer (within-topic): C4 compensation → sparse + low-rank hybrid (R-Sparse template)

- Date: 2026-07-27
- Type: direction change within oracle-residual-sparsity (NOT a topic pivot;
  H1–H3 frame stays, the deployable-C4 lever changes)
- Trigger: user request — graft the R-Sparse (ICLR25, reproduced in
  topics/rsparse-repro) sparse/low-rank split onto the Mx compensation branch
  to improve the mean-gated-residual deployable form.

## Previous direction (gist as of 2026-07-27, pre-steer)

Deployable C4 front: plain uniform rank is the only working lever
(r=1024 → 7.229 @ s=0.9, +6.2% compute); whitening and spectral-energy
allocation proven dead ends. Next-experiment proposals awaiting user pick:
1. Quantized full-rank M, W4/W8 (LQER/ASER template) + r=2048 reference arm.
2. Neuron-level sparse + low-rank split (LoSparse template) with
   hot-set-removal r90 offline gate.
3. Loss-aligned A,B fitting (Low-Rank Correction template) /
   anti-whitening alpha-sweep probe.

## Why we steer

- Phase-0 finding: M is NOT low-rank mid-stack (r=512 keeps only 0.54–0.60
  Frobenius energy, layers 4–17) — a heavy singular tail is exactly the regime
  R-Sparse's hybrid targets: capture the tail with exact sparse computation,
  the bulk with low rank.
- rsparse-repro evidence: sparse-heavy splits win (searched alpha mean 0.95;
  pure low-rank fails, paper Table 3) — the sparse component carries the load.
- Our M has structure R-Sparse's generic W lacks: M = Σ_j ḡ_j W_d[:,j]W_u[j,:]
  is an explicit sum of d rank-1 terms with static weights → a natural static
  sparse axis (hot neurons) in addition to R-Sparse's dynamic input-channel
  axis. Old proposal 2 (LoSparse) is subsumed by this frame, not discarded.

## New direction

C4 compensation comp(x) ≈ Mx becomes a budget-split hybrid,
comp = B(Ax) + sparse-part, two variants at matched MAC budget
(budget B_eff ≡ equivalent uniform rank; hot neuron ≡ rank 1, sparse input
channel ≡ rank 1/2):

- **S1 (slr_neuron, static)**: hot set H = top-n_h neurons by
  c_j = ḡ_j·‖W_d[:,j]‖·‖W_u[j,:]‖; exact rank-1 terms for H, SVD(M_cold) for
  the rest. Cost 2h·(|H| + r).
- **S2 (slr_input, dynamic, literal R-Sparse)**: A,B = SVD_r(M),
  R = M − BA; per token keep top-k input channels S(x) (score |x_c| or
  |x_c|·‖R[:,c]‖), comp += R[:,S] x_S. Cost 2hr + k·h.

Diagnostics-first: offline gates (hot-set-removal spectra; x channel
concentration + matched-budget calibration approx error) prune the arm count
before any PPL job. PPL is the referee — the whitening round proved input-L2
metrics can invert downstream (screening only).

## Status of prior proposals (deprioritized, NOT abandoned)

- Quantized full-rank M (W4/W8): stays in gist Next Experiments, lower
  priority; orthogonal to SLR (can compose later).
- r=2048 reference arm: folded into the SLR budget-2048 comparison round.
- Loss-aligned A,B fitting / anti-whitening alpha-sweep: deferred, unchanged.
- No PENDING journal cards in this topic; no running jobs affected.

## Insights at time of pivot (gist Key Findings, verbatim)

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
