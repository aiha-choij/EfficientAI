# oracle-residual-sparsity — Mean-gate residual decomposition + rank-r compensation (oracle)

## Status
REOPENED 2026-08-04 (was: paused/wrapped 2026-07-28 on frontier dominance).
Reopen trigger: a zero-runtime-cost lever (local-loss-refit's anchored
closed-form refit, C1-fixed) composes linearly with the deployable SLR arm
-- the frontier accounting that killed the line assumed compensation gains
cost compute; refit's don't. First fusion experiment:
journal/2026-08-04_reopen-refit-fusion.md (R0/R1/R2 ladder; thresholds
6.9417 SLR / 6.7088 TIS@0.85 / 6.6381 exact C3).

## Active Jobs (reopen)
- `050-20260804-165432-fusion-r3-llama2` (a100-40-2, PENDING) — round 3
  amplification: r3 regression-first / r4 refit x C8 sketch-tail /
  r5 refit-TIS frontier control. Card:
  journal/2026-08-04_experiment-fusion-r3-amplify.md
- (done) 050-20260804-130635 round 2 — R2 6.1954 @ s=0.9, all thresholds
  passed; card: journal/2026-08-04_reopen-refit-fusion.md

## Authoritative spec
`spec.md` in this topic (preserved verbatim, 2026-07-22). This gist is the
working summary; when they disagree, spec.md wins for the experiment design and
this gist wins for infra adaptations decided afterwards.

## Notation
FFN(x) = W_d i, i = u ⊙ g, u = W_u x, g = σ(W_g x) — σ is the model's own
act_fn (SiLU for LLaMA/Qwen, GELU-tanh for Gemma; never hardcode).
- ḡ[j] = E[g_j] (calibration mean), ḡ*[j] = E[u_j²g_j]/E[u_j²]
- residual r_j = u_j·(g_j − ḡ[j]); col_norm[j] = ‖W_d[:,j]‖₂
- Compensation M = W_d diag(ḡ) W_u ∈ R^{h×h}; rank-r via SVD, runtime 2hr/token

## Hypothesis
Oracle setting (mask from true activations; no predictor/kernel/training):
- **H1**: r = u⊙(g−ḡ) is more concentrated than i = u⊙g, so the mean-gate
  decomposition (sparse residual + linear tail compensation) holds accuracy at
  higher sparsity for equal effective compute.
- **H2**: weight-aware scoring |i_j|·col_norm[j] alone already beats plain |i_j|.
- **H3**: some r ≤ h/8 exists where rank-r approximation of M does not eat H1's gain.
- **H4 (2026-07-27 steer, R-Sparse template)**: at matched MAC budget, a
  sparse + low-rank hybrid of comp(x) ≈ Mx beats plain uniform rank-r,
  because M's heavy singular tail mid-stack is concentrated in few exact
  contributions — static hot rank-1 neuron terms (S1) and/or dynamic
  top-|x| input channels of the SVD residual (S2). Steer card:
  journal/2026-07-27_pivot-c4-slr-compensation.md

Ablation ladder (spec §2): C0 dense / C1 |i| top-p / C2 +col_norm /
C3 residual score + exact tail compensation (diagnostic, not deployable) /
C4 rank-r deployable form / C5 ḡ* variant / C6 optional group mask (G=8/32).
Selection is top-p per token per layer; report axis is ACHIEVED sparsity.
Judgment: critical sparsity = max achieved sparsity with normalized accuracy
≥ 0.99 (lm-eval zero-shot 8-task suite); wikitext PPL as secondary signal.
Go/no-go (spec §8): Go if 8B C4(r≤d/8) critical sparsity ≥ C2 + 15%p;
No-go if C3 − C2 < 5%p.

## Design decisions (2026-07-22 pivot Q&A)
- **Selection = top-K, not top-p (user, 2026-07-23)**: PPL sweeps use exact
  per-token sparsity s (K = int((1−s)·d), same tie semantics as top_k_new) so
  C1 exactly reproduces the larosa topk_intermediate setup and **C3 vs C1 at
  equal s is the primary readout** (matched per-token compute; no achieved-
  sparsity interpolation). The spec's top-p mask stays implemented
  (select=topp) for spec-faithful runs; unit test pins C1-topk ==
  topk_intermediate bitwise. s grid: {0.5, 0.7, 0.9} only (user, 2026-07-23 —
  match the existing Top-K experiment's levels exactly).
- **Scope narrowed (user, 2026-07-22)**: LLaMA2-7B ONLY (h=4096, d=11008), to
  compare directly against the larosa Top-K results; evaluation is the SAME
  wikitext-2 PPL pipeline (`eval_ppl_wikitext_with_inference_sparsity`) — no
  lm-eval, no critical-sparsity-by-accuracy; C4 runs r = h/8 = 512 only.
  The spec's multi-model/accuracy program is deferred, not canceled.
- **No R-Sparse fork** (user decision): implement standalone in EfficientAI on
  the existing HF loading/eval pipeline; the spec's "reuse from R-Sparse" items
  (model loading, calibration loader, SVD util, lm-eval integration) are
  implemented fresh. Spec §3's "disable R-Sparse modules" is trivially
  satisfied — none exist here. Contamination rule still binding: attention and
  all linears stay dense; only the MLP forward is wrapped.
- Old topic's 3-model Top-K PPL extension preserved as backlog in
  larosa-intermediate-sparsity gist (not orphaned).
- All sparsification simulated as compute-then-mask (oracle-equivalent).
- **SLR budget accounting (2026-07-27 steer)**: compare all C4 variants at
  equivalent-rank budget B_eff — low-rank r costs 2h·r MACs/token, one hot
  neuron (S1) costs 2h (≡ rank 1), one kept input channel (S2) costs h
  (≡ rank 1/2). r=1024 ≙ +6.2% compute, r=2048 ≙ +12.4%. Diagnostics-first:
  offline gates prune arms; input-L2 error is SCREENING ONLY (whitening
  lesson) — PPL is the referee.

## Key Findings
- **[MAIN, REOPEN ROUND] Anchored refit fused onto the deployable SLR arm
  beats the exact-compensation ceiling at zero runtime cost (2026-08-04).**
  LLaMA2-7B, wikitext-2, fixed-s residual-score selection, in-pipeline
  (dense sanity 5.4735): R0(SLR) 5.599/5.750/6.780 at s=0.5/0.7/0.9 ->
  R2(joint anchored refit of W_d+B, lam=0.1) 5.560/5.650/6.195. At s=0.9
  all three pre-registered thresholds passed: SLR 6.94, TIS@0.85 6.7088,
  exact C3 6.6381 — refit absorbs masking-adaptation bias in W_d that even
  EXACT mean-gate compensation cannot (C3 keeps the original W_d). Gain is
  dominated by the W_d block (R1 6.2075 ~ R2 6.1954; B-refit adds ~0.01).
  When relevant: (a) frontier verdict is contested but NOT settled — the
  fair control is refit-TIS (same free lever on the plain-mask arm),
  unmeasured; (b) any compensated arm should ship with anchored W_d refit
  by default (harmless at low s, large at high s); (c) R0 here lands below
  E1's 6.9417 (g_bar calibration sample differs) — use in-pipeline deltas.
  Journal: 2026-08-04_reopen-refit-fusion.md
- **[MAIN] E-W0 frontier test (2026-07-28): the compensation line is
  dominated at matched compute — line wrapped.** TIS fill-in: c1 PPL
  5.8827 / 6.1546 / 6.7088 at s=0.75 / 0.8 / 0.85. Under BOTH accountings
  (oracle 2hd+hK, predictor hd+2hK), every compensated arm at s=0.9 —
  SLR-B1024 (6.9417), SLR-B2048 (6.6344), and the exact ceiling C3
  (6.6381) — is strictly dominated by a cheaper-and-better TIS point
  (TIS@0.85 = 6.7088 beats SLR-B1024; TIS@0.75 = 5.8827 beats C3). Near
  s=0.9, MACs spent on more neurons buy ~0.62 PPL/M vs ~0.14 for the comp
  branch (~4.5x). Also: the diagonal loss-aligned metric FAILED its gate —
  identical Spearman to plain L2 (+0.945), cannot fix the whitening
  inversion (per-channel diagonal cannot see covariance-rotated error);
  positive control passed, so the negative is real.
  When relevant: (a) the surviving claim is fixed-sparsity only ("if s is
  pinned at 0.9, SLR is the best deployable form"); (b) NEVER present a
  compensated arm as a frontier win without the TIS-at-lower-s comparison;
  (c) reopen criteria: extreme corner (s>=0.95, B<=512), kernel-level
  wall-clock, or an externally-pinned-sparsity application.
  Journal: 2026-07-28_experiment-oracle-llama2-ew0-gradstats.md
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

## Dead Ends
- 2026-07-28 — **Compensation as a frontier method (the line's premise)**:
  at the measured grid every compensated arm at s=0.9, including exact C3,
  is strictly dominated on the MAC-PPL frontier by plain TIS at lower s
  (both accountings). The line survives only as a fixed-sparsity claim.
  Wrapped by user decision; refuges (s>=0.95 corner, kernel wall-clock)
  unmeasured. Journal: 2026-07-28_experiment-oracle-llama2-ew0-gradstats.md
- 2026-07-28 — **Diagonal output-gradient metric weighting**: w =
  sqrt(E[(dL/dy)^2]) per channel cannot fix the whitening inversion —
  Spearman identical to plain L2 across 11 known-PPL variants; whitening's
  error redistribution lives in covariance-rotated directions invisible to
  a per-channel diagonal. Full-covariance (Fisher) variant untried by
  choice. Journal: 2026-07-28_experiment-oracle-llama2-ew0-gradstats.md
- 2026-07-27 — **S1 / slr_neuron (static hot-neuron sparse + low-rank
  split, LoSparse-style)**: tried trading rank for exact hot rank-1 terms
  of M (c_j = ḡ_j‖W_d[:,j]‖‖W_u[j,:]‖). Failed because the premise
  inverted: removing hot terms makes M_cold HARDER to approximate (mean r90
  1270 → ~1470; energy@2048 drops) — the static hot terms ARE the low-rank-
  aligned part of M. Every arm lost to plain lr_r1024 at matched budget,
  monotonically in hot_n (mid-stack 0.37→0.72 vs 0.33). User: dead end.
  Journal: 2026-07-27_experiment-oracle-llama2-e0-slr-diag.md
- (whitening-round negatives remain ON HOLD, not dead: see Open Questions
  "deferred" items; user decision 2026-07-24, revisit with the
  adaptive-rank-allocation line of work)

## Open Questions
- **SLR risks (2026-07-27; (1) resolved by E0)**: (1) ~~S2 premise
  unverified~~ → E0: concentration only moderate (top-2048 = 93.4%) yet S2
  wins anyway — structural advantage, not outlier-driven; (2) E0's
  approx-error metric is input-L2, which the whitening round proved can
  invert vs downstream loss — use it only to prune arms, never to claim a
  win (S2-vs-LR is cross-family, so E1 PPL remains the referee); (3) oracle
  sim ignores dynamic top-k selection/gather kernel cost for S2 (consistent
  with topic convention; must be stated in claims); (4) S1-hot-in-mask
  variant (true gate, 3h/neuron) NOT tried — and now moot for compensation
  (S1 dead), but note the E0 lesson: hot terms ≈ M's top subspace.
- **DEFERRED (user, 2026-07-24 — revisit with adaptive rank allocation
  research)**: (1) whitened SVD compensation — worsened PPL at every rank
  despite −13% input-L2, but the objective-mismatch mechanism deserves its
  own study before writing it off; (2) per-layer rank allocation by spectral
  energy — failed with the whitened metric; an allocation metric aligned
  with downstream loss may still work.
  Journal: 2026-07-24_experiment-oracle-llama2-c4-whitening.md
- ~~Rank grid~~ → resolved: r = 512 (h/8) only (user, 2026-07-22).
- ~~Model/eval choices~~ → resolved: LLaMA2-7B + wikitext-2 PPL (user).
- **Calibration corpus download on gateway**: 01_calibrate defaults to
  allenai/c4 en streaming — network access from execution servers worked for
  wikitext, c4 streaming untested. Fallback flag: `--dataset wikitext103`.
- **PPL success threshold**: same open convention as the old topic — judge
  ΔPPL vs dense (5.4736) against the Top-K anchors (C1-equivalent: 5.521/
  5.730/8.108 at s=50/70/90); spec's §8 %p margins translate to "C3/C4 hold
  ≤ Top-K's ΔPPL at ≥15%p higher achieved sparsity" — formalize once curves exist.

## Next Experiments
(topic paused 2026-07-28 — none. Reopen criteria in the wrap-up card:
extreme-corner frontier s∈{0.92,0.95,0.97} × comp{none,B256,B512};
kernel-level dense-GEMM-vs-gather wall-clock; externally-pinned-sparsity
application.)

## Active Jobs
- `050-20260729-022242-oracle-llama2-e2-s2-b2048` (PENDING, a6000-2) — E2
  B_eff=2048 round, 12 runs; card
  journal/2026-07-28_experiment-oracle-llama2-e2-s2-b2048.md

## Pointers
- Spec: `spec.md` (this topic). Pivot record:
  `../larosa-intermediate-sparsity/journal/2026-07-22_pivot-oracle-residual-sparsity.md`
- Prior confirmed result to beat (C1-like, Top-K, PPL axis): LLaMA2-7B
  intermediate Top-K 50% → +0.047 PPL (larosa-intermediate-sparsity Key Findings).
- Code base: EfficientAI `larosa/inference/mlp.py` (existing mode plumbing:
  config.sparse_mode flag pattern from commit 40edf40), conda env `larosa` on
  gateway (flash-attn 2.7.4.post1 pinned).
- Models on gateway: /raid/LLM/llama2-7b, /raid/LLM/llama3-8b (read-only),
  ~/workspace/models/Qwen2.5-7B. HF cache symlinked to ~/workspace/cache.
- Compute note: M is h×h (8B: 4096² × 32 layers ≈ 1.1GB fp32 total) — SVD per
  layer fits easily on any A100; calibration + Phase-0 are single-GPU jobs.
