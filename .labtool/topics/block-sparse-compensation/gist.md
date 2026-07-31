# block-sparse-compensation — Can input-dependent compensation recover the sharing tax of a block-shared mask?

## Status
Phase 1-3 CONFIRMED, **Phase 3's formal spec §5 gate MET: GO** (8B, g=16,
s≈0.9 — see Key Findings for the recovery table and caveats). Phase 4
(coactivation combination on LLaMA2-7B) not yet started — next up.
Started 2026-07-31 from spec (full spec preserved verbatim in
`spec.md`; see there for the authoritative math). Continues the
`oracle-residual-sparsity` C0–C6 numbering (paused, kept active as anchor)
and is directly motivated by two closed threads:
- `local-loss-refit` (done): closed-form W_down refit alone only helps at
  s=0.9; it cannot recover token-specific information lost to a *shared*
  mask because it's a static linear correction. Advisor consensus
  (2026-07-31): the g>1 performance lever has to be input-dependent
  compensation (this topic) + neuron permutation (`coactivation-block-
  structure` P2, combined here in Phase 4/P3′).
- `coactivation-block-structure` P3 (CONFIRMED 2026-07-25, interpretation
  completed 2026-07-31 as part of this topic's setup): a bare block-shared
  mask (clustered or random neuron blocks, no compensation) is catastrophic
  — PPL 4.5k–24k vs the 8.11 per-token anchor at s=0.9, regardless of
  clustering quality. This topic exists because that result establishes the
  size of the gap any compensation scheme has to close.

## Hypothesis
- **H4 (primary)**: the sharing tax of a block-shared mask (m_T, blocks of
  g=16–64 consecutive same-sequence tokens) can be recovered ≥50% by
  block-wise compensation — C7 (block version of oracle C4's mean-gate
  compensation) and C8 (per-token low-rank gate/up/down sketch instead of
  the mean-gate ḡ).
- **H5**: sharing-tax neurons are the ones whose gate deviates most from ḡ,
  so a per-token low-rank gate estimate ĝ (C8) beats the mean-gate ḡ
  compensation (C7) — the direct evidence is C8a (diagnostic, exact gate
  sketch-free upper bound) − C7.
- g=1 is kept as the anchor (the existing oracle C2/C4 results), not
  discarded — it's this experiment's upper-bound reference, not a rejected
  thread.

## Key Findings
- **Phase 1 gate met (2026-07-31)**: `block_comp_mlp.py` implements C7a/C7/
  C8a/C8; all 5 spec-required unit tests pass (CPU, tiny model), no
  regression in `test_oracle_units.py`. One documented interpretation call:
  the block mask's selection score is the C3/C4/C5 residual score
  (`|u*(g-g_bar)|*col_norm`), not the `|i|*col_norm` score the spec's
  generic block-notation section literally writes — required for unit test
  1 (C7 at g=1 must bit-match C4). Flagged in spec.md/PR/code docstring;
  unconfirmed against the spec author's actual intent.
- PR #2 (Phase 1+2 code): https://github.com/aiha-choij/EfficientAI/pull/2
- **Phase 2 gate met (2026-07-31, CONFIRMED — llama3.2-3b-instruct)**:
  fully interpolated sharing-tax curve (in-family residual-score g=1
  anchor bracketed from both sides for every C7a point, no extrapolation
  left). Sharing tax (ΔPPL vs the g=1 anchor at matched achieved
  sparsity) is strongly **nonlinear in sparsity**: ΔPPL=4.64 (g=16) /
  4.77 (g=64) at sparsity≈0.20-0.24 (p=0.9), vs ΔPPL≈22.79 (g=16) / 21.28
  (g=64) at sparsity≈0.47-0.52 (p=0.7) — a 4.5-5x jump in absolute PPL
  cost for ~2x more sparsity. Consistent with the coactivation topic's
  overlap-collapse finding (adjacent-token neuron-set overlap only 3.15x
  chance at s=0.9). The higher-sparsity/higher-tax regime (ΔPPL≈21-23) is
  the harder test for Phase 3's C7/C8 compensation, not the milder one.
  Absolute C7a PPL (15.7-33.9) is far better than coactivation P3's
  neuron+token bare-block catastrophe (4.5k-24k) — token-only mask
  sharing (no neuron permutation) is survivable, not catastrophic.
  Journal: journal/2026-07-31_experiment-block-comp-phase2-3b-round1.md
  (full data table + interpolation method)
- **Phase 3 round 1: bug found + fixed (2026-07-31)**: 2 of 3 first-round
  jobs OOM'd — root-caused to two real bugs (GPU SVD on full projection
  matrices growing unbounded across 28 layers; building both C7 and
  C8/C8a factor sets unconditionally regardless of which the condition
  needs). Fixed (CPU SVD + `condition`-gated factor building, commit
  `80942a6`), unit tests re-verified, both jobs resubmitted.
- **C8a (diagnostic upper bound) result is striking**: g=16, p=0.7,
  r_sk=256 → sparsity 0.5397, **PPL 11.3815** — recovers ~99% of the
  sharing tax vs C7a's PPL 33.9006 at this regime (anchor ~11.1). Strong
  early evidence for H4/H5: most of the sharing tax is captured by a
  low-rank *gate* estimate alone (u, W_down still exact in this
  diagnostic condition).
- **C7 result — H4 Partial-go signature**: g=16, p=0.7, rank=512 →
  sparsity 0.5317, PPL 28.2506, recovery ≈**25%** (well under the 50% Go
  threshold, far below C8a's ~99%). Matches spec section 5's Partial-go
  pattern exactly.
- **Phase 3 3B round CONFIRMED — cross-g replication (2026-07-31)**: the
  g=16 recovery pattern (C7a 0%, C7 ~25%, C8 climbing 27%→42%→66% as
  r_sk goes d/32→d/16→d/8, C8a ~99%) replicates almost exactly at g=64
  (C7 ~18%, C8@d/8 ~67.5%, C8a ~98%). Four findings hold at both block
  sizes: (1) plain mean-gate compensation (C7) recovers only a minority
  (~18-25%); (2) a gate-sketch-only diagnostic with u/W_down exact (C8a)
  recovers nearly everything (~98-99%) even at small rank — **u-exactness,
  not gate-estimate quality, is the dominant lever** (a refinement of
  H5 beyond the spec's original framing); (3) the deployable form (C8)
  needs much more rank to approach that ceiling but DOES cross the spec's
  50% Go threshold at r_sk=d/8 at both g; (4) recovery is monotonic in
  r_sk, no saturation in the swept range.
  **Caveat, still unresolved**: spec's formal Go gate is defined for 8B
  at s≈0.9 — this is strong, consistent 3B dev-model preliminary
  evidence, not a formal verdict. Full table + all data in the journal.
  Journal now marked CONFIRMED for the 3B/g∈{16,64}/p=0.7 scope.

## Open Questions
- Is the block score for C7/C8 really meant to be the C3/C4/C5 residual
  score, or did the spec intend the `|i|*col_norm` (C1/C2) score for a
  different reason and the "C7=C4's math" line was only about the
  compensation formula, not the selection criterion? Implemented as
  residual-score (see Key Findings) because that's what unit test 1
  requires; not confirmed with the spec author.
- **Recovery rate has been computed as a PPL-ratio proxy this entire
  topic, not the spec's literal critical-sparsity-ratio formula**
  (spec.md §5 지표: recovery = Δcritical_sparsity ratios, where
  critical_sparsity = oracle spec's "normalized accuracy ≥0.99" max
  sparsity — an accuracy-sweep quantity, not a PPL-at-one-point
  quantity). Discovered while re-reading spec.md during the 8B p=0.5
  round's verdict check; not caught in Phase 2 or the 3B leg. Every
  recovery number reported so far (Phase 2 tax curve, 3B C7/C8a/C8,
  8B p=0.5) uses the proxy. Not retroactively recomputed (would require
  a full accuracy sweep per condition, done nowhere in this topic) —
  flagged here so the eventual formal verdict is read with this caveat
  rather than as a literal match to spec's formula.
- ~~Phase 2 round 1's C2-vs-C7a PPL gap mixed score family + sharing
  tax.~~ RESOLVED: in-family C3 g=1 anchor (score-family fix) +
  interpolation across a 4-point p-sweep (sparsity-matching fix) — see
  Key Findings for the final table.
- **Is it the up-sketch or the down-sketch that collapses C8's recovery
  from C8a's ~99% to ~27%?** No implemented condition isolates this (only
  c7a/c7/c8a/c8 exist) — a "u exact, gate+down sketched" diagnostic would
  answer it directly. Not fabricated, not yet built.
- **Small unexplained achieved-sparsity gap between C7a and C8a at the
  same nominal (g,p)**: C7a at g=16,p=0.7 measured s_block=0.5202; C8a at
  the identical (g,p,model,stats) measured s_block=0.5397. Both use the
  exact same mask-selection code path (`_resid_score` + `block_p_mask`),
  so this should be bit-identical in principle. Leading hypothesis (not
  confirmed): bf16 + per-GPU kernel nondeterminism across the two
  separate job runs (different physical GPUs) compounding over 28 layers
  of attention+MLP, nudging borderline neurons across the cumulative-mass
  cutoff in a score region that may be fairly flat/plateaued. Not
  re-verified with a same-GPU, same-process A/B check. Worth revisiting
  if the gap recurs at a size that would distort a recovery-rate
  computation (this round's ~2pp gap is small relative to the PPL
  differences being measured, but flagging rather than assuming away).

## Dead Ends
See the consolidated Dead Ends section below (qsub naming, GPU-SVD memory
growth) — none of this topic's actual research directions (block sharing,
compensation) are dead; see `coactivation-block-structure` gist for the
"bare block mask, no compensation" dead end this topic responds to.

## Next Experiments
Status: Phase 1 DONE, Phase 2 DONE (CONFIRMED), Phase 3 3B leg DONE
(CONFIRMED, cross-g). Remaining, in priority order:
1. **Phase 3 → 8B extension, DONE — formal GO verdict (2026-07-31)**: 8B
   oracle calibration + p-probe done. Hit the SVD memory-growth bug class
   twice more (OOM, then a 35-49min stall) before root-causing it
   properly as a missing `torch.no_grad()` (see Dead Ends) — fixed
   (commit `6b80dcf`). Two rounds run: p=0.5 (s_block~0.74-0.75 — short
   of spec's "s≈0.9" target; C7~78%/C8~89%/C8a~98% recovery) and p=0.3
   (s_block~0.887-0.898 — squarely in the target region; **C7 97.2% /
   C8 99.0% / C8a 99.7% recovery**, C7a's own PPL blows up to 1243.4 at
   this sparsity, consistent with the coactivation topic's catastrophic-
   collapse finding). **Spec §5's primary Go criterion (C8 recovery ≥50%)
   is met with a very wide margin at the actual target sparsity — formal
   verdict: GO.** Two caveats on record, not blocking the verdict: (a)
   recovery is a PPL-ratio proxy, not spec's literal critical-sparsity-
   ratio formula (see Open Questions); (b) spec's *alternate* criterion
   (ΔPPL≤+1.0 vs anchor) is NOT met (C8's ΔPPL is +12.13) — the huge
   recovery ratio reflects escaping C7a's catastrophic collapse, not
   closing the gap to dense-level absolute quality. H5 refines further:
   the u-exactness/gate-quality gap between C7/C8/C8a that was large at
   moderate sparsity (s~0.5-0.75) nearly vanishes under the ratio metric
   once C7a's baseline is this catastrophic — all three land within
   2.5pp of each other. Full tables in the journal:
   journal/2026-07-31_experiment-block-comp-phase3-8b.md
2. **§4 (local-loss-refit honesty corrections, C1/C2/M1) — DONE**,
   confirmed on both 3B and 8B (C1 fixes the s=0.5 "hurts" headline —
   was a ridge-prior artifact on both models — and strengthens s=0.9's
   Go on both). Tracked in the `local-loss-refit` topic's own
   gist/journal; PR #3 open.
3. **Phase 4 (P3′) — SCOPED, implementation not started**: combine with
   `coactivation-block-structure` P2's PPMI neuron-cluster permutation.
   Design + scoping done (journal/2026-07-31_init-phase4-p3prime.md):
   P3′ = replace P3's plain "zero the dropped neuron-blocks" masking with
   block_comp's C7 (mean-gate)/C8 (sketch) compensation on those dropped
   blocks, using the existing partition file's per-neuron cluster
   assignments (fetched from `a6000-4` to
   `~/workspace/analysis/llama2_p3_partitions_s09.pt`, confirmed:
   llama2-7b, 32 layers, intermediate_size 11008, B∈{64,256},
   sparsity=0.9 — conveniently matches Phase 3's own target regime).
   LLaMA2-7B only for the first pass (dense anchor 5.4738 known,
   partition is model-specific). Needs real new code (2D token-block ×
   neuron-block score aggregation, compensation applied only to dropped
   blocks, new unit tests) before any GPU job — not started yet. With
   Phase 3's 8B gate now formally met, this is the remaining piece of the
   request's Main Thread A/B.
C9 (overflow hybrid) is explicitly NOT implemented — Phase 3 landed on
GO, not Partial-go/No-go, so C9 is not promoted per spec's own rule.

## Active Jobs
- Phase 2 (calibration + 4 rounds of eval): all DONE. One transient
  failure (`bc-c3-g1-p095`, CUDA OOM, shared-cluster contention) worked
  around by submitting a more useful point instead of a blind retry —
  see journal.
- Phase 3 round 1: all 4 jobs (c7a already had it; c7/c8a/c8) DONE — see
  Key Findings for the full recovery table.
- Phase 3 3B round (g∈{16,64}, p=0.7): all 7 jobs done, CONFIRMED (see
  Key Findings + journal for the full table).
- Phase 3 8B round (p=0.5): all done (see Key Findings for the recovery
  table). C7/C8a/C8 stalled once on pre-fix code (35-49min, no
  progress), killed via `runs -k`, resubmitted post-fix and landed clean.
- Phase 3 8B round (p=0.3, targeting s_block≈0.9): all done — this is the
  round the formal Go verdict is based on (see Key Findings).
- Phase 3 8B leg: DONE. No jobs outstanding for this topic's main thread
  until Phase 4 kicks off.

## Dead Ends
- **qsub job names must not contain `.` (2026-07-31)**: named the first
  round of eval jobs with the literal p-value (`bc-c2-g1-p0.9`). The
  dispatcher passes the job ID straight into a tmux session name
  (`qcom-<id>`), and tmux rejects `.` in session names ("bad session
  name") — every dispatch attempt failed silently into an infinite ~20s
  retry loop (no GPU ever touched, but ~54 min of wasted dispatcher
  cycles before caught during a periodic check). Fix: use `p09`/`p07`
  (no dot) in job names; only the qsub `-n` NAME is affected, not the
  `--p` argument value passed to the script. Applies to any future job
  name built from a float knob (p, lambda, etc.) in this or other topics.
- **Per-layer SVD "OOM"/"hang" was never about GPU vs CPU — missing
  torch.no_grad() was the real bug, misdiagnosed twice (2026-07-31)**:
  first hit in the gate/up/down sketch path (Phase 3 round 1, 3B) as a
  CUDA OOM, "fixed" by moving `_svd_factors` to CPU; recurred in C7's
  comp_lr path at 8B scale as another OOM, "fixed" the same way with a
  local CPU-SVD `_build_comp_lr_factors`. Both fixes worked (unblocked
  the immediate crash) but were treating a symptom: `_svd_factors` and
  friends are called with model weight tensors, which have
  `requires_grad=True` by default (`.eval()` doesn't change that), so
  every layer's SVD call built and RETAINED an autograd graph that
  nothing ever freed — unbounded memory growth regardless of device. The
  CPU move just traded a fast GPU OOM for slow CPU compute, which then
  surfaced as a *third* incident: 3 concurrent 8B sweep jobs stalled
  35-49 minutes with zero eval-loop progress (CPU SVD of matrices up to
  14336x4096, contended across 3 simultaneous CPU-heavy jobs on one
  host). Properly fixed by reverting to GPU SVD and adding
  `@torch.no_grad()` to `attach_block_factors_inplace` (commit
  `6b80dcf`) — the real lesson: any future per-layer SVD (or other
  no-backward-needed tensor op) on model weights in this codebase MUST
  be wrapped in `torch.no_grad()` explicitly; CPU-vs-GPU placement was
  never the actual fix.
- **Killed jobs are recoverable via `runs -k <ID>`** — corrects an
  earlier assumption in this topic's journal that a running remote job
  had no safe kill mechanism. `runs -k` sends `tmux kill-session -t
  qcom-<id>` on the assigned host; the job then shows up in
  `queue/failed/` with its original `meta`/`cmd` intact, so the exact
  command can be recovered and resubmitted. Used to clear 3 stalled
  pre-fix 8B jobs before resubmitting them on the no_grad-fixed code.

## Pointers
- Full spec (verbatim, Korean, authoritative): `spec.md` in this topic —
  the gateway cannot read the host wiki `block-sparse-compensation-spec.md`,
  so this is the only surviving copy (quoted inside the originating agent
  request, preserved here in full including §§0,1,2,3,4,5 operating rules).
- Anchor infra: `oracle-residual-sparsity/spec.md` (C0–C6 notation this
  topic's §1 extends), code `larosa/inference/oracle_mlp.py`,
  `modeling_llama_larosa.py` `sparse_mode='oracle'` wiring.
- Block-aggregation precedent already in the repo:
  `larosa/inference/refit_mlp.py`'s `aggregate_block_score`/`block_mask`
  implement exactly the S_j(T) block-score aggregation this spec's §1
  requires (fp32, sequence-boundary-respecting, padding-excluded) — reuse
  rather than reimplement.
- coactivation P3 partitions (Phase 4 input):
  `a6000-4:~/workspace/analysis/llama2_p3_partitions_s09.pt` (LLaMA2-7B,
  PPMI spectral + balanced k-means, B ∈ {64,256}, + random control),
  collected by `larosa/scripts/p3_collect_cluster_all.py`.
- Requesting agent request dir (host-side tracking):
  `requests/active/20260731-162935-block-compensation/` (report.md has the
  running judgment log; this repo's journal/gist is the durable record).
