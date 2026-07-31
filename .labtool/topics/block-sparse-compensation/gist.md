# block-sparse-compensation — Can input-dependent compensation recover the sharing tax of a block-shared mask?

## Status
active — started 2026-07-31 from spec (full spec preserved verbatim in
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
(none yet in this topic; see `coactivation-block-structure` gist for the
"bare block mask, no compensation" dead end this topic responds to)

## Next Experiments
Status: Phase 1 DONE, Phase 2 DONE (CONFIRMED), Phase 3 3B leg DONE
(CONFIRMED, cross-g). Remaining, in priority order:
1. **Phase 3 → 8B extension** (needed for spec's actual Go gate: 8B,
   s≈0.9): 8B oracle calibration DONE (`bc-calib-8b`,
   `~/workspace/oracle/Llama-3.1-8B/stats/wikitext103`, 32 layers, no
   anomaly). Correction to an earlier note here: 3B's p-vs-sparsity
   curve needs a LOWER p (further from 1) to reach HIGHER sparsity
   (p=0.9->sp=0.51, p=0.7->sp=0.76 -- the direction is p down = sparsity
   up), so reaching s≈0.9 needs p well below 0.7, not above 0.9. Queued
   2 probe jobs on 8B (`bc-c3-g1-8b-p05`, `bc-c3-g1-8b-p03`, oracle C3
   condition, reusing the 8B calibration) to find which p lands near
   sparsity 0.9 before committing to the full C7a/C7/C8a/C8 sweep. Not
   landed yet. Only after this can Go/Partial-go/No-go (spec §5) be
   formally declared — current 3B result is Go-crossing but preliminary.
2. **§4 (local-loss-refit honesty corrections, C1/C2/M1)** — IN PROGRESS
   on branch `auto/refit-honesty-corrections` (off `auto/local-loss-
   refit`): C1 ridge-anchor fix coded + unit-tested, M1 log-PPL
   correction done (headline revised 37%/60%->~12%/~22%), recalibration
   + dense-anchor GPU jobs queued. Tracked in the `local-loss-refit`
   topic's own gist/journal, not here — see there for results.
3. Phase 4 (P3′) — combine with `coactivation-block-structure` P2's PPMI
   neuron-cluster permutation: rerun C7/C8 on top of the clustered blocks
   from `a6000-4:~/workspace/analysis/llama2_p3_partitions_s09.pt`
   (LLaMA2-7B only — model mismatch with this topic's main models, 3B/8B;
   plan is to run Phase 4 on llama2-7b first (cheap, reuses partitions,
   dense anchor 5.4738 known) and only extend to 8B if the direction looks
   worthwhile).
C9 (overflow hybrid) is explicitly NOT implemented yet — only promoted if
Phase 3's Go/Partial-go/No-go gate (spec §5) lands on Partial-go/No-go.

## Active Jobs
- Phase 2 (calibration + 4 rounds of eval): all DONE. One transient
  failure (`bc-c3-g1-p095`, CUDA OOM, shared-cluster contention) worked
  around by submitting a more useful point instead of a blind retry —
  see journal.
- Phase 3 round 1: all 4 jobs (c7a already had it; c7/c8a/c8) DONE — see
  Key Findings for the full recovery table.
- Phase 3 3B round (g∈{16,64}, p=0.7): all 7 jobs done, CONFIRMED (see
  Key Findings + journal for the full table). No jobs currently running.

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
