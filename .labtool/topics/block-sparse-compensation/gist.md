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
(no accuracy findings yet — first Phase 2 PPL numbers are queued, not back)
- **Phase 1 gate met (2026-07-31)**: `block_comp_mlp.py` implements C7a/C7/
  C8a/C8; all 5 spec-required unit tests pass (CPU, tiny model), no
  regression in `test_oracle_units.py`. One documented interpretation call:
  the block mask's selection score is the C3/C4/C5 residual score
  (`|u*(g-g_bar)|*col_norm`), not the `|i|*col_norm` score the spec's
  generic block-notation section literally writes — required for unit test
  1 (C7 at g=1 must bit-match C4). Flagged in spec.md/PR/code docstring;
  unconfirmed against the spec author's actual intent.
- PR #2 (Phase 1+2 code): https://github.com/aiha-choij/EfficientAI/pull/2

## Open Questions
- Is the block score for C7/C8 really meant to be the C3/C4/C5 residual
  score, or did the spec intend the `|i|*col_norm` (C1/C2) score for a
  different reason and the "C7=C4's math" line was only about the
  compensation formula, not the selection criterion? Implemented as
  residual-score (see Key Findings) because that's what unit test 1
  requires; not confirmed with the spec author.

## Dead Ends
(none yet in this topic; see `coactivation-block-structure` gist for the
"bare block mask, no compensation" dead end this topic responds to)

## Open Questions
(record here, don't invent answers, if any spec ambiguity surfaces during
implementation)

## Next Experiments
Execution order per spec §5 operating rules:
1. Phase 1 — extend `oracle_mlp.py` (or a sibling module, following the
   `refit_mlp.py` precedent of not touching the existing oracle file) with
   `block_size` + conditions C7a/C7/C8a/C8, plus the 5 required unit tests
   (g=1 reduction to C4, p=1 identity, C8 full-rank == dense, block-sharing
   assertion, masking equivalence). CPU-only, no GPU job.
2. Phase 2 — sharing-tax curve (3B): C7a vs the existing g=1 C2 anchor,
   g ∈ {16, 64} × p grid.
3. Phase 3 — C7/C8a/C8 sweep (3B → 8B), r_sk ∈ {d/32, d/16, d/8} →
   recovery-rate table (model × g × condition).
4. Phase 4 (P3′) — combine with `coactivation-block-structure` P2's PPMI
   neuron-cluster permutation: rerun C7/C8 on top of the clustered blocks
   from `a6000-4:~/workspace/analysis/llama2_p3_partitions_s09.pt`
   (LLaMA2-7B only — model mismatch with this topic's main models, 3B/8B;
   plan is to run Phase 4 on llama2-7b first (cheap, reuses partitions,
   dense anchor 5.4738 known) and only extend to 8B if the direction looks
   worthwhile).
C9 (overflow hybrid) is explicitly NOT implemented yet — only promoted if
Phase 3's Go/Partial-go/No-go gate (spec §5) lands on Partial-go/No-go.

## Active Jobs
- `block-comp-calib-3b` (050-20260731-164842): DONE — oracle-format g_bar/
  col_norm calibration for `/raid/LLM/llama3.2-3b-instruct`
  (wikitext103, n=512, seqlen=2048), saved to
  `~/workspace/oracle/llama3.2-3b-instruct/stats/wikitext103` (28 layers).
  Prerequisite for every block_comp condition (all use the residual score).
- Phase 2 sharing-tax-curve jobs (round 1, resubmitted after a naming
  bug — see Dead Ends): C2 g=1 anchor at p∈{0.7,0.9}, C7a at (g=16,p=0.9)
  and (g=64,p=0.9), llama3.2-3b-instruct. IDs `050-20260731-1746{51,54,59}`
  + `050-20260731-174703` (`bc-c2-g1-p09`, `bc-c2-g1-p07`,
  `bc-c7a-g16-p09`, `bc-c7a-g64-p09`); one running, three queued as of
  2026-07-31 17:47.

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
