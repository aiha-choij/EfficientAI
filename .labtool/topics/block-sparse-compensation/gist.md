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
(none yet — Phase 1 implementation in progress)

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
(none yet)

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
