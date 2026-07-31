# Init: Phase 4 (P3′) — combine block-comp compensation with coactivation P2 neuron-cluster permutation

Date: 2026-07-31

## Why now
Phase 3's spec §5 gate is now formally GO (8B, g=16, s≈0.9 — see
`2026-07-31_experiment-block-comp-phase3-8b.md`). Phase 4 is the last
piece of the request's Main Thread A/B: does combining block_comp's
token-block compensation (C7/C8) with `coactivation-block-structure`'s
P2 PPMI-clustered neuron-block permutation do better than either axis
alone? Motivation restated from spec.md §3: P3 (no compensation) showed
clustered neuron blocks beat random 2-3x but both are still
catastrophic (PPL 4.5k-24k vs ~8 anchor at s=0.9) — the open question is
whether adding block_comp's per-token-block compensation on top of the
clustered partition closes that gap the way it closed the token-only
sharing tax in Phase 3.

## What P3′ actually requires (not yet implemented)
Read both source scripts to scope this:
- `p3_collect_cluster_all.py`: produces
  `partitions[layer][B] = {'clustered': assign[d], 'random': assign[d]}`
  — a per-neuron cluster id in `[0, d/B)`, one tensor per (layer, B).
  This *is* P2's validated clustering (PPMI → spectral embed → balanced
  k-means); there's no separate "P2 script", the partition file is the
  P2 artifact.
- `p3_block_ppl.py`'s `patched_mlp_forward` (block mode): builds a
  one-hot `M [d, nb]` from the assignment, aggregates a per-token
  |i|*colnorm score into token-groups of size `g` (`gs = score summed
  over g`), projects into neuron-block space (`bscore = gs @ M`), keeps
  the top-`m` neuron blocks per token-group (`m = round(K/B)`), and
  masks every token in the group to the union of kept blocks' neurons.
  This is pure masking — no compensation on the dropped blocks, which is
  exactly the "no comp" leg block_comp's C7a already represents (its
  token-only analog).

P3′ = replace P3's "zero the dropped blocks" with block_comp's C7
(mean-gate ḡ) or C8 (per-token gate/up/down sketch) compensation,
applied only to the neurons living in dropped blocks. Concretely:

1. **Scoring convention decision (interpretation call, same category as
   Phase 1's block-score choice)**: block_comp uses the C3/C4/C5 residual
   score (`|u*(g-g_bar)|*col_norm`); P3 uses a plain `|i|*col_norm` score.
   Need one convention for P3′'s neuron-block selection — reusing
   block_comp's residual score keeps this consistent with everything
   measured in Phases 2-3, so that's the default choice unless there's a
   reason not to. Flag as unconfirmed either way, per this topic's
   existing precedent.
2. **New selection logic**: token-block aggregation (already in
   `block_comp_mlp.py`'s `aggregate_block_score`) THEN neuron-block
   projection (`gs @ M`, new — borrowed from `p3_block_ppl.py`) THEN
   top-m neuron-BLOCK selection (new — quantized to blocks of size B,
   not top-p over individual neurons like block_comp's current
   `block_p_mask`).
3. **Compensation on dropped blocks only**: kept neuron-blocks compute
   exactly (gate/up/down all exact); dropped neuron-blocks get C7's
   mean-gate or C8's sketch treatment — a genuinely new code path, since
   block_comp's existing C7/C8 apply compensation to whichever neurons
   the *unstructured* top-p mask drops, not a block-quantized set.
4. **New unit tests needed** (not yet written): B=d (single neuron-block
   = everything kept or everything scored as one unit) should reduce to
   plain g-only C7a/C7/C8 (sanity check that the neuron-block axis is a
   true generalization, not a parallel implementation); g=1 with neuron
   blocks should match a per-token neuron-block-masked oracle; block
   sharing assertions for both axes independently.

## Model/scope decision
LLaMA2-7B only for the first pass (spec's own recommendation, §3 in this
topic's Main Thread B): the existing partition file
(`a6000-4:~/workspace/analysis/llama2_p3_partitions_s09.pt`) is
LLaMA2-7B-specific (PPMI/clustering computed on that model's actual
activations — not portable to 3B/8B without recollecting). Model
confirmed present locally at `/raid/LLM/llama2-7b`. Dense anchor 5.4738
already known from the coactivation topic (not re-measured here).
Extend to 8B only if the LLaMA2-7B result looks directionally
worthwhile (recollecting P1/P2 on 8B is expensive — a new full
coactivation pass, not just an eval script).

## Starting phase
Design/scoping done (this card). Partition file fetched from `a6000-4`
to the gateway (`/home/choij/workspace/analysis/llama2_p3_partitions_s09.pt`,
via `scp`) and inspected: `sparsity=0.9` (conveniently matches Phase 3's
own target regime), `model=/raid/LLM/llama2-7b`, `block_sizes=[64,256]`,
32 layers, per-layer assignment tensors of shape `[11008]` (llama2-7b's
intermediate_size) with 172 clusters at B=64 and 43 clusters at B=256 —
confirms the structure assumed above, no surprises.

Next: implement the new neuron-block selection + compensation path in
`block_comp_mlp.py` (likely as new conditions or a `neuron_partition`
argument on existing C7/C8), with unit tests before any GPU eval job,
per this topic's established discipline. Not started yet — this is a
real implementation task (new scoring path, new masking-with-
compensation-on-dropped-blocks logic, new tests), not a quick script.

## Implementation (2026-08-01)
Implemented as designed above, with one simplification worth noting
explicitly: the existing C7a/C7/C8a/C8 compensation formulas in
`block_comp_mlp_forward` needed **zero changes** — they only depend on
`m_bool` being a per-neuron/per-token boolean (kept vs dropped), not on
how that boolean was constructed. So the whole P3′ combination reduces
to a new *mask-construction* path, isolated to three new functions:
- `neuron_block_topm_mask(block_score, M, m)`: projects an already
  token-block-aggregated score into neuron-block space via `score @ M`
  (M = one-hot [D, nb_neuron] partition), keeps the top-m neuron-blocks
  by aggregate score, broadcasts back via `keep_b @ M.T` — same
  construction as `p3_block_ppl.py`'s block branch.
- `block_p3_mask(score, g, M, m, seq_mask)`: composes this with the
  EXISTING `aggregate_block_score` (token-block axis, unchanged from
  Phase 1) — the spec's "2D tile" is just these two aggregations
  composed, not a new tiling mechanism.
- `build_neuron_partition_onehots(partition_path, B, device, layers)`:
  loads a `p3_collect_cluster_all.py` file and builds the per-layer M.

Wired into `set_condition` via optional `neuron_partitions`/`neuron_m`
kwargs (default None = Phase 1-3's exact existing behavior, unchanged)
and into `block_comp_mlp_forward` via a `getattr(mlp, "blk_neuron_M",
None)` branch.

**4 new unit tests, all passing** (no regression in the existing 10
block-comp tests or `test_oracle_units.py`):
1. B=1 (each neuron its own block) reduces exactly to
   `oracle_mlp.top_count_mask` (bit-identical, continuous random score →
   no ties).
2. neuron_m = nb_neuron (all blocks kept) → C7a output == dense.
3. Neuron-block sharing: mask constant within every neuron-block
   (synthetic partition, B=4, 6 blocks).
4. **The key integration test**: C8 at full rank + a P3′ neuron-block
   mask (some blocks dropped, at both g=16 and g=64) == dense exactly —
   confirms the compensation formulas compose correctly with
   block-quantized (not just unstructured top-p) mask selection.

New eval script `scripts/block_comp/02_eval_p3prime.py` mirrors
coactivation-block-structure's own P3 budget formula (`K =
round((1-sparsity)*d)`, `m = round(K/B)`) exactly, so PPL numbers here
are directly comparable to that topic's existing clustered/random-block
no-compensation results at the same (B, g, sparsity) — the natural
baseline P3′ needs to beat.

Committed `a6f4627`, pushed to `auto/block-sparse-compensation`.

**First validation batch submitted** (LLaMA2-7B, g=16, B=64, sparsity=
0.9 — matching the partition file's own collection sparsity and P3's
finest tested granularity): `p3prime-c7a-g16-B64` (control, no
compensation — NOT expected to reproduce P3's own clustered-block
numbers bit-exactly, since block_comp's residual score differs from
P3's `|i|*col_norm` score, an interpretation choice already on record
from Phase 1; only meant to sanity-check the pipeline lands in the same
catastrophic ballpark P3 found), `p3prime-c7-g16-B64-r688` (rank=d/16),
`p3prime-c8a-g16-B64-rsk344` (r_sk=d/32), `p3prime-c8-g16-B64-rsk1376`
(r_sk=d/8) — rank fractions chosen to match Phase 3's own convention.
All 4 queued/running, none landed yet.
