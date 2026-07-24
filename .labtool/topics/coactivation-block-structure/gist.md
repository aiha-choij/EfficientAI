# coactivation-block-structure — Can neuron permutation make group masks block-structured?

## Status
active — started 2026-07-24 from spec
`~/Workspace/research-wiki/plans/coactivation-block-structure-spec.md`
(derived from the 2026-07-24 group-sparsity discussion, Idea C; parent doc:
research-wiki `plans/predictor-group-sparsity-research-direction.md`).

## Notation (shared with larosa-intermediate-sparsity)
```
FFN(x) = W_d i,  i = u ⊙ g,  u = W_u x,  g = σ(W_g x)
d = intermediate size (LLaMA2-7B: 11008),  L = 32 layers
S_t ⊂ {1..d} : Top-K surviving-neuron set of token t, K = ⌊(1−s)·d⌋
C(δ) = E_t[|S_t ∩ S_{t+δ}|] / K_eff   (containment overlap)
chance = K_eff / d
A_jj′  = E_t[1[j ∈ S_t]·1[j′ ∈ S_t]]          (same-token co-activation)
A^g_jj′ = E_{|t−t′|<g}[1[j ∈ S_t]·1[j′ ∈ S_t′]] (window-g co-activation)
PMI_jj′ = log(A_jj′ / (f_j·f_j′)),  f_j = selection frequency
```

## Hypothesis
Sharing intermediate Top-K masks across token groups (g = 16/32/64) pays a
heavy tax because neuron selection is dominantly token-dependent (prior
measurement: adjacent overlap only 3.15× chance at s=0.9, ~68% disagreement).
Permutation is a free, exactly function-preserving degree of freedom for
i = u ⊙ g (only monomial transforms commute with the elementwise structure —
parent doc §3.2). Hypothesis: reordering neurons into blocks by co-activation
statistics (PMI / Jaccard clustering) makes the group mask a block-selection
problem, absorbing union inflation at the block level.

Strong null to beat: if clustered blocks give < ~1.3× gain over random
permutation blocks on the P2 metrics, the permutation axis is rejected and
priority moves to shared-backbone + residual (Idea A) / learned predictor
(Idea D).

## Plan (from spec §3)
- **P1** — collect co-activation statistics A, A^g, f on sample layers
  {0, 8, 16, 24, 31} at s=0.9 (layer 31 mandatory: the exceptional layer),
  32 × 2048 tokens, wikitext-2, actual sparsified forward.
- **P2** — balanced clustering (spectral / METIS / balanced k-means; cf.
  MoEfication) on PMI, block sizes B ∈ {64, 128, 256}; evaluate
  within-block co-activation mass, block-level union ratio, block coverage
  @ budget — all reported as multiples over random permutation.
- **P3** (only if P2 promising) — block-mask oracle PPL with gauge-fixed
  score Σ‖W_d[:,j]‖·|i_j|, vs dense / per-token top-K / random-permutation
  blocks. Sweep s ∈ {0.7, 0.9} × g ∈ {16, 64} × B ∈ {64, 256} (selected).
- **P0** (optional, alongside P1) — group-tax curve: union inflation
  |∪_{t∈group} S_t|/K and group-shared top-K PPL.

## Key Findings
(none yet)

## Dead Ends
(none yet)

## Open Questions
- Which normalization (PMI vs Jaccard) is the better clustering substrate?
  Decide after seeing P1 distributions.
- Window statistic A^g: which g values to accumulate (16/32/64 all, or 64
  only, since smaller windows are sub-sums)?
- P3 success criterion is provisional (e.g. ΔPPL ≤ +1.0 vs per-token at
  s=0.9) — finalize after first numbers, per house convention.

## Next Experiments
1. **P1 — co-activation statistics collection** (this session's card):
   hook on down_proj input, sel [B,T,d] bool → accumulate selᵀ·sel in fp32
   on GPU (11008² ≈ 484 MB/layer), layers {0,8,16,24,31}, s=0.9,
   32 × 2048 wikitext-2 tokens; save .pt to ~/workspace/analysis/.
2. P2 clustering — CPU/laptop-scale, needs only P1 outputs.

## Active Jobs
- (none)

## Boundaries / coordination
- RB-Sparse (Dowon Kim) owns block-shared masks *on top of rotation*; this
  topic is the rotation-free original-basis permutation axis — no overlap.
  The §2 tax measurements are relevant to both threads (worth sharing).
- Weight-aware score ‖W_d[:,j]‖·|i_j| is shared material with
  oracle-residual-sparsity H2 (P3 uses the same definition).
- The parent doc's §3.2 *theory framing* is on hold with the advisor; this
  topic proceeds as an experimental axis, independent of that framing.

## Pointers
- Spec: `~/Workspace/research-wiki/plans/coactivation-block-structure-spec.md`
  (mac-local; self-contained).
- Prior measurements: topic `larosa-intermediate-sparsity` gist Key Findings
  ("Cross-token selection overlap…", 2026-07-24) + journal
  `2026-07-24_experiment-larosa-llama2-topk-overlap.md`; report artifact
  https://claude.ai/code/artifact/c73a7f23-2ac7-4b27-9857-6c21a60d184f
- PPL anchors (trusted): dense 5.4736; per-token topk_intermediate
  s=0.5/0.7/0.9 → 5.5210/5.7296/8.1083 (a6000-2 sdpa remeasure
  5.5216/5.7284/8.1096; backend delta ~1e-3).
- Code: `larosa/inference/modeling_llama_larosa.py`
  (`sparse_mode='topk_intermediate'`, validated s=0 ≡ dense bitwise);
  hook pattern: `larosa/scripts/analyze_topk_overlap.py`.
- Execution hosts (2026-07-24): a6000-4 (venv ~/workspace/venv-larosa,
  sdpa only, model /raid/LLM/llama2-7b, repo synced via gateway scp/tar —
  no GitHub fetch); a100-40-2 gateway (conda `larosa`, flash-attn, A100s
  often busy).
- Known traps: qsub workdir must be an absolute path (literal `~` fails
  silently); dispatcher only places on fully-idle GPUs (util ≤ 10%);
  `runs` ELAPSED includes queue wait; eval_ppl.py mlp/attn sparsity labels
  are swapped (upstream bug) — prefer direct hooks.
