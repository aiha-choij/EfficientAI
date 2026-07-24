# Experiment: larosa-llama2-topk-overlap (analysis)

Status: PENDING
Date: 2026-07-24

Note: the topic is closed (done); this is an analysis card added post-closure
at the user's request — it does not reopen the topic. Findings feed both the
oracle-residual-sparsity thread (same sparsification point i) and the future
RB-Sparse block-shared-mask question.

## Hypothesis tested
Question, not a directional hypothesis: in topk_intermediate inference on
LLaMA2-7B, how much do the per-token Top-K neuron index sets of i = u ⊙ g
agree across tokens, and does rising sparsity (s = 0.5 → 0.7 → 0.9) concentrate
selection onto a shared "important neuron" pool (overlap ≫ chance, high
always-on fraction, survivor sets nested across s) or diversify it
(overlap ≈ chance K/d)?

## What we're testing over alternatives
Measured on the actually-sparsified forward (selection at layer L reflects
upstream masking), not on dense activations — the user asked about real
inference behavior. Single model (LLaMA2-7B) since this is a mechanism probe,
not a universality claim.

## Prior art check
- This topic's PPL card (2026-07-22): topk_intermediate confirmed
  (+0.047/+0.256/+2.635 at s=0.5/0.7/0.9); eval-log mlp/attn labels swapped
  (upstream bug) — the analysis script computes its own stats, unaffected.
- Old larosa-repro gist listed "cross-token top-k index agreement in the
  rotated basis" as the RB-Sparse first step; this card answers the
  original-basis version.
- Concurrent oracle-residual-sparsity topic sparsifies the same i via oracle
  top-p on a mean-gate residual; C1 anchors re-measured there
  (5.5216/5.7284/8.1096, a6000-2/sdpa).

## Expected outcome
Descriptive analysis — success is a complete, internally consistent table:
per-layer overlap at token distances 1/4/16/64/256 and random pairs vs chance
K/d, selection-frequency structure (always-on / rare / never, Gini), union
coverage per 2048-token sequence, and cross-s top-frequency-set overlap.
Sanity: overlap values must lie in [chance, 1]; K_eff ≈ (1−s)·11008.
Failure: metrics inconsistent with those bounds (script bug) or job error.

## Reproducibility
- **Git tag**: `exp/2026-07-24_larosa-llama2-topk-overlap` (repo
  `aiha-choij/EfficientAI`, commit eb2caa1; analysis script added in cf90f5e)
- **Job ID**: `20260724-173030-larosa-llama2-topk-overlap` (4th submission;
  see Notes)
- **Assigned host/GPU**: a6000-4 (pinned via -H), GPU 0 (fully idle 48GB)
- **Command**: `bash -c "mkdir -p /home/choij/workspace/analysis && /home/choij/workspace/venv-larosa/bin/python scripts/analyze_topk_overlap.py --model_name /raid/LLM/llama2-7b --sparsities 0.5,0.7,0.9 --nsamples 32 --attn sdpa --out /home/choij/workspace/analysis/llama2_topk_overlap.pt"`
  (cwd `/home/choij/workspace/repos/EfficientAI/larosa`; qsub `-g 1 -m 40`)
- **Config path**: n/a — parameters as script args
- **Key parameters**: sparse_mode=topk_intermediate; s ∈ {0.5, 0.7, 0.9} set by
  reassigning mlp.sparse_level_h2 between runs (single model load); 32 × 2048
  wikitext-2 test tokens; selection read from down_proj input hook (!= 0);
  distances 1/4/16/64/256 + one random permutation pairing per batch;
  survivor analysis top 10% most-frequent neurons; bf16; attn=sdpa
  (backend/arch effects ~1e-3 PPL per the oracle topic's phase-3 gate; index
  selection is a magnitude argmax, insensitive at that scale)
- **Key deps**: python 3.10, torch 2.6.0+cu124, transformers 4.46.3 (venv
  `~/workspace/venv-larosa`, no flash-attn)
- **Model**: `/raid/LLM/llama2-7b` (a6000-4 local copy); output artifact
  `/home/choij/workspace/analysis/llama2_topk_overlap.pt` (on a6000-4)

## Notes (submission history)
- Attempt 1 (`...-171801`, a100-40-2 `-m 30`): could never dispatch — all four
  A100s ~24GiB free; cancelled from pending.
- Attempt 2 (`...-172004`, a100-40-2 `-m 20`): dispatched to PCI GPU 1 but the
  co-tenant process grew to 34.3GiB between probe and model load → CUDA OOM.
- Attempt 3 (`...-172313`, a6000-2 `-m 22`): never dispatched — dispatcher
  requires util ≤ 10% and all three a6000-2 GPUs were compute-busy; cancelled.
- Attempt 4: replicated the a6000-2 env to a6000-4 (tar-relay via gateway:
  venv-larosa 5.4G + repo, identical paths; torch 2.6.0+cu124 verified) and
  submitted to its fully idle GPU 0.

### Results

### Interpretation
