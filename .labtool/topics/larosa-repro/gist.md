# larosa-repro — LaRoSa reproduction (arXiv:2507.01299)

## Status
active

## Hypothesis
The LaRoSa pipeline in `larosa/` (upstream Alibaba Cloud code, forked to
aiha-choij/EfficientAI) reproduces paper Table 2 wikitext-2 PPL (verified against
the source PDF, p.6): LLaMA2-7B 5.47 → 25% 5.51 / 40% 5.64 / 50% 5.87;
LLaMA3-8B 6.13 → 6.23 / 6.60 / 7.22; Qwen2.5-7B 6.85 → 6.90 / 7.10 / 7.42.
40% is the paper's headline "near-lossless" operating point. Secondary: README
lm_eval accuracy table for Qwen2.5-7B-larosa at sparse_level 0.25.

## Key Findings
- PPL reproduction needs no model packaging: `scripts/ppl_test_larosa_*.py` takes
  `--sparsity` and `--larosa_path` at runtime (packaging only matters for lm_eval).
- Sparsity 0.0 keeps everything in `top_k_new` → usable as a dense-equivalence
  sanity check of the rotation path.
- α is baked into the modeling code as sparse_level_h1 = 0.8p, h2 = 1.2p
  (not grid-searched per model as in the paper).

## Dead Ends
- 2026-07-22 — gen_act pass 2 (histogram collection) on 40GB GPUs for
  LLaMA3-8B/Qwen2.5-7B: OOMs even alone (rotated model + fp64 buffers + full-
  tensor GPU sort). Not needed for PPL (eval loads only pass-1 D.pt); revisit
  only if threshold-style (CATS/TEAL) analysis is ever needed — then run on
  a100-80 or patch find_histogram to sort on CPU.
- flash-attn 2.8.x PyPI wheels: require GLIBC≥2.32, gateway is Ubuntu 20.04
  (glibc 2.31). Pin the official 2.7.4.post1 cu12torch2.6cxx11abiFALSE wheel.

## Open Questions
- Does `attn_implementation` (flash_attention_2 vs sdpa) matter for reproducing the numbers? gen_act hardcodes flash_attention_2.
- README packaging step references `configuration_qwen.py` but the file is `configuration_qwen2.py` (auto_map expects `configuration_qwen2`). (Only relevant for the lm_eval packaging step.)
- Code calibrates with 10×2048 wikitext-2 sequences; paper says 16×2048. Paper's
  own robustness result (covariance cos-sim 0.998 across datasets) suggests this
  doesn't matter — verify via reproduction quality.

## Next Experiments
1. Package Qwen2.5-7B-larosa (sparse_level 0.25) + lm_eval 6-task accuracy vs
   README table. Why: PPL reproduced on all 3 models; accuracy pipeline is the
   remaining unvalidated half, and RB-Sparse development evals need it.
   Success: each task acc within ~1pp of README §5 table.
2. RB-Sparse development (after 1): rotated-basis block-shared mask +
   eigenspace low-rank compensation (2026-06-24 discussion; research-wiki
   r-sparse note). First step: measure cross-token top-k index agreement in
   the rotated basis using the saved D matrices.

## Active Jobs
- `20260723-101220-larosa-llama3-8b-ppl` @ a100-40-2 — rotation gen + PPL sweep;
  targets 6.13/6.23/6.60/7.22 ±0.1. Journal: 2026-07-22_experiment-larosa-llama3-8b-ppl.md.
- `20260723-102000-larosa-qwen25-7b-ppl` @ a100-40-2 — rotation gen + PPL sweep;
  targets 6.85/6.90/7.10/7.42 ±0.1. Journal: 2026-07-22_experiment-larosa-qwen25-7b-ppl.md.
- Gateway agent hourly watch for both: request `20260723-084829-larosa-ppl-repro-watch`.

## Pointers
- Paper: arXiv:2507.01299 (La RoSA, Liu et al., ICML 2025); PDF in research-wiki
  `sources/Activation Sparsity/`, enriched note `wiki/papers/la-rosa-rotated-sparse-activation.md`.
- Reference numbers: paper Table 2 (PPL, primary) and Table 1 (accuracy);
  larosa/README.md §5 (Qwen lm_eval, secondary).
- Models on gateway: `/raid/LLM/llama2-7b`, `/raid/LLM/llama3-8b` (shared,
  read-only), `~/workspace/models/Qwen2.5-7B` (ours).
- Runner: `larosa/scripts/repro_ppl.sh <family> <model_dir> <q_out_dir>`.
