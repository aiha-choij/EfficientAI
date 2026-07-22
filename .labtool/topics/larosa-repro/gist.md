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
- flash-attn 2.8.x PyPI wheels: require GLIBC≥2.32, gateway is Ubuntu 20.04
  (glibc 2.31). Pin the official 2.7.4.post1 cu12torch2.6cxx11abiFALSE wheel.

## Open Questions
- Does `attn_implementation` (flash_attention_2 vs sdpa) matter for reproducing the numbers? gen_act hardcodes flash_attention_2.
- README packaging step references `configuration_qwen.py` but the file is `configuration_qwen2.py` (auto_map expects `configuration_qwen2`). (Only relevant for the lm_eval packaging step.)
- Code calibrates with 10×2048 wikitext-2 sequences; paper says 16×2048. Paper's
  own robustness result (covariance cos-sim 0.998 across datasets) suggests this
  doesn't matter — verify via reproduction quality.

## Next Experiments
1. `larosa-llama3-8b-ppl` — repro_ppl.sh llama `/raid/LLM/llama3-8b`.
2. `larosa-qwen25-7b-ppl` — repro_ppl.sh qwen `~/workspace/models/Qwen2.5-7B`.
3. After PPL matches: package Qwen2.5-7B-larosa (sparse_level 0.25) + lm_eval
   6 tasks vs README table; dense baseline comparison.
4. Development direction (after repro): RB-Sparse — rotated-basis block-shared
   mask + eigenspace low-rank compensation (2026-06-24 discussion, see
   research-wiki r-sparse note).

## Active Jobs
- `20260723-070108-larosa-llama2-7b-ppl` @ a100-40-2 — rotation gen +
  PPL sweep 0.0/0.25/0.4/0.5. Journal: 2026-07-22_experiment-larosa-llama2-7b-ppl.md.
  Gateway agent hourly watch: request `20260723-070147-larosa-llama2-7b-ppl-watch3`.

## Pointers
- Paper: arXiv:2507.01299 (La RoSA, Liu et al., ICML 2025); PDF in research-wiki
  `sources/Activation Sparsity/`, enriched note `wiki/papers/la-rosa-rotated-sparse-activation.md`.
- Reference numbers: paper Table 2 (PPL, primary) and Table 1 (accuracy);
  larosa/README.md §5 (Qwen lm_eval, secondary).
- Models on gateway: `/raid/LLM/llama2-7b`, `/raid/LLM/llama3-8b` (shared,
  read-only), `~/workspace/models/Qwen2.5-7B` (ours).
- Runner: `larosa/scripts/repro_ppl.sh <family> <model_dir> <q_out_dir>`.
