# larosa-repro — LaRoSa reproduction (arXiv:2507.01299)

## Status
active

## Hypothesis
The LaRoSa pipeline in `larosa/` (upstream Alibaba Cloud code, forked to aiha-choij/EfficientAI) reproduces the README reference numbers on Qwen2.5-7B: rotation-based top-k activation sparsification at sparse_level 0.25 keeps zero-shot accuracy near dense (e.g. arc_easy 0.796, boolq 0.851, hellaswag acc_norm 0.782, winogrande 0.707).

## Key Findings
- (none yet)

## Dead Ends
- (none yet)

## Open Questions
- Does `attn_implementation` (flash_attention_2 vs sdpa) matter for reproducing the numbers? gen_act hardcodes flash_attention_2.
- README packaging step references `configuration_qwen.py` but the file is `configuration_qwen2.py` (auto_map expects `configuration_qwen2`).
- `Q_path` in config is a relative path — eval must run from a cwd where it resolves; use absolute path in packaged config.

## Next Experiments
1. gen_act rotation generation on Qwen2.5-7B (wikitext-2 + alpaca calibration).
2. Package Qwen2.5-7B-larosa (sparse_level 0.25) and run lm_eval on openbookqa, arc_easy, winogrande, hellaswag, arc_challenge, boolq + wikitext PPL; compare to README table.

## Active Jobs
- (none yet)

## Pointers
- Paper: arXiv:2507.01299 (La RoSA, Liu et al. 2025)
- Code: `larosa/` in this repo; README has the reference results table.
- Reference numbers source: larosa/README.md §5 Performance Results (Qwen2.5-7B-larosa).
