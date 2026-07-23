# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| larosa-repro | 🟢 active | Reproduce LaRoSa (rotated sparse activation) paper Table 2 PPL on LLaMA2/3 + Qwen2.5 |

## This Session
Focus: larosa-repro — Table 2 PPL reproduction COMPLETE on all 3 models (12/12 points ±0.1).

## Active Jobs
- (none)
  targets 6.13/6.23/6.60/7.22 ±0.1. Journal: 2026-07-22_experiment-larosa-llama3-8b-ppl.md.
- `20260723-102000-larosa-qwen25-7b-ppl` @ a100-40-2 — rotation gen + PPL sweep;
  targets 6.85/6.90/7.10/7.42 ±0.1. Journal: 2026-07-22_experiment-larosa-qwen25-7b-ppl.md.
- Gateway agent hourly watch for both: request `...-084829-larosa-ppl-repro-watch`.

## Direction
Reproduce paper Table 2 wikitext-2 PPL at 25/40/50% (40% is the headline "near-lossless" point) on LLaMA2-7B, LLaMA3-8B, Qwen2.5-7B via `larosa/scripts/repro_ppl.sh` (no packaging needed — sparsity is a runtime arg). lm_eval accuracy reproduction and RB-Sparse development come after PPL matches.

## Next Experiments
1. Package Qwen2.5-7B-larosa (0.25) + lm_eval 6-task accuracy vs README table.
2. RB-Sparse: cross-token top-k index agreement in rotated basis (uses saved D matrices).

## Latest
- 2026-07-22: llama3-8b + qwen25-7b PPL DONE — all 12 points across 3 models within ±0.1 of paper Table 2. Reproduction complete.
- 2026-07-22: `larosa-llama3-8b-ppl` + `larosa-qwen25-7b-ppl` submitted (validated runner, both on a100-40-2).
- 2026-07-22: labtool initialized; conda env `larosa` + Qwen2.5-7B download started on a100-40-2.
- 2026-07-22: found LLaMA2-7B/LLaMA3-8B already on gateway at `/raid/LLM/` (read-only) — no HF token needed.
- 2026-07-22: fixed env: flash-attn pinned to 2.7.4.post1 (2.8.x wheel needs GLIBC≥2.32, gateway has 2.31).

## If you're starting a new session
Read `.labtool/topics/larosa-repro/gist.md`. Code runs on gateway a100-40-2 (`~/workspace/repos/EfficientAI/larosa`, conda env `larosa`). Submit jobs via `~/workspace/bin/qsub`; check with `runs`.
