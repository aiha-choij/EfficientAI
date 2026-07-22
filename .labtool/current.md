# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| larosa-repro | 🟢 active | Reproduce LaRoSa (rotated sparse activation) results on Qwen2.5-7B |

## This Session
Focus: larosa-repro — first reproduction run (env setup + rotation gen + eval).

## Active Jobs
- (none yet — env setup and Qwen2.5-7B download running outside qsub on a100-40-2)

## Direction
Reproduce README reference lm_eval numbers for Qwen2.5-7B-larosa at sparse_level 0.25, then compare against dense baseline.

## Next Experiments
1. gen_act rotation generation on Qwen2.5-7B.
2. Package Qwen2.5-7B-larosa + lm_eval (6 tasks) + wikitext PPL.

## Latest
- 2026-07-22: labtool initialized; conda env `larosa` + Qwen2.5-7B download started on a100-40-2.

## If you're starting a new session
Read `.labtool/topics/larosa-repro/gist.md`. Code runs on gateway a100-40-2 (`~/workspace/repos/EfficientAI/larosa`, conda env `larosa`). Submit jobs via `~/workspace/bin/qsub`.
