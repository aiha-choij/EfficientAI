# current

## Active Topics
| topic | status | one-liner |
|---|---|---|
| larosa-repro | 🟢 active | Reproduce LaRoSa (rotated sparse activation) paper Table 2 PPL on LLaMA2/3 + Qwen2.5 |

## This Session
Focus: larosa-repro — Table 2 PPL reproduction, first job (LLaMA2-7B) submitted.

## Active Jobs
- (none)

## Direction
Reproduce paper Table 2 wikitext-2 PPL at 25/40/50% (40% is the headline "near-lossless" point) on LLaMA2-7B, LLaMA3-8B, Qwen2.5-7B via `larosa/scripts/repro_ppl.sh` (no packaging needed — sparsity is a runtime arg). lm_eval accuracy reproduction and RB-Sparse development come after PPL matches.

## Next Experiments
1. `larosa-llama3-8b-ppl` — same runner, `/raid/LLM/llama3-8b`, targets 6.13 → 6.23/6.60/7.22 (±0.1).
2. `larosa-qwen25-7b-ppl` — same runner, `~/workspace/models/Qwen2.5-7B`, targets 6.85 → 6.90/7.10/7.42 (±0.1).
3. After both match: lm_eval 6-task accuracy (needs model packaging), prerequisite for RB-Sparse evals.

## Latest
- 2026-07-22: labtool initialized; conda env `larosa` + Qwen2.5-7B download started on a100-40-2.
- 2026-07-22: found LLaMA2-7B/LLaMA3-8B already on gateway at `/raid/LLM/` (read-only) — no HF token needed.
- 2026-07-22: fixed env: flash-attn pinned to 2.7.4.post1 (2.8.x wheel needs GLIBC≥2.32, gateway has 2.31).
- 2026-07-22: fixed QCom dispatcher: `CUDA_DEVICE_ORDER=PCI_BUS_ID` (PCI GPU index could map to the 4GB DGX Display at runtime).
- 2026-07-22: `larosa-llama2-7b-ppl` DONE in 26 min — all 4 PPL points within ±0.06 of paper Table 2 (5.4736/5.5017/5.6167/5.8167 vs 5.47/5.51/5.64/5.87). Reproduction on LLaMA2-7B succeeded.

## If you're starting a new session
Read `.labtool/topics/larosa-repro/gist.md`. Code runs on gateway a100-40-2 (`~/workspace/repos/EfficientAI/larosa`, conda env `larosa`). Submit jobs via `~/workspace/bin/qsub`; check with `runs`.
