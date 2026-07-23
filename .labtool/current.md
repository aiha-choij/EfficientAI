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
Baseline reproduction is COMPLETE: Table 2 PPL matched on LLaMA2-7B/LLaMA3-8B/
Qwen2.5-7B (12/12 points ±0.1). Leading approach next: validate the accuracy
half (package Qwen2.5-7B-larosa 0.25 + lm_eval 6 tasks vs README), then start
RB-Sparse using the saved per-layer D matrices. If lm_eval also matches →
freeze the baseline and move fully to RB-Sparse development.

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
- Check first: no active jobs (`ssh a100-40-2 '~/workspace/bin/runs'`); all three
  PPL reproduction cards are DONE (see journal 2026-07-22_experiment-*).
- Immediate next action: package Qwen2.5-7B-larosa (config sparse_level 0.25,
  copy configuration_qwen2.py + modeling_qwen2_larosa.py, set absolute Q_path to
  ~/workspace/models/qwen25_7b_larosa_Q) and run lm_eval 6 tasks vs README §5.
- Context: (1) rotation matrices (D.pt) for all 3 models already exist under
  ~/workspace/models/*_larosa_Q/histograms/ — do NOT rerun gen_act; its pass 2
  OOMs on 40GB and is unnecessary (eval loads only pass-1 D.pt — see gist Dead
  Ends). (2) The eval model reuses the attn Q for mlp; sparsity is a runtime
  config (α baked in: h1=0.8p, h2=1.2p). (3) Infra fixes this session live in
  QCom (dispatcher PCI_BUS_ID + subshell placement fix) and conda env larosa
  (flash-attn pinned 2.7.4.post1) — QCom commits are local-only, not pushed.
