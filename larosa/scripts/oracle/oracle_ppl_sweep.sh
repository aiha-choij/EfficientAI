#!/usr/bin/env bash
# Oracle condition x sparsity wikitext-2 PPL sweep on one model (LLaMA2-7B scope).
#
#   oracle_ppl_sweep.sh <model_dir> <out_root> <condition> [knob ...]
#     model_dir  HF model directory (read-only), e.g. /raid/LLM/llama2-7b
#     out_root   e.g. $HOME/workspace/oracle/llama2-7b (absolute path)
#     condition  dense | c1 | c2 | c3 | c4 | c5
#     knob ...   optional explicit knob values; default grid below
#
# SELECT=topk (default): knob = exact per-token sparsity s, K = int((1-s)*d) —
#   identical neuron-selection semantics to the larosa topk_intermediate
#   experiment, so C1 here reproduces that setup and C3 vs C1 at the same s is
#   a matched-compute comparison (2026-07-23 user decision).
# SELECT=topp: knob = the spec's cumulative-mass p (report axis = achieved
#   sparsity); kept for spec-faithful runs.
#
# Prereqs (once, before c3/c4/c5):
#   01_calibrate.py  --out_dir $OUT/stats/c4
#   03_build_M.py    --stats_dir $OUT/stats/c4 --rank 512 --out_dir $OUT/factors/r512
set -euo pipefail

MODEL=$1
OUT=$2
COND=$3
shift 3
SELECT=${SELECT:-topk}
if [ "$SELECT" = "topk" ]; then
  GRID=${*:-"0.5 0.7 0.9"}
else
  GRID=${*:-"0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.93 0.95 0.97 0.99"}
fi
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}
RANK=512

cd "$(dirname "$0")/../.."

STATS_ARGS=()
case "$COND" in
  c3|c5) STATS_ARGS=(--stats_dir "$OUT/stats/c4") ;;
  c4)    STATS_ARGS=(--stats_dir "$OUT/stats/c4" --factors_dir "$OUT/factors/r${RANK}" --rank "$RANK") ;;
esac

if [ "$COND" = "dense" ]; then
  echo "===== oracle condition=dense (C0 baseline) ====="
  "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition dense \
    --out_json "$OUT/results/dense.json"
  exit 0
fi

for K in $GRID; do
  echo "===== oracle condition=$COND select=$SELECT knob=$K ====="
  if [ "$SELECT" = "topk" ]; then
    "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition "$COND" \
      --select topk --s "$K" "${STATS_ARGS[@]+"${STATS_ARGS[@]}"}" \
      --out_json "$OUT/results/${COND}_topk_s${K}.json"
  else
    "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition "$COND" \
      --select topp --p "$K" "${STATS_ARGS[@]+"${STATS_ARGS[@]}"}" \
      --out_json "$OUT/results/${COND}_topp_p${K}.json"
  fi
done
