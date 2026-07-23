#!/usr/bin/env bash
# Oracle condition x p wikitext-2 PPL sweep on one model (LLaMA2-7B scope).
#
#   oracle_ppl_sweep.sh <model_dir> <out_root> <condition> [p ...]
#     model_dir  HF model directory (read-only), e.g. /raid/LLM/llama2-7b
#     out_root   e.g. ~/workspace/runs-data/oracle/llama2-7b (absolute path)
#     condition  dense | c1 | c2 | c3 | c4 | c5
#     p ...      optional explicit p values; default full grid (spec section 5)
#
# Prereqs (once, before c3/c4/c5):
#   01_calibrate.py  --out_dir $OUT/stats/c4
#   03_build_M.py    --stats_dir $OUT/stats/c4 --rank 512 --out_dir $OUT/factors/c4_r512
set -euo pipefail

MODEL=$1
OUT=$2
COND=$3
shift 3
P_GRID=${*:-"0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.93 0.95 0.97 0.99"}
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}
RANK=512

cd "$(dirname "$0")/../.."

STATS_ARGS=()
case "$COND" in
  c3|c5) STATS_ARGS=(--stats_dir "$OUT/stats/c4") ;;
  c4)    STATS_ARGS=(--stats_dir "$OUT/stats/c4" --factors_dir "$OUT/factors/c4_r${RANK}" --rank "$RANK") ;;
esac

if [ "$COND" = "dense" ]; then
  echo "===== oracle condition=dense (C0 baseline) ====="
  "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition dense \
    --out_json "$OUT/results/dense.json"
  exit 0
fi

for P in $P_GRID; do
  echo "===== oracle condition=$COND p=$P ====="
  "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition "$COND" \
    --p "$P" "${STATS_ARGS[@]}" \
    --out_json "$OUT/results/${COND}_p${P}.json"
done
