#!/usr/bin/env bash
# Completion arms for the whitening round (Task 2.3 uniform-vs-alloc at equal
# r_bar, plus the whitening ablation at r=1024):
#   wht_r1024   — uniform whitened rank 1024 (vs wht_alloc1024, same budget)
#   plain_r1024 — uniform plain rank 1024   (isolates whitening from rank)
set -euo pipefail

MODEL=${1:-/raid/LLM/llama2-7b}
OUT=${2:-$HOME/workspace/oracle/llama2-7b}
PY=${PY:-$HOME/workspace/venv-larosa/bin/python}
S_GRID="0.5 0.7 0.9"

cd "$(dirname "$0")/../.."

"$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" --stats_dir "$OUT/stats/c4" \
  --rank 1024 --whiten --out_dir "$OUT/factors/wht_r1024"
"$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" --stats_dir "$OUT/stats/c4" \
  --rank 1024 --out_dir "$OUT/factors/plain_r1024"

for VAR in wht_r1024 plain_r1024; do
  for S in $S_GRID; do
    echo "----- c4 variant=$VAR s=$S -----"
    "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition c4 \
      --select topk --s "$S" --stats_dir "$OUT/stats/c4" \
      --factors_dir "$OUT/factors/$VAR" \
      --out_json "$OUT/results/c4_${VAR}_topk_s${S}.json"
  done
done
echo "===== rank1024 arms complete ====="
