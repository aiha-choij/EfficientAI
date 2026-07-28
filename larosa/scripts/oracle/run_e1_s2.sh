#!/usr/bin/env bash
# E1 S2 PPL sweep (H4, gist Next Experiments 1; E0 verdict: S2-only):
# c4 comp_mode=slr_input, abs score, B_eff=1024 arms
#   slr_r512_k1024 / slr_r256_k1536 / slr_r0_k2048   x  s in {0.5, 0.7, 0.9}
# Anchors (phase 3-4 / whitening round): dense 5.4738; C1 8.1096, C3 6.638,
# c4 plain_r1024 7.229 (all @ s=0.9).
set -euo pipefail

MODEL=${1:-/raid/LLM/llama2-7b}
OUT=${2:-$HOME/workspace/oracle/llama2-7b}
PY=${PY:-$HOME/workspace/venv-larosa/bin/python}
S_GRID="0.5 0.7 0.9"
SPLITS="512:1024 256:1536 0:2048"

cd "$(dirname "$0")/../.."

for SPLIT in $SPLITS; do
  R=${SPLIT%%:*}; K=${SPLIT##*:}
  "$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" \
    --stats_dir "$OUT/stats/c4" --rank "$R" \
    --comp_mode slr_input --sparse_k "$K" --x_score abs \
    --out_dir "$OUT/factors/slr_r${R}_k${K}"
done

for SPLIT in $SPLITS; do
  R=${SPLIT%%:*}; K=${SPLIT##*:}
  VAR="slr_r${R}_k${K}"
  for S in $S_GRID; do
    echo "----- c4 variant=$VAR s=$S -----"
    "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition c4 \
      --select topk --s "$S" --rank "$R" --stats_dir "$OUT/stats/c4" \
      --factors_dir "$OUT/factors/$VAR" \
      --out_json "$OUT/results/c4_${VAR}_topk_s${S}.json"
  done
done
echo "===== E1 S2 sweep complete ====="
