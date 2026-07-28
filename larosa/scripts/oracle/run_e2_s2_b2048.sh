#!/usr/bin/env bash
# E2 B_eff=2048 round (H4 escalation after the E1 gate pass):
#   lr_r2048        — plain rank reference (subsumes the old r=2048 arm)
#   slr_r1024_k2048 / slr_r512_k3072 / slr_r256_k3584 — s2 splits at rank
#   share 50/25/12.5%, bracketing E1's mixed optimum (r256:k1536 @ B1024)
# x s in {0.5, 0.7, 0.9}. Anchors: dense 5.4738; C3 6.6381, E1 best
# s2_r256:k1536 6.9417, lr_r1024 7.2294 (all @ s=0.9). B_eff=2048 = +12.4%.
set -euo pipefail

MODEL=${1:-/raid/LLM/llama2-7b}
OUT=${2:-$HOME/workspace/oracle/llama2-7b}
PY=${PY:-$HOME/workspace/venv-larosa/bin/python}
S_GRID="0.5 0.7 0.9"
SPLITS="1024:2048 512:3072 256:3584"

cd "$(dirname "$0")/../.."

"$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" \
  --stats_dir "$OUT/stats/c4" --rank 2048 --out_dir "$OUT/factors/plain_r2048"
for SPLIT in $SPLITS; do
  R=${SPLIT%%:*}; K=${SPLIT##*:}
  "$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" \
    --stats_dir "$OUT/stats/c4" --rank "$R" \
    --comp_mode slr_input --sparse_k "$K" --x_score abs \
    --out_dir "$OUT/factors/slr_r${R}_k${K}"
done

for S in $S_GRID; do
  echo "----- c4 variant=plain_r2048 s=$S -----"
  "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition c4 \
    --select topk --s "$S" --rank 2048 --stats_dir "$OUT/stats/c4" \
    --factors_dir "$OUT/factors/plain_r2048" \
    --out_json "$OUT/results/c4_plain_r2048_topk_s${S}.json"
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
echo "===== E2 B2048 sweep complete ====="
