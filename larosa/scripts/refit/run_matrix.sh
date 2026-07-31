#!/usr/bin/env bash
# Reduced-cost L0/L1 matrix (request spec cost-reduction ladder, rung 1):
#   g in {1, 32}: all s in {0.5, 0.7, 0.9}
#   g in {8, 128}: s=0.9 only
# lambda fixed at 0.01 (the spec's default; sweep is dev-model only and not
# run here -- single lambda keeps the grid a reasonable size).
set -euo pipefail

MODEL=${1:?model path}
OUT=${2:-$HOME/workspace/refit/$(basename "$MODEL")}
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}
LAM=0.01
PAIRS="0.5:1 0.7:1 0.9:1 0.5:32 0.7:32 0.9:32 0.9:8 0.9:128"

cd "$(dirname "$0")/../.."

for PAIR in $PAIRS; do
  S=${PAIR%%:*}; G=${PAIR##*:}
  echo "===== building L1 s=$S g=$G ====="
  "$PY" scripts/refit/01_build_l1.py --model_name "$MODEL" \
    --dataset wikitext103 --nsamples 512 --seqlen 2048 \
    --s "$S" --g "$G" --lambdas "$LAM" \
    --out_dir "$OUT/weights/l1_s${S}_g${G}"

  echo "===== eval L0 s=$S g=$G ====="
  "$PY" scripts/refit/02_eval_ppl.py --model_name "$MODEL" \
    --mode l0 --s "$S" --g "$G" \
    --out_json "$OUT/results/l0_s${S}_g${G}.json"

  echo "===== eval L1 s=$S g=$G lam=$LAM ====="
  "$PY" scripts/refit/02_eval_ppl.py --model_name "$MODEL" \
    --mode l1 --s "$S" --g "$G" \
    --weights_dir "$OUT/weights/l1_s${S}_g${G}_lam${LAM}" \
    --out_json "$OUT/results/l1_s${S}_g${G}_lam${LAM}.json"
done
echo "===== L0/L1 reduced matrix complete: $OUT/results ====="
