#!/usr/bin/env bash
# L2 single verification point, mirroring run_validate_l0l1.sh: g=1, s=0.9,
# small model, PPL only -- confirm L2 (sequential refit against the sparse
# stream) works end-to-end at real model scale, and compare against L0/L1
# at the same (s,g) before committing to a full L2 matrix entry.
set -euo pipefail

MODEL=${1:?model path}
OUT=${2:-$HOME/workspace/refit/$(basename "$MODEL")}
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}
S=0.9
G=1
LAM=0.01

cd "$(dirname "$0")/../.."

"$PY" scripts/refit/02_build_l2.py --model_name "$MODEL" \
  --dataset wikitext103 --nsamples 128 --seqlen 2048 \
  --s "$S" --g "$G" --lam "$LAM" \
  --out_dir "$OUT/weights/l2_s${S}_g${G}"

"$PY" scripts/refit/02_eval_ppl.py --model_name "$MODEL" \
  --mode l2 --s "$S" --g "$G" \
  --weights_dir "$OUT/weights/l2_s${S}_g${G}_lam${LAM}" \
  --out_json "$OUT/results/l2_s${S}_g${G}_lam${LAM}.json"

echo "===== L2 validation complete: $OUT/results ====="
