#!/usr/bin/env bash
# Single verification point (request spec "실행 순서"): g=1, s=0.9, L0 vs L1,
# small model, PPL only -- confirm the refit effect exists before submitting
# the full matrix. wikitext103 calibration (c4 streaming untested on the
# gateway per oracle-residual-sparsity gist's Open Questions).
set -euo pipefail

MODEL=${1:?model path}
OUT=${2:-$HOME/workspace/refit/$(basename "$MODEL")}
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}
S=0.9
G=1
LAM=0.01

cd "$(dirname "$0")/../.."

"$PY" scripts/refit/01_build_l1.py --model_name "$MODEL" \
  --dataset wikitext103 --nsamples 512 --seqlen 2048 \
  --s "$S" --g "$G" --lambdas "$LAM" \
  --out_dir "$OUT/weights/l1_s${S}_g${G}"

"$PY" scripts/refit/02_eval_ppl.py --model_name "$MODEL" \
  --mode l0 --s "$S" --g "$G" \
  --out_json "$OUT/results/l0_s${S}_g${G}.json"

"$PY" scripts/refit/02_eval_ppl.py --model_name "$MODEL" \
  --mode l1 --s "$S" --g "$G" \
  --weights_dir "$OUT/weights/l1_s${S}_g${G}_lam${LAM}" \
  --out_json "$OUT/results/l1_s${S}_g${G}_lam${LAM}.json"

echo "===== L0 vs L1 validation complete: $OUT/results ====="
