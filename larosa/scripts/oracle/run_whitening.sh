#!/usr/bin/env bash
# C4 improvement round (spec-c4-whitening.md), LLaMA2-7B scope:
#   1. Sigma calibration pass (reuses the saved calibration token list)
#   2. factors: plain spectra (diagnostic), whitened r=512 uniform,
#      whitened + budget allocation r_bar in {256, 512, 1024}
#   3. Task-0 diagnostics (trunc_err vs tail_norm, plain vs whitened)
#   4. C4 PPL sweeps (top-K s = 0.5/0.7/0.9) for each factor variant
# Existing plain-SVD factors/results stay untouched (baseline preservation).
set -euo pipefail

MODEL=${1:-/raid/LLM/llama2-7b}
OUT=${2:-$HOME/workspace/oracle/llama2-7b}
PY=${PY:-$HOME/workspace/venv-larosa/bin/python}
S_GRID="0.5 0.7 0.9"

cd "$(dirname "$0")/../.."

echo "===== 1. sigma calibration (xxt) ====="
"$PY" scripts/oracle/01_calibrate.py --model_name "$MODEL" --dataset c4 \
  --nsamples 512 --seqlen 2048 --out_dir "$OUT/stats/c4" --xxt

echo "===== 2a. plain spectra (diagnostic only; baseline factors preserved) ====="
"$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" --stats_dir "$OUT/stats/c4" \
  --rank 512 --out_dir "$OUT/factors/plain_r512_spectra" \
  --spectra_out "$OUT/results/spectra_plain.json"

echo "===== 2b. whitened uniform r=512 ====="
"$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" --stats_dir "$OUT/stats/c4" \
  --rank 512 --whiten --out_dir "$OUT/factors/wht_r512" \
  --spectra_out "$OUT/results/spectra_wht.json"

for RB in 256 512 1024; do
  echo "===== 2c. whitened + budget allocation r_bar=$RB ====="
  "$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" --stats_dir "$OUT/stats/c4" \
    --whiten --alloc "budget:$RB" --out_dir "$OUT/factors/wht_alloc$RB"
done

echo "===== 3. Task-0 diagnostics ====="
"$PY" scripts/oracle/07_diag_c4.py --model_name "$MODEL" --stats_dir "$OUT/stats/c4" \
  --factors "plain=$OUT/factors/r512" "wht=$OUT/factors/wht_r512" \
  --s_op 0.7 0.9 --nsamples 4 --out_csv "$OUT/results/diag_trunc_vs_tail.csv"

echo "===== 4. C4 sweeps per variant ====="
for VAR in wht_r512 wht_alloc256 wht_alloc512 wht_alloc1024; do
  for S in $S_GRID; do
    echo "----- c4 variant=$VAR s=$S -----"
    "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition c4 \
      --select topk --s "$S" --stats_dir "$OUT/stats/c4" \
      --factors_dir "$OUT/factors/$VAR" \
      --out_json "$OUT/results/c4_${VAR}_topk_s${S}.json"
  done
done

echo "===== whitening round complete: $OUT ====="
