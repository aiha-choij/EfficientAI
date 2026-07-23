#!/usr/bin/env bash
# Phase 2 job (oracle-residual-sparsity): calibration + phase-0 distribution
# report + C4 compensation factors on LLaMA2-7B.
#
#   run_phase2.sh [model_dir] [out_root]
#
# Order: wikitext-103 first (guaranteed local/HF-cache), then c4 streaming.
# If the c4 download fails (network), fall back to wikitext-103 as the primary
# calibration corpus; the chosen primary is recorded in $OUT/primary_stats.txt.
set -euo pipefail

MODEL=${1:-/raid/LLM/llama2-7b}
OUT=${2:-$HOME/workspace/oracle/llama2-7b}
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}
RANK=512

cd "$(dirname "$0")/../.."

echo "===== 01 calibrate: wikitext103 ====="
"$PY" scripts/oracle/01_calibrate.py --model_name "$MODEL" --dataset wikitext103 \
  --nsamples 512 --seqlen 2048 --out_dir "$OUT/stats/wikitext103"

echo "===== 01 calibrate: c4 ====="
if "$PY" scripts/oracle/01_calibrate.py --model_name "$MODEL" --dataset c4 \
     --nsamples 512 --seqlen 2048 --out_dir "$OUT/stats/c4"; then
  PRIMARY="$OUT/stats/c4"
  B_ARGS=(--stats_dir_b "$OUT/stats/wikitext103")
else
  echo "WARN: c4 calibration failed (network?); using wikitext103 as primary"
  PRIMARY="$OUT/stats/wikitext103"
  B_ARGS=()
fi
echo "$PRIMARY" > "$OUT/primary_stats.txt"

echo "===== 02 phase-0 distribution report (primary: $PRIMARY) ====="
"$PY" scripts/oracle/02_distribution_report.py --model_name "$MODEL" \
  --stats_dir "$PRIMARY" "${B_ARGS[@]+"${B_ARGS[@]}"}" \
  --nsamples 32 --out_dir "$OUT/phase0"

echo "===== 03 build M factors r=$RANK ====="
"$PY" scripts/oracle/03_build_M.py --model_name "$MODEL" \
  --stats_dir "$PRIMARY" --rank "$RANK" --out_dir "$OUT/factors/r${RANK}"

echo "===== phase 2 complete: $OUT ====="
