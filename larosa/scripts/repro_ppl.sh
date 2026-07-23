#!/usr/bin/env bash
# Reproduce LaRoSA (arXiv:2507.01299) Table 2 wikitext-2 PPL:
# rotation generation (if missing) + PPL sweep over sparsity levels.
#
#   repro_ppl.sh <family> <model_dir> <q_out_dir>
#     family     llama | qwen  (picks gen_act / ppl scripts)
#     model_dir  HF model directory (read-only)
#     q_out_dir  rotation output dir (histograms/ written here)
#
# Sparsity 0.0 keeps all elements in top_k_new -> dense-equivalent baseline.
set -euo pipefail

FAMILY=$1
MODEL=$2
QOUT=$3
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}

cd "$(dirname "$0")/.."

# The PPL eval model loads only pass-1's per-layer self_attn/D.pt (the mlp
# reuses the same Q). gen_act's pass 2 (histograms for threshold analysis) can
# OOM on 40GB GPUs and is not needed here — accept a gen_act failure as long
# as every layer produced its D.pt. Pass 1 writes layers in order, so
# count(D.pt) == count(layer dirs) with a non-empty set means it completed.
rotation_done() {
  local ndirs dcount
  ndirs=$(ls -d "$QOUT"/histograms/layer-* 2>/dev/null | wc -l)
  dcount=$(ls "$QOUT"/histograms/layer-*/self_attn/D.pt 2>/dev/null | wc -l)
  [ "$ndirs" -gt 0 ] && [ "$dcount" -eq "$ndirs" ]
}

if ! rotation_done; then
  echo "===== rotation generation ($FAMILY, $MODEL) ====="
  set +e
  "$PY" "gen_act/grab_acts_rotate_diff_Q_${FAMILY}.py" \
    --model_name "$MODEL" --output_path "$QOUT"
  gen_rc=$?
  set -e
  if rotation_done; then
    [ "$gen_rc" -ne 0 ] && echo "warn: gen_act exited $gen_rc (pass-2 histograms incomplete); D matrices complete — continuing"
  else
    echo "rotation generation failed (incomplete D.pt set)"; exit 1
  fi
fi

for S in 0.0 0.25 0.4 0.5; do
  echo "===== ppl sparsity=$S ====="
  "$PY" "scripts/ppl_test_larosa_${FAMILY}.py" \
    --model_name "$MODEL" --larosa_path "$QOUT" --sparsity "$S"
done
