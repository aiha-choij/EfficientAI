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

if [ ! -f "$QOUT/histograms/.done" ]; then
  echo "===== rotation generation ($FAMILY, $MODEL) ====="
  "$PY" "gen_act/grab_acts_rotate_diff_Q_${FAMILY}.py" \
    --model_name "$MODEL" --output_path "$QOUT"
  touch "$QOUT/histograms/.done"
fi

for S in 0.0 0.25 0.4 0.5; do
  echo "===== ppl sparsity=$S ====="
  "$PY" "scripts/ppl_test_larosa_${FAMILY}.py" \
    --model_name "$MODEL" --larosa_path "$QOUT" --sparsity "$S"
done
