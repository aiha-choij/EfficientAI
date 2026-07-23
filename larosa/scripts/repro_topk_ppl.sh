#!/usr/bin/env bash
# Intermediate activation Top-K wikitext-2 PPL sweep:
# FFN-only per-token magnitude Top-K on i = u * g (no rotation, dense
# attention). Loads the vanilla HF model — no gen_act / rotation stage.
#
#   repro_topk_ppl.sh <family> <model_dir>
#     family     llama | qwen  (picks the ppl script)
#     model_dir  HF model directory (read-only)
#
# Sparsity 0.0 keeps all elements in top_k_new -> dense-equivalent baseline
# (must match the dense PPL from the larosa-repro baseline within noise).
set -euo pipefail

FAMILY=$1
MODEL=$2
PY=${PY:-$HOME/miniconda3/envs/larosa/bin/python}

cd "$(dirname "$0")/.."

for S in 0.0 0.5 0.7 0.9; do
  echo "===== ppl mode=topk_intermediate sparsity=$S ====="
  "$PY" "scripts/ppl_test_larosa_${FAMILY}.py" \
    --model_name "$MODEL" --mode topk_intermediate --sparsity "$S"
done
