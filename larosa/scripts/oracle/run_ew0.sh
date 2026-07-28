#!/usr/bin/env bash
# E-W0: loss-aligned (output-side) sensitivity — gradient stats, metric
# validation against 11 factor variants with KNOWN PPL@s=0.9, corpus
# stability check, plus the TIS frontier fill-in s={0.75,0.8,0.85}.
# Gate: the weighted metric must fix the whitening inversion (wht > plain in
# error, matching PPL) that plain input-L2 provably gets backwards.
set -euo pipefail

MODEL=${1:-/raid/LLM/llama2-7b}
OUT=${2:-$HOME/workspace/oracle/llama2-7b}
PY=${PY:-$HOME/workspace/venv-larosa/bin/python}
F="$OUT/factors"

cd "$(dirname "$0")/../.."

"$PY" scripts/oracle/10_grad_calibrate.py --model_name "$MODEL" \
  --tokens_pt "$OUT/stats/c4/calib_tokens.pt" --nsamples 128 \
  --out_dir "$OUT/grad_stats/c4"
"$PY" scripts/oracle/10_grad_calibrate.py --model_name "$MODEL" \
  --dataset wikitext103 --nsamples 128 \
  --out_dir "$OUT/grad_stats/wt103"

"$PY" scripts/oracle/11_metric_check.py --model_name "$MODEL" \
  --stats_dir "$OUT/stats/c4" \
  --grad_dir "$OUT/grad_stats/c4" --grad_dir_alt "$OUT/grad_stats/wt103" \
  --factors plain512="$F/r512" plain1024="$F/plain_r1024" \
    plain2048="$F/plain_r2048" wht512="$F/wht_r512" wht1024="$F/wht_r1024" \
    alloc256="$F/wht_alloc256" alloc512="$F/wht_alloc512" \
    alloc1024="$F/wht_alloc1024" slr512k1024="$F/slr_r512_k1024" \
    slr256k1536="$F/slr_r256_k1536" slr0k2048="$F/slr_r0_k2048" \
  --ppl plain512=8.7638 plain1024=7.2294 plain2048=6.7098 wht512=9.7606 \
    wht1024=7.3974 alloc256=17.4118 alloc512=10.2638 alloc1024=7.6336 \
    slr512k1024=6.9962 slr256k1536=6.9417 slr0k2048=6.9526 \
  --nsamples 8 --out_json "$OUT/results/metric_check.json"

for S in 0.75 0.8 0.85; do
  echo "----- c1 (TIS frontier) s=$S -----"
  "$PY" scripts/oracle/04_eval_ppl.py --model_name "$MODEL" --condition c1 \
    --select topk --s "$S" \
    --out_json "$OUT/results/c1_topk_s${S}.json"
done
echo "===== E-W0 complete ====="
