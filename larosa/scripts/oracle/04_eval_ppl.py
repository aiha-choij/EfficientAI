# coding=utf-8
# Condition x p PPL evaluation. Reuses the EXACT wikitext-2 PPL pipeline from
# the larosa-repro / topk_intermediate experiments
# (eval_ppl_wikitext_with_inference_sparsity) so numbers are directly
# comparable with the trusted dense baseline and the Top-K results.
#
# NOTE the known upstream label swap in that eval: the printed "attn h1/h2"
# lines are really MLP values and vice versa. The JSON written here uses the
# correct per-layer values read directly from layer.mlp.
#
# Usage:
#   python scripts/oracle/04_eval_ppl.py --model_name /raid/LLM/llama2-7b \
#       --condition c3 --p 0.9 --stats_dir oracle_out/llama2-7b/stats/c4 \
#       --out_json oracle_out/llama2-7b/results/c3_p0.9.json

import argparse
import json
import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp
from utils.eval_ppl import eval_ppl_wikitext_with_inference_sparsity


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=parent_dir).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--condition", type=str, required=True, choices=list(oracle_mlp.CONDITIONS))
    ap.add_argument("--select", type=str, default="topp", choices=["topp", "topk"],
                    help="topp: spec cumulative-mass knob; topk: exact sparsity s "
                         "(K=int((1-s)*d), matches the larosa topk_intermediate setup)")
    ap.add_argument("--p", type=float, default=1.0, help="top-p knob (select=topp)")
    ap.add_argument("--s", type=float, default=0.0, help="exact sparsity (select=topk)")
    ap.add_argument("--rank", type=int, default=512, help="C4 compensation rank (record only)")
    ap.add_argument("--stats_dir", type=str, default=None, help="required for c3/c4/c5")
    ap.add_argument("--factors_dir", type=str, default=None, help="required for c4")
    ap.add_argument("--exclude_layers", type=int, nargs="*", default=[])
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    if args.condition in ("c3", "c4", "c5") and not args.stats_dir:
        ap.error(f"--stats_dir is required for {args.condition}")
    if args.condition == "c4" and not args.factors_dir:
        ap.error("--factors_dir is required for c4")

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "oracle"
    config.oracle_condition = args.condition
    config.oracle_select = args.select
    config.oracle_p = args.p
    config.oracle_s = args.s
    config.oracle_exclude_layers = args.exclude_layers

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    oracle_mlp.attach_col_norms(model)
    if args.stats_dir:
        oracle_mlp.load_stats(model, args.stats_dir)
    if args.condition == "c4":
        oracle_mlp.load_factors(model, args.factors_dir)
    oracle_mlp.set_condition(model, args.condition, p=args.p, select=args.select,
                             s=args.s, exclude_layers=args.exclude_layers)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    knob = f"s={args.s}" if args.select == "topk" else f"p={args.p}"
    print("=" * 40)
    print(f"oracle condition={args.condition} select={args.select} {knob} rank={args.rank}")
    with torch.no_grad():
        ppl = eval_ppl_wikitext_with_inference_sparsity(
            model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"Oracle PPL: {ppl}")

    per_layer = oracle_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)
    print(f"achieved sparsity mean={mean_sp:.4f}")

    factors_meta = None
    if args.factors_dir:
        meta_path = os.path.join(args.factors_dir, "factors_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                factors_meta = json.load(f)
                factors_meta.pop("ranks", None)  # keep the JSON small

    result = {
        "model_name": args.model_name,
        "condition": args.condition,
        "factors_meta": factors_meta,
        "select": args.select,
        "s": args.s if args.select == "topk" else None,
        "p": args.p if args.select == "topp" else None,
        "rank": args.rank if args.condition == "c4" else None,
        "ppl": ppl,
        "achieved_sparsity_mean": mean_sp,
        "achieved_sparsity_per_layer": per_layer,
        "stats_dir": args.stats_dir,
        "factors_dir": args.factors_dir,
        "exclude_layers": args.exclude_layers,
        "git_commit": git_commit(),
    }
    print("RESULT_JSON " + json.dumps({k: v for k, v in result.items()
                                       if k != "achieved_sparsity_per_layer"}))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
