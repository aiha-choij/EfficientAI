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
    ap.add_argument("--p", type=float, default=1.0, help="top-p knob (ignored for dense)")
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
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "oracle"
    config.oracle_condition = args.condition
    config.oracle_p = args.p
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
    oracle_mlp.set_condition(model, args.condition, args.p, exclude_layers=args.exclude_layers)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("=" * 40)
    print(f"oracle condition={args.condition} p={args.p} rank={args.rank}")
    with torch.no_grad():
        ppl = eval_ppl_wikitext_with_inference_sparsity(
            model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"Oracle PPL: {ppl}")

    per_layer = oracle_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)
    print(f"achieved sparsity mean={mean_sp:.4f}")

    result = {
        "model_name": args.model_name,
        "condition": args.condition,
        "p": args.p,
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
