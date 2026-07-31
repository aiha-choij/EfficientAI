# coding=utf-8
# Condition x p x g PPL evaluation for the block-shared-mask compensation
# conditions (C7a/C7/C8a/C8). Reuses the oracle calibration format exactly
# (g_bar/col_norm from scripts/oracle/01_calibrate.py -- no separate
# calibration pass for this topic, see block_comp_mlp.py's C3/C4/C5-style
# residual score) and the same trusted wikitext-2 PPL pipeline as
# scripts/oracle/04_eval_ppl.py, so numbers are directly comparable with
# the existing oracle C2/C4 g=1 anchors.
#
# Usage:
#   python scripts/block_comp/01_eval_ppl.py --model_name /raid/LLM/llama3.2-3b-instruct \
#       --condition c7a --p 0.5 --g 16 --stats_dir oracle_out/llama3.2-3b/stats/wikitext103 \
#       --out_json block_comp_out/llama3.2-3b/results/c7a_p0.5_g16.json

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
from inference import oracle_mlp, block_comp_mlp
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
    ap.add_argument("--condition", type=str, required=True, choices=list(block_comp_mlp.CONDITIONS))
    ap.add_argument("--p", type=float, default=1.0, help="top-p knob, applied to the block-aggregated score")
    ap.add_argument("--g", type=int, default=1, help="token-block size (g=1 = per-token, oracle C4 anchor)")
    ap.add_argument("--rank", type=int, default=512, help="C7 comp_lr rank (M = W_down diag(g_bar) W_up)")
    ap.add_argument("--r_sk", type=int, default=256, help="C8/C8a gate/up/down sketch rank (single knob)")
    ap.add_argument("--stats_dir", type=str, required=True,
                    help="oracle-format g_bar/col_norm stats (scripts/oracle/01_calibrate.py output); "
                         "required for every condition here (all use the residual score)")
    ap.add_argument("--factors_dir", type=str, default=None,
                    help="precomputed block factors dir (block_comp_mlp.save_block_factors); "
                         "if omitted for c7/c8/c8a, factors are built in-process from --rank/--r_sk")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "block_comp"
    config.blk_condition = args.condition
    config.blk_p = args.p
    config.blk_g = args.g

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    oracle_mlp.attach_col_norms(model)
    oracle_mlp.load_stats(model, args.stats_dir)  # attaches oracle_g_bar (residual score needs it, all conditions)

    if args.condition != "c7a":
        if args.factors_dir:
            block_comp_mlp.load_block_factors(model, args.factors_dir)
        else:
            block_comp_mlp.attach_block_factors_inplace(model, rank=args.rank, r_sk=args.r_sk)

    block_comp_mlp.set_condition(model, args.condition, p=args.p, g=args.g)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("=" * 40)
    print(f"block_comp condition={args.condition} p={args.p} g={args.g} rank={args.rank} r_sk={args.r_sk}")
    with torch.no_grad():
        ppl = eval_ppl_wikitext_with_inference_sparsity(
            model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"block_comp PPL: {ppl}")

    per_layer = block_comp_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)
    print(f"achieved s_block mean={mean_sp:.4f}")

    result = {
        "model_name": args.model_name,
        "condition": args.condition,
        "p": args.p,
        "g": args.g,
        "rank": args.rank if args.condition in ("c7",) else None,
        "r_sk": args.r_sk if args.condition in ("c8", "c8a") else None,
        "factors_dir": args.factors_dir,
        "ppl": ppl,
        "achieved_s_block_mean": mean_sp,
        "achieved_s_block_per_layer": per_layer,
        "stats_dir": args.stats_dir,
        "git_commit": git_commit(),
    }
    print("RESULT_JSON " + json.dumps({k: v for k, v in result.items()
                                       if k != "achieved_s_block_per_layer"}))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
