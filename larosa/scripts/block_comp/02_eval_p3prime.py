# coding=utf-8
# Phase 4 (P3') evaluation: block_comp's C7a/C7/C8a/C8 compensation
# conditions, but with mask selection quantized to coactivation-block-
# structure's P2 PPMI-clustered neuron blocks (instead of the unstructured
# top-p over individual neurons that scripts/block_comp/01_eval_ppl.py
# uses). The token-block axis (g) is identical to Phase 1-3; the neuron-
# block axis (B, from the partition file) is new -- together they form
# the spec's "2D tile" (see block_comp_mlp.py's block_p3_mask /
# .labtool/topics/block-sparse-compensation/journal/
# 2026-07-31_init-phase4-p3prime.md for the design).
#
# Budget convention matches coactivation-block-structure's P3 exactly
# (K = round((1-sparsity)*d), m = round(K/B)) so PPL numbers here are
# directly comparable to that topic's existing clustered/random-block
# (no-compensation) results at the same (B, g, sparsity).
#
# Usage:
#   python scripts/block_comp/02_eval_p3prime.py --model_name /raid/LLM/llama2-7b \
#       --condition c8 --g 16 --B 64 --sparsity 0.9 --r_sk 1024 \
#       --partitions ~/workspace/analysis/llama2_p3_partitions_s09.pt \
#       --stats_dir oracle_out/llama2-7b/stats/wikitext103 \
#       --out_json block_comp_out/llama2-7b/results/p3prime_c8_g16_B64.json

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
    ap.add_argument("--g", type=int, default=16, help="token-block size")
    ap.add_argument("--B", type=int, required=True, help="neuron-block size (must exist in the partition file)")
    ap.add_argument("--sparsity", type=float, default=0.9,
                    help="target overall sparsity; K=round((1-sparsity)*d), m=round(K/B) -- "
                         "same formula as coactivation-block-structure's P3, for direct comparability")
    ap.add_argument("--rank", type=int, default=None, help="C7 comp_lr rank (M = W_down diag(g_bar) W_up)")
    ap.add_argument("--r_sk", type=int, default=None, help="C8/C8a gate/up/down sketch rank (single knob)")
    ap.add_argument("--partitions", type=str, required=True,
                    help="p3_collect_cluster_all.py output (coactivation-block-structure P2/P3 artifact)")
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

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    oracle_mlp.attach_col_norms(model)
    oracle_mlp.load_stats(model, args.stats_dir)  # attaches oracle_g_bar (residual score needs it, all conditions)

    device = next(model.parameters()).device
    d = config.intermediate_size
    K = round((1.0 - args.sparsity) * d)
    m_keep = max(1, round(K / args.B))
    neuron_partitions = block_comp_mlp.build_neuron_partition_onehots(
        os.path.expanduser(args.partitions), args.B, device)

    if args.condition != "c7a":
        if args.factors_dir:
            block_comp_mlp.load_block_factors(model, args.factors_dir)
        else:
            block_comp_mlp.attach_block_factors_inplace(model, rank=args.rank, r_sk=args.r_sk,
                                                        condition=args.condition)

    block_comp_mlp.set_condition(model, args.condition, g=args.g,
                                  neuron_partitions=neuron_partitions, neuron_m=m_keep)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("=" * 40)
    print(f"P3' condition={args.condition} g={args.g} B={args.B} sparsity={args.sparsity} "
          f"K={K} m={m_keep} (budget {m_keep * args.B}/{d}={m_keep * args.B / d:.4f}) "
          f"rank={args.rank} r_sk={args.r_sk}")
    with torch.no_grad():
        ppl = eval_ppl_wikitext_with_inference_sparsity(
            model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"P3' PPL: {ppl}")

    per_layer = block_comp_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)
    print(f"achieved s_block mean={mean_sp:.4f}")

    result = {
        "model_name": args.model_name,
        "condition": args.condition,
        "g": args.g,
        "B": args.B,
        "sparsity": args.sparsity,
        "K": K,
        "m_keep": m_keep,
        "rank": args.rank if args.condition in ("c7",) else None,
        "r_sk": args.r_sk if args.condition in ("c8", "c8a") else None,
        "factors_dir": args.factors_dir,
        "partitions": args.partitions,
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
