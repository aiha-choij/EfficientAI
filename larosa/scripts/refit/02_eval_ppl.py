# coding=utf-8
# L0/L1/L2 PPL evaluation. Reuses the exact wikitext-2 PPL pipeline
# (eval_ppl_wikitext_with_inference_sparsity) from larosa-repro /
# oracle-residual-sparsity so numbers are directly comparable to the trusted
# dense/C1 anchors.
#
# NOTE the known upstream label swap in that eval (see gist "Pitfalls"):
# printed "attn h1/h2" lines are really MLP values. This script reads
# per-layer sparsity directly from refit_mlp.achieved_sparsity_per_layer,
# not the printed lines.
#
# Usage:
#   python scripts/refit/02_eval_ppl.py --model_name /raid/LLM/llama2-7b \
#       --mode l0 --s 0.9 --g 1 \
#       --out_json refit_out/llama2-7b/results/l0_s0.9_g1.json
#   python scripts/refit/02_eval_ppl.py --model_name /raid/LLM/llama2-7b \
#       --mode l1 --s 0.9 --g 1 --weights_dir refit_out/llama2-7b/weights/l1_s0.9_g1_lam0.01 \
#       --out_json refit_out/llama2-7b/results/l1_s0.9_g1_lam0.01.json

import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(current_dir)

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, refit_mlp
from utils.eval_ppl import eval_ppl_wikitext_with_inference_sparsity
import common

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--mode", type=str, required=True, choices=list(refit_mlp.MODES))
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--g", type=int, required=True)
    ap.add_argument("--weights_dir", type=str, default=None, help="required for l1/l2")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    if args.mode in ("l1", "l2") and not args.weights_dir:
        ap.error(f"--weights_dir is required for mode={args.mode}")

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "refit"
    config.refit_mode = args.mode
    config.refit_s = args.s
    config.refit_g = args.g

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    refit_mlp.attach_col_norms(model)  # ALWAYS from the original weight, before any load
    if args.weights_dir:
        refit_mlp.load_refit_weights(model, args.weights_dir)
    refit_mlp.set_condition(model, args.mode, s=args.s, g=args.g)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("=" * 40)
    print(f"refit mode={args.mode} s={args.s} g={args.g} weights_dir={args.weights_dir}")
    with torch.no_grad():
        ppl = eval_ppl_wikitext_with_inference_sparsity(
            model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"Refit PPL: {ppl}")

    per_layer = refit_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)
    print(f"achieved sparsity mean={mean_sp:.4f}")

    weights_meta = None
    if args.weights_dir:
        meta_path = os.path.join(args.weights_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                weights_meta = json.load(f)

    result = {
        "model_name": args.model_name,
        "mode": args.mode,
        "s": args.s,
        "g": args.g,
        "weights_dir": args.weights_dir,
        "weights_meta": weights_meta,
        "ppl": ppl,
        "achieved_sparsity_mean": mean_sp,
        "achieved_sparsity_per_layer": per_layer,
        "git_commit": common.git_commit(parent_dir),
    }
    print("RESULT_JSON " + json.dumps({k: v for k, v in result.items()
                                       if k != "achieved_sparsity_per_layer"}))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
