# coding=utf-8
# Build the compensation matrix M = W_down diag(g_bar) W_up per layer, take its
# SVD, and save rank-r factors A [r,h], B [h,r] for the C4 condition. Offline,
# once per (model, calibration corpus, rank).
#
# Usage:
#   python scripts/oracle/03_build_M.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir oracle_out/llama2-7b/stats/c4 --rank 512 \
#       --out_dir oracle_out/llama2-7b/factors/c4_r512

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

import torch
import transformers

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = "eager"  # weights only; no forward passes
    config.sparse_mode = "oracle"
    config.oracle_condition = "dense"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    oracle_mlp.load_stats(model, args.stats_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for layer_idx, mlp in oracle_mlp.iter_mlps(model):
            A, B, S = oracle_mlp.build_M_factors(mlp, args.rank)
            # rank-r spectral coverage: retained fraction of the Frobenius energy
            energy = (S[:args.rank] ** 2).sum() / (S ** 2).sum().clamp(min=1e-30)
            torch.save({"A": A.cpu(), "B": B.cpu(), "S": S.cpu(), "rank": args.rank},
                       os.path.join(args.out_dir, f"layer_{layer_idx}.pt"))
            print(f"layer {layer_idx}: rank {args.rank}/{S.shape[0]}, "
                  f"Frobenius energy retained {energy.item():.4f}", flush=True)
    print(f"saved factors to {args.out_dir}")
