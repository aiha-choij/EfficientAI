# coding=utf-8
# L1 build: one dense calibration sweep accumulates per-layer (G, C) against
# a frozen C2-score mask (s, g); solve the closed-form ridge refit of
# down_proj for one or more lambda values (shape unchanged -> zero runtime
# cost). Every layer's regression uses the DENSE model's own activations
# (L1's definition -- no cross-layer coupling, unlike L2).
#
# Usage:
#   python scripts/refit/01_build_l1.py --model_name /raid/LLM/llama2-7b \
#       --s 0.9 --g 1 --nsamples 512 --seqlen 2048 \
#       --out_dir refit_out/llama2-7b/weights/l1_s0.9_g1_lam0.01

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
from transformers import AutoTokenizer

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, refit_mlp
import common

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="c4", choices=["c4", "wikitext103"])
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s", type=float, required=True, help="target sparsity (exact-K)")
    ap.add_argument("--g", type=int, required=True, help="token block size sharing a mask")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01],
                    help="one --out_dir_lamX per value; ridge lambda sweep (dev models only)")
    ap.add_argument("--calib_tokens", type=str, default=None,
                    help="path to reuse/save the exact calibration token tensor")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="base dir; per-lambda weights go in {out_dir}_lam{L}")
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "refit"
    config.refit_mode = "l1"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    tok_path = args.calib_tokens or os.path.join(os.path.dirname(args.out_dir), "calib_tokens.pt")
    if os.path.exists(tok_path):
        tokens = refit_mlp.load_calib_tokens(tok_path, args.nsamples, args.seqlen)
        print(f"reusing saved calibration tokens: {tok_path}")
    else:
        tokens = common.build_calib_tokens(args.dataset, tokenizer, args.nsamples, args.seqlen, args.seed)
        refit_mlp.save_calib_tokens(tokens, tok_path)
        print(f"saved calibration tokens: {tok_path}")

    refit_mlp.attach_col_norms(model)
    refit_mlp.enable_l1_collect_mode(model, s=args.s, g=args.g)

    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for i in range(0, args.nsamples, args.batch_size):
            batch = tokens[i:i + args.batch_size].to(device)
            model(batch)
            if i % 32 == 0:
                print(f"calib sample {i}/{args.nsamples}", flush=True)

    stats = refit_mlp.finalize_l1(model)

    for lam in args.lambdas:
        weights = {layer_idx: refit_mlp.solve_refit(st["G"], st["C"], lam=lam)
                   for layer_idx, st in stats.items()}
        out_dir_lam = f"{args.out_dir}_lam{lam}"
        meta = {
            "model_name": args.model_name, "mode": "l1", "s": args.s, "g": args.g,
            "lam": lam, "dataset": args.dataset, "nsamples": args.nsamples,
            "seqlen": args.seqlen, "seed": args.seed,
            "calib_tokens": tok_path,
            "git_commit": common.git_commit(parent_dir),
        }
        refit_mlp.save_refit_weights(out_dir_lam, weights, meta=meta)
        print(f"saved L1 refit weights (lambda={lam}) to {out_dir_lam}")
