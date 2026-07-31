# coding=utf-8
# L2 build: sequential (GPTQ-style single sweep) refit against the SPARSE
# stream. Layer ell in order 0..L-1: (G, C) accumulated from the sparse
# stream's actual masked input at that layer (mask recomputed against it,
# using the frozen ORIGINAL col_norm -- never the refit weight) vs the
# ORIGINAL dense model's output at the same layer/tokens (teacher, always
# dense regardless of what's happened to the sparse stream so far); solve;
# install; advance the sparse stream through the now-refit layer to seed
# layer ell+1. Attention is dense/unmodified in both streams throughout.
#
# Cost: O(L) total layer-equivalent forward passes (one dense + one sparse
# accumulation pass + one cheap sparse "apply" pass per layer), NOT O(L^2)
# -- each layer needs its own extra pass, not a full-model replay, because
# the previous layer's OUTPUT is cached and reused as this layer's input
# (dense_hidden / sparse_hidden below), never recomputed from scratch.
#
# Memory: dense_hidden and sparse_hidden are [nsamples, seqlen, h] tensors
# kept on CPU (model dtype, e.g. bf16) between layers -- O(N*h), not
# O(L*N*h). Default nsamples is smaller than L1's (128 vs 512): L2 costs
# roughly 3x a plain dense forward PER LAYER (teacher + student-accumulate
# + student-apply), so a smaller calibration set keeps wall-clock
# reasonable; document this deviation in the journal card.
#
# Usage:
#   python scripts/refit/02_build_l2.py --model_name /raid/LLM/llama2-7b \
#       --s 0.9 --g 1 --nsamples 128 --seqlen 2048 \
#       --out_dir refit_out/llama2-7b/weights/l2_s0.9_g1

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
    ap.add_argument("--dataset", type=str, default="wikitext103", choices=["c4", "wikitext103"])
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--chunk_size", type=int, default=4, help="sequences per GPU forward chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s", type=float, required=True, help="target sparsity (exact-K)")
    ap.add_argument("--g", type=int, required=True, help="token block size sharing a mask")
    ap.add_argument("--lam", type=float, default=0.01,
                    help="ridge lambda -- ONE value per run (unlike L1, L2's solve is "
                         "baked into the sequential sweep, no cheap post-hoc resweep)")
    ap.add_argument("--calib_tokens", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True,
                    help="weights land in {out_dir}_lam{lam}")
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "refit"
    config.refit_mode = "l2"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    tok_path = args.calib_tokens or os.path.join(os.path.dirname(args.out_dir), "calib_tokens_l2.pt")
    if os.path.exists(tok_path):
        tokens = refit_mlp.load_calib_tokens(tok_path, args.nsamples, args.seqlen)
        print(f"reusing saved calibration tokens: {tok_path}")
    else:
        tokens = common.build_calib_tokens(args.dataset, tokenizer, args.nsamples, args.seqlen, args.seed)
        refit_mlp.save_calib_tokens(tokens, tok_path)
        print(f"saved calibration tokens: {tok_path}")

    refit_mlp.attach_col_norms(model)

    device = model.model.embed_tokens.weight.device
    num_layers = model.config.num_hidden_layers
    out_dir_lam = f"{args.out_dir}_lam{args.lam}"
    os.makedirs(out_dir_lam, exist_ok=True)

    with torch.no_grad():
        embeds = model.model.embed_tokens(tokens.to(device)).to("cpu")
        dense_hidden = embeds
        sparse_hidden = embeds.clone()

        for layer_idx in range(num_layers):
            refit_mlp.enable_l2_layer_collect(model, layer_idx, s=args.s, g=args.g)
            dense_next = torch.empty_like(dense_hidden)

            # Teacher and student calls MUST alternate per chunk (not two
            # separate full passes): refit_last_y_star only holds the most
            # recent teacher call's output, so the paired student call has
            # to run immediately after, on the SAME chunk, or it would
            # accumulate (G, C) against a stale/mismatched target.
            for i in range(0, args.nsamples, args.chunk_size):
                refit_mlp.set_l2_role(model, layer_idx, "teacher")
                chunk = dense_hidden[i:i + args.chunk_size].to(device)
                out = refit_mlp.single_layer_forward(model, layer_idx, chunk)
                dense_next[i:i + args.chunk_size] = out.to("cpu")

                refit_mlp.set_l2_role(model, layer_idx, "student")
                sparse_chunk = sparse_hidden[i:i + args.chunk_size].to(device)
                refit_mlp.single_layer_forward(model, layer_idx, sparse_chunk)  # side effect only

            stats = refit_mlp.finalize_l2_layer(model, layer_idx)
            w_tilde = refit_mlp.solve_refit(stats["G"], stats["C"], lam=args.lam)
            mlp = model.model.layers[layer_idx].mlp
            mlp.down_proj.weight.data.copy_(w_tilde.to(mlp.down_proj.weight.dtype))
            torch.save({"W_down": w_tilde.cpu()}, os.path.join(out_dir_lam, f"layer_{layer_idx}.pt"))

            refit_mlp.set_l2_role(model, layer_idx, "apply")
            sparse_next = torch.empty_like(sparse_hidden)
            for i in range(0, args.nsamples, args.chunk_size):
                chunk = sparse_hidden[i:i + args.chunk_size].to(device)
                out = refit_mlp.single_layer_forward(model, layer_idx, chunk)
                sparse_next[i:i + args.chunk_size] = out.to("cpu")

            dense_hidden, sparse_hidden = dense_next, sparse_next
            print(f"layer {layer_idx + 1}/{num_layers} done "
                  f"(n={stats['n']}, mean|diag(G)|={stats['G'].diagonal().mean().item():.4e})",
                  flush=True)

    meta = {
        "model_name": args.model_name, "mode": "l2", "s": args.s, "g": args.g,
        "lam": args.lam, "dataset": args.dataset, "nsamples": args.nsamples,
        "seqlen": args.seqlen, "seed": args.seed,
        "calib_tokens": tok_path,
        "git_commit": common.git_commit(parent_dir),
    }
    with open(os.path.join(out_dir_lam, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved L2 refit weights to {out_dir_lam}")
