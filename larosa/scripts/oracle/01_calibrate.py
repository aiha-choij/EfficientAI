# coding=utf-8
# Phase-2 calibration: accumulate per-neuron gate statistics (g_bar, g_bar_star,
# E[g^2]) over a general corpus, fp32/fp64 accumulation, and save per-layer
# stats plus the exact calibration token tensor for reproducibility.
#
# Usage:
#   python scripts/oracle/01_calibrate.py --model_name /raid/LLM/llama2-7b \
#       --dataset c4 --nsamples 512 --seqlen 2048 --out_dir oracle_out/llama2-7b/stats/c4

import argparse
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


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=parent_dir).decode().strip()
    except Exception:
        return "unknown"


def build_calib_tokens(dataset_name, tokenizer, nsamples, seqlen, seed):
    """Deterministic [nsamples, seqlen] token tensor from a general corpus."""
    need = nsamples * seqlen
    buf = []
    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for ex in ds:
            ids = tokenizer(ex["text"]).input_ids
            buf.extend(ids)
            if len(buf) >= need:
                break
    elif dataset_name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(ds), generator=g).tolist()
        for k in order:
            text = ds[k]["text"]
            if not text.strip():
                continue
            buf.extend(tokenizer(text).input_ids)
            if len(buf) >= need:
                break
    else:
        raise ValueError(dataset_name)
    assert len(buf) >= need, f"corpus exhausted: {len(buf)} < {need} tokens"
    return torch.tensor(buf[:need], dtype=torch.long).reshape(nsamples, seqlen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="c4", choices=["c4", "wikitext103"])
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "oracle"
    config.oracle_condition = "dense"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    tokens = build_calib_tokens(args.dataset, tokenizer, args.nsamples, args.seqlen, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(tokens, os.path.join(args.out_dir, "calib_tokens.pt"))

    oracle_mlp.enable_stats_mode(model)
    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for s in range(0, args.nsamples, args.batch_size):
            batch = tokens[s:s + args.batch_size].to(device)
            model(batch)
            if s % 32 == 0:
                print(f"calib sample {s}/{args.nsamples}", flush=True)

    stats = oracle_mlp.finalize_stats(model)
    meta = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "git_commit": git_commit(),
    }
    oracle_mlp.save_stats(stats, args.out_dir, meta=meta)
    print(f"saved stats for {len(stats)} layers to {args.out_dir}")
