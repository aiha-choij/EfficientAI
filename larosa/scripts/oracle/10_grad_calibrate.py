# coding=utf-8
# E-W0 gradient calibration: per-layer second moment of the LM-loss gradient
# at the MLP block output, w2[c] = E[(dL/dy_c)^2] for y = down_proj(...) in
# R^h. The vector w = sqrt(w2) is the output-side (loss-aligned) sensitivity
# the whitening round called for ("weight the OUTPUT side, not the input
# distribution") — consumed by 11_metric_check.py and, if validated, by
# weighted selection scores / weighted SVD factors.
#
# Parameters are frozen; the autograd graph is rooted at the embedding output,
# so only activation gradients are computed (no weight grads, ~fwd+bwd cost).
# Calibration corpus only — NEVER the evaluation set (leakage).
#
# Usage:
#   python scripts/oracle/10_grad_calibrate.py --model_name /raid/LLM/llama2-7b \
#       --tokens_pt .../stats/c4/calib_tokens.pt --nsamples 128 \
#       --out_dir .../grad_stats/c4
#   python scripts/oracle/10_grad_calibrate.py --model_name ... \
#       --dataset wikitext103 --nsamples 128 --out_dir .../grad_stats/wt103

import argparse
import json
import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(current_dir)

import torch
import transformers
from transformers import AutoTokenizer

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=parent_dir).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--tokens_pt", type=str, default=None,
                    help="reuse an existing calib_tokens.pt (exact-token reuse)")
    ap.add_argument("--dataset", type=str, default="c4",
                    choices=["c4", "wikitext103"],
                    help="corpus to build tokens from when --tokens_pt is absent")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
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
    for p in model.parameters():
        p.requires_grad_(False)

    if args.tokens_pt:
        tokens = torch.load(args.tokens_pt)[:args.nsamples]
        assert tokens.shape[1] == args.seqlen, tokens.shape
        source = args.tokens_pt
    else:
        from importlib import import_module
        calibrate = import_module("01_calibrate")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, use_fast=True, trust_remote_code=True)
        tokens = calibrate.build_calib_tokens(
            args.dataset, tokenizer, args.nsamples, args.seqlen, args.seed)
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(tokens, os.path.join(args.out_dir, "calib_tokens.pt"))
        source = args.dataset

    # root the graph at the embedding output (params stay frozen)
    def embed_hook(module, inp, out):
        out.requires_grad_(True)
        return out
    model.model.embed_tokens.register_forward_hook(embed_hook)

    # accumulate E[(dL/dy)^2] at each down_proj output, fp64 on CPU
    h = model.config.hidden_size
    sum_g2 = {li: torch.zeros(h, dtype=torch.float64) for li, _ in oracle_mlp.iter_mlps(model)}
    count = {li: 0 for li in sum_g2}

    def make_capture(li):
        def on_grad(g):
            g32 = g.detach().float().reshape(-1, g.shape[-1])
            sum_g2[li] += (g32 * g32).sum(0).cpu().double()  # cpu first: no fp64 on mps
            count[li] += g32.shape[0]
        def fwd_hook(module, inp, out):
            if out.requires_grad:
                out.register_hook(on_grad)
        return fwd_hook

    for li, mlp in oracle_mlp.iter_mlps(model):
        mlp.down_proj.register_forward_hook(make_capture(li))

    device = model.model.embed_tokens.weight.device
    for b in range(tokens.shape[0]):
        batch = tokens[b:b + 1].to(device)
        out = model(batch, labels=batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        if b % 16 == 0:
            print(f"grad calib sample {b}/{tokens.shape[0]} loss {out.loss.item():.4f}",
                  flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for li in sum_g2:
        w2 = (sum_g2[li] / max(count[li], 1)).float()
        torch.save({"w2": w2, "count": count[li]},
                   os.path.join(args.out_dir, f"layer_{li}.pt"))
    meta = {"model_name": args.model_name, "source": source,
            "nsamples": int(tokens.shape[0]), "seqlen": args.seqlen,
            "seed": args.seed, "git_commit": git_commit()}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved grad stats for {len(sum_g2)} layers to {args.out_dir}")


def load_w(grad_dir, layer_idx, device="cpu", eps_frac=1e-4):
    """w = sqrt(E[(dL/dy)^2]) with a floor at eps_frac * mean(w)."""
    d = torch.load(os.path.join(grad_dir, f"layer_{layer_idx}.pt"), map_location=device)
    w = d["w2"].float().clamp(min=0).sqrt()
    return w.clamp(min=eps_frac * w.mean())
