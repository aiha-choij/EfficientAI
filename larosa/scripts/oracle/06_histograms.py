# coding=utf-8
# Magnitude histograms of i = u*g and r = u*(g - g_bar) for representative
# layers, on calibration data. Purpose: visualize whether r's mass is pulled
# harder toward zero than i's (the distribution-level H1 claim behind the
# phase-0 induced-sparsity curves).
#
# Bins are log10(|v| + 1e-8), fixed range [-8, 2], shared between i and r and
# across layers so the shapes are directly comparable. Also records |v|
# quantiles per layer.
#
# Usage:
#   python scripts/oracle/06_histograms.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir ~/workspace/oracle/llama2-7b/stats/c4 \
#       --layers 0 7 16 24 31 --nsamples 8 \
#       --out_json ~/workspace/oracle/llama2-7b/phase0/histograms.json

import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

import torch
import transformers

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp

BINS = 100
LOG_MIN, LOG_MAX = -8.0, 2.0
QUANTS = [0.25, 0.5, 0.75, 0.9, 0.99]


class HistProbe:
    def __init__(self, mlp, layer_idx):
        self.mlp = mlp
        self.layer_idx = layer_idx
        dev = mlp.down_proj.weight.device
        self.hist_i = torch.zeros(BINS, dtype=torch.float64, device=dev)
        self.hist_r = torch.zeros(BINS, dtype=torch.float64, device=dev)
        self.q_i = []
        self.q_r = []
        self._u = self._gate = None
        mlp.up_proj.register_forward_hook(lambda m, a, o: setattr(self, "_u", o))
        mlp.gate_proj.register_forward_hook(lambda m, a, o: setattr(self, "_gate", o))
        mlp.register_forward_hook(self._measure)

    def _measure(self, module, inp, out):
        u = self._u.float().reshape(-1, self._u.shape[-1])
        g = self.mlp.act_fn(self._gate).float().reshape(-1, self._gate.shape[-1])
        self._u = self._gate = None
        g_bar = self.mlp.oracle_g_bar.to(g.device)
        for vec, hist, qs in ((u * g, self.hist_i, self.q_i),
                              (u * (g - g_bar), self.hist_r, self.q_r)):
            mag = vec.abs()
            hist += torch.histc(torch.log10(mag + 1e-8), bins=BINS,
                                min=LOG_MIN, max=LOG_MAX).double()
            # stride-4 subsample keeps torch.quantile under its ~16.7M-element cap
            qs.append(torch.quantile(
                mag.reshape(-1)[::4], torch.tensor(QUANTS, device=mag.device)).cpu())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 7, 16, 24, 31])
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "oracle"
    config.oracle_condition = "dense"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    oracle_mlp.load_stats(model, args.stats_dir)

    probes = {idx: HistProbe(mlp, idx) for idx, mlp in oracle_mlp.iter_mlps(model)
              if idx in set(args.layers)}
    tokens = torch.load(os.path.join(args.stats_dir, "calib_tokens.pt"))[:args.nsamples]

    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for b in range(0, tokens.shape[0], args.batch_size):
            model(tokens[b:b + args.batch_size].to(device))
            print(f"hist sample {b}/{tokens.shape[0]}", flush=True)

    edges = torch.linspace(LOG_MIN, LOG_MAX, BINS + 1).tolist()
    result = {"bins_log10_edges": edges, "quantile_levels": QUANTS,
              "nsamples": args.nsamples, "stats_dir": args.stats_dir, "layers": {}}
    for idx, pr in sorted(probes.items()):
        q_i = torch.stack(pr.q_i).mean(0).tolist()
        q_r = torch.stack(pr.q_r).mean(0).tolist()
        result["layers"][str(idx)] = {
            "hist_i": pr.hist_i.cpu().tolist(),
            "hist_r": pr.hist_r.cpu().tolist(),
            "quantiles_abs_i": q_i,
            "quantiles_abs_r": q_r,
            "median_ratio_r_over_i": q_r[1] / max(q_i[1], 1e-30),
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out_json}")
    for idx in sorted(probes):
        d = result["layers"][str(idx)]
        print(f"layer {idx}: median|i|={d['quantiles_abs_i'][1]:.4g} "
              f"median|r|={d['quantiles_abs_r'][1]:.4g} "
              f"ratio r/i={d['median_ratio_r_over_i']:.3f}")
