# coding=utf-8
# Direction-resolved compensation-error analysis: where in input space does
# each factorization put its error, relative to the input covariance?
#
# For sample layers: eigendecompose Sigma = V diag(lambda) V^T (lambda
# ascending), and for each factor variant compute the per-direction error
#   n_k = || (B A - M) v_k ||_2
# then aggregate by eigenvalue decile (D1 = lowest-variance directions,
# D10 = highest). Whitening minimizes sum_k lambda_k n_k^2 (the input-L2
# objective), so it should trade LOWER error on high-lambda deciles for
# HIGHER error on low-lambda deciles vs plain SVD. If PPL nevertheless
# worsens, loss-relevant signal lives in the low-variance directions.
#
# Offline: weights + stats + factors only, no forwards.

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--factors", type=str, nargs="+", required=True, help="name=dir pairs")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 7, 16, 24, 31])
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = "eager"
    config.sparse_mode = "oracle"
    config.oracle_condition = "dense"
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    oracle_mlp.load_stats(model, args.stats_dir)

    variant_dirs = dict(kv.split("=", 1) for kv in args.factors)
    result = {"layers": {}, "deciles": 10, "note": "decile 1 = lowest input variance"}
    with torch.no_grad():
        for layer_idx, mlp in oracle_mlp.iter_mlps(model):
            if layer_idx not in set(args.layers):
                continue
            dev = mlp.down_proj.weight.device
            M = oracle_mlp.compute_M(mlp)
            sigma = oracle_mlp.load_sigma(args.stats_dir, layer_idx, dev)
            lam, V = torch.linalg.eigh(sigma)      # ascending
            h = lam.shape[0]
            edges = [int(h * i / 10) for i in range(11)]
            entry = {"eig_decile_mean_lambda":
                     [float(lam[edges[i]:edges[i + 1]].mean()) for i in range(10)],
                     "variants": {}}
            for name, vdir in variant_dirs.items():
                d = torch.load(os.path.join(vdir, f"layer_{layer_idx}.pt"), map_location=dev)
                E = (d["B"].float() @ d["A"].float() - M) @ V   # [h, h]; col k = err on v_k
                n = E.norm(dim=0)                               # [h]
                per_decile = [float(n[edges[i]:edges[i + 1]].mean()) for i in range(10)]
                # expected E||err x||^2 under N(0, Sigma): sum_k lambda_k n_k^2
                expected_l2 = float((lam * n ** 2).sum().sqrt())
                entry["variants"][name] = {"decile_err": per_decile,
                                           "expected_rms_err": expected_l2,
                                           "rank": int(d["rank"])}
            result["layers"][str(layer_idx)] = entry
            print(f"layer {layer_idx}: " + "; ".join(
                f"{k}: D1(low-var) {v['decile_err'][0]:.3f} vs D10(high-var) {v['decile_err'][9]:.3f}"
                for k, v in entry["variants"].items()), flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out_json}")
