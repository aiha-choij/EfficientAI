# coding=utf-8
# Build the compensation matrix M = W_down diag(g_bar) W_up per layer and save
# rank-r factors A [r,h], B [h,r] for the C4 condition.
#
# Modes (spec-c4-whitening.md):
#   plain          : SVD of M (original behavior)
#   --whiten       : SVD of M @ C, C = cholesky(Sigma + eps I) from the
#                    calibration input autocorrelation (01_calibrate --xxt);
#                    A folds C^{-1} back in via triangular solve
#   --alloc tau:T  : per-layer rank from cumulative spectral energy >= T
#   --alloc budget:R : bisect tau so mean rank ~= R
#
# Usage:
#   python scripts/oracle/03_build_M.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir .../stats/c4 --rank 512 --whiten --alloc budget:512 \
#       --out_dir .../factors/wht_alloc512 --spectra_out .../results/spectra_wht.json

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
    ap.add_argument("--rank", type=int, default=512, help="uniform rank (ignored with --alloc)")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--alloc", type=str, default=None,
                    help="'tau:0.95' or 'budget:512' for per-layer rank allocation")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--spectra_out", type=str, default=None,
                    help="write per-layer singular values + stable rank as JSON")
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

    with torch.no_grad():
        # pass 1: spectra (needed up front for allocation and diagnostics)
        spectra, stable = {}, {}
        for layer_idx, mlp in oracle_mlp.iter_mlps(model):
            sigma = (oracle_mlp.load_sigma(args.stats_dir, layer_idx,
                                           mlp.down_proj.weight.device)
                     if args.whiten else None)
            _, _, S = oracle_mlp.build_M_factors(mlp, 1, sigma=sigma)
            spectra[layer_idx] = S.cpu()
            stable[layer_idx] = float((S ** 2).sum() / (S ** 2).max())
            print(f"layer {layer_idx}: stable rank {stable[layer_idx]:.1f}", flush=True)

        if args.spectra_out:
            os.makedirs(os.path.dirname(args.spectra_out), exist_ok=True)
            with open(args.spectra_out, "w") as f:
                json.dump({"whiten": args.whiten,
                           "stable_rank": stable,
                           "spectra": {str(k): v.tolist() for k, v in spectra.items()}}, f)
            print(f"wrote {args.spectra_out}")

        ranks = None
        if args.alloc:
            mode, val = args.alloc.split(":")
            if mode == "tau":
                ranks = oracle_mlp.ranks_from_tau(spectra, float(val))
            elif mode == "budget":
                ranks = oracle_mlp.ranks_for_budget(spectra, int(val))
            else:
                raise ValueError(args.alloc)
            mean_r = sum(ranks.values()) / len(ranks)
            print(f"alloc {args.alloc}: mean rank {mean_r:.1f}, "
                  f"min {min(ranks.values())}, max {max(ranks.values())}")

        # pass 2: factors at the chosen ranks
        meta = oracle_mlp.save_factors(model, args.rank, args.out_dir,
                                       stats_dir=args.stats_dir,
                                       whiten=args.whiten, ranks=ranks)
    print(f"saved factors to {args.out_dir} "
          f"(whiten={args.whiten}, mean rank {meta['mean_rank']:.1f})")
