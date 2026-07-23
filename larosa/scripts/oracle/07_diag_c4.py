# coding=utf-8
# Task-0 diagnostics for the C4 compensation branch (spec-c4-whitening.md):
# per layer, on calibration data,
#   trunc_err[variant] = E_token || (M_hat - M) x ||_2   (compensation error)
#   tail_norm[s_op]    = E_token || ((1-m) * (g_bar*u)) @ W_down^T ||_2
#                        with m = top-K mask of the C3 score at operating s
#   ratio = trunc_err / tail_norm  (>1 => compensation error dominates the
#                                   signal it is supposed to replace)
# Spectra / stable rank come from 03_build_M --spectra_out files (offline).
#
# Usage:
#   python scripts/oracle/07_diag_c4.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir .../stats/c4 \
#       --factors plain=.../factors/r512 wht=.../factors/wht_r512 \
#       --s_op 0.7 0.9 --nsamples 4 --out_csv .../results/diag_trunc_vs_tail.csv

import argparse
import csv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

import torch
import transformers

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp


class DiagProbe:
    def __init__(self, mlp, layer_idx, variants, s_ops):
        self.mlp = mlp
        self.layer_idx = layer_idx
        self.variants = variants          # {name: (A, B) on device}
        self.s_ops = s_ops
        self.n = 0
        self.trunc = {k: 0.0 for k in variants}
        self.tail = {s: 0.0 for s in s_ops}
        self._x = self._u = self._gate = None
        mlp.up_proj.register_forward_hook(lambda m, a, o: setattr(self, "_u", o))
        mlp.gate_proj.register_forward_hook(lambda m, a, o: setattr(self, "_gate", o))
        mlp.register_forward_pre_hook(lambda m, a: setattr(self, "_x", a[0]))
        mlp.register_forward_hook(self._measure)

    @torch.no_grad()
    def _measure(self, module, inp, out):
        mlp = self.mlp
        x = self._x.float().reshape(-1, self._x.shape[-1])
        u = self._u.float().reshape(-1, self._u.shape[-1])
        g = mlp.act_fn(self._gate).float().reshape(-1, self._gate.shape[-1])
        self._x = self._u = self._gate = None
        M = oracle_mlp.compute_M(mlp)
        g_bar = mlp.oracle_g_bar.to(x.device)
        w_down = mlp.down_proj.weight.float()
        for name, (A, B) in self.variants.items():
            err = x @ (B @ A - M).T
            self.trunc[name] += err.norm(dim=-1).mean().item()
        score = (u * (g - g_bar)).abs() * mlp.oracle_col_norm
        gu = g_bar * u
        for s in self.s_ops:
            m = oracle_mlp.top_k_mask(score, s).float()
            tail = ((1.0 - m) * gu) @ w_down.T
            self.tail[s] += tail.norm(dim=-1).mean().item()
        self.n += 1

    def rows(self):
        n = max(self.n, 1)
        out = []
        for s in self.s_ops:
            tail = self.tail[s] / n
            row = {"layer": self.layer_idx, "s_op": s, "tail_norm": tail}
            for name in self.variants:
                te = self.trunc[name] / n
                row[f"trunc_err_{name}"] = te
                row[f"ratio_{name}"] = te / max(tail, 1e-30)
            out.append(row)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--factors", type=str, nargs="+", required=True,
                    help="name=dir pairs of factor variants to compare")
    ap.add_argument("--s_op", type=float, nargs="+", default=[0.7, 0.9])
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--out_csv", type=str, required=True)
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
    oracle_mlp.attach_col_norms(model)

    variant_dirs = dict(kv.split("=", 1) for kv in args.factors)
    probes = []
    for layer_idx, mlp in oracle_mlp.iter_mlps(model):
        dev = mlp.down_proj.weight.device
        variants = {}
        for name, vdir in variant_dirs.items():
            d = torch.load(os.path.join(vdir, f"layer_{layer_idx}.pt"), map_location=dev)
            variants[name] = (d["A"].float(), d["B"].float())
        probes.append(DiagProbe(mlp, layer_idx, variants, args.s_op))

    tokens = torch.load(os.path.join(args.stats_dir, "calib_tokens.pt"))[:args.nsamples]
    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for b in range(tokens.shape[0]):
            model(tokens[b:b + 1].to(device))
            print(f"diag sample {b}/{tokens.shape[0]}", flush=True)

    rows = [r for pr in probes for r in pr.rows()]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}")
    for s in args.s_op:
        for name in variant_dirs:
            rs = [r for r in rows if r["s_op"] == s]
            worst = max(rs, key=lambda r: r[f"ratio_{name}"])
            mean_ratio = sum(r[f"ratio_{name}"] for r in rs) / len(rs)
            print(f"s_op={s} variant={name}: mean trunc/tail ratio {mean_ratio:.3f}, "
                  f"worst layer {worst['layer']} ({worst[f'ratio_{name}']:.3f})")
