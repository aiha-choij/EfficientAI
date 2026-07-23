# coding=utf-8
# Phase-0 distribution report: early go/no-go signal for H1, computed on
# calibration data BEFORE any accuracy evaluation.
#
# Per layer, for both v = i (u*g) and v = r (u*(g - g_bar)):
#   - induced sparsity of top-p at p in {0.7, 0.8, 0.9, 0.95, 0.99}
#     (H1 preview: the r curve must sit ABOVE the i curve)
#   - Hoyer measure and kurtosis (per token, then averaged)
# Gate statistics:
#   - CV^2[j] = Var(g_j)/E[g_j]^2 distribution per layer (from saved stats)
#   - P(|g_j - g_bar_j| < |g_j|) (fraction where compensation wins termwise)
#   - optional Pearson(g_bar_A, g_bar_B) per layer across two calibration corpora
#
# Usage:
#   python scripts/oracle/02_distribution_report.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir oracle_out/llama2-7b/stats/c4 \
#       --stats_dir_b oracle_out/llama2-7b/stats/wikitext103 \
#       --nsamples 32 --out_dir oracle_out/llama2-7b/phase0

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

P_GRID = [0.7, 0.8, 0.9, 0.95, 0.99]


def hoyer(v):
    # v: [T, d] fp32 -> mean Hoyer over tokens
    d = v.shape[-1]
    l1 = v.abs().sum(-1)
    l2 = v.norm(dim=-1).clamp(min=1e-20)
    return ((d ** 0.5 - l1 / l2) / (d ** 0.5 - 1)).mean().item()


def kurtosis(v):
    mu = v.mean(-1, keepdim=True)
    c = v - mu
    m2 = (c ** 2).mean(-1).clamp(min=1e-20)
    m4 = (c ** 4).mean(-1)
    return (m4 / m2 ** 2).mean().item()


class LayerProbe:
    """Captures u and gate pre-activation per forward, accumulates metrics."""

    def __init__(self, mlp, layer_idx):
        self.mlp = mlp
        self.layer_idx = layer_idx
        self.n_batches = 0
        self.ind_sp_i = {p: 0.0 for p in P_GRID}
        self.ind_sp_r = {p: 0.0 for p in P_GRID}
        self.hoyer_i = self.hoyer_r = 0.0
        self.kurt_i = self.kurt_r = 0.0
        self.comp_wins = 0.0
        self._u = self._gate = None
        mlp.up_proj.register_forward_hook(self._grab_u)
        mlp.gate_proj.register_forward_hook(self._grab_gate)
        mlp.register_forward_hook(self._measure)

    def _grab_u(self, module, inp, out):
        self._u = out

    def _grab_gate(self, module, inp, out):
        self._gate = out

    def _measure(self, module, inp, out):
        u = self._u.float().reshape(-1, self._u.shape[-1])
        g = self.mlp.act_fn(self._gate).float().reshape(-1, self._gate.shape[-1])
        self._u = self._gate = None
        g_bar = self.mlp.oracle_g_bar.to(g.device)
        i_vec = u * g
        r_vec = u * (g - g_bar)
        for p in P_GRID:
            self.ind_sp_i[p] += 1.0 - oracle_mlp.top_p_mask(i_vec.abs(), p).float().mean().item()
            self.ind_sp_r[p] += 1.0 - oracle_mlp.top_p_mask(r_vec.abs(), p).float().mean().item()
        self.hoyer_i += hoyer(i_vec)
        self.hoyer_r += hoyer(r_vec)
        self.kurt_i += kurtosis(i_vec)
        self.kurt_r += kurtosis(r_vec)
        self.comp_wins += ((g - g_bar).abs() < g.abs()).float().mean().item()
        self.n_batches += 1

    def row(self):
        n = max(self.n_batches, 1)
        row = {
            "layer": self.layer_idx,
            "hoyer_i": self.hoyer_i / n, "hoyer_r": self.hoyer_r / n,
            "kurt_i": self.kurt_i / n, "kurt_r": self.kurt_r / n,
            "comp_win_frac": self.comp_wins / n,
        }
        for p in P_GRID:
            row[f"ind_sp_i_p{p}"] = self.ind_sp_i[p] / n
            row[f"ind_sp_r_p{p}"] = self.ind_sp_r[p] / n
        return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--stats_dir_b", type=str, default=None,
                    help="second-corpus stats for the g_bar Pearson stability check")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "oracle"
    config.oracle_condition = "dense"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    oracle_mlp.load_stats(model, args.stats_dir)
    oracle_mlp.attach_col_norms(model)

    tokens = torch.load(os.path.join(args.stats_dir, "calib_tokens.pt"))[:args.nsamples]
    probes = [LayerProbe(mlp, idx) for idx, mlp in oracle_mlp.iter_mlps(model)]

    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for s in range(0, tokens.shape[0], args.batch_size):
            model(tokens[s:s + args.batch_size].to(device))
            print(f"phase0 sample {s}/{tokens.shape[0]}", flush=True)

    rows = [pr.row() for pr in probes]

    # CV^2 per layer (median over neurons) from saved stats; optional Pearson
    for row in rows:
        d = torch.load(os.path.join(args.stats_dir, f"layer_{row['layer']}.pt"))
        g_bar, e_g2 = d["g_bar"].double(), d["e_g2"].double()
        cv2 = (e_g2 - g_bar ** 2).clamp(min=0) / (g_bar ** 2).clamp(min=1e-30)
        row["cv2_median"] = cv2.median().item()
        row["cv2_q90"] = cv2.quantile(0.9).item()
        if args.stats_dir_b:
            g_b = torch.load(os.path.join(args.stats_dir_b, f"layer_{row['layer']}.pt"))["g_bar"].double()
            ga, gb = g_bar - g_bar.mean(), g_b - g_b.mean()
            row["g_bar_pearson_ab"] = (
                (ga * gb).sum() / (ga.norm() * gb.norm()).clamp(min=1e-30)).item()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "phase0_report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # curve plots for 4 sample layers: first / early / middle / last
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        L = len(rows)
        sample_layers = sorted(set([0, L // 4, L // 2, L - 1]))
        fig, axes = plt.subplots(1, len(sample_layers), figsize=(4 * len(sample_layers), 3.5))
        for ax, li in zip(axes, sample_layers):
            row = rows[li]
            ax.plot(P_GRID, [row[f"ind_sp_i_p{p}"] for p in P_GRID], "o-", label="i = u*g")
            ax.plot(P_GRID, [row[f"ind_sp_r_p{p}"] for p in P_GRID], "s-", label="r = u*(g-g_bar)")
            ax.set_title(f"layer {li}")
            ax.set_xlabel("p")
            ax.set_ylabel("induced sparsity")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "phase0_curves.png"), dpi=150)
        print("wrote phase0_curves.png")
    except Exception as e:  # matplotlib may be missing on some nodes
        print(f"plot skipped: {e}")

    # headline: mean induced-sparsity gap (r - i) at p=0.9 across layers
    gap = sum(r_["ind_sp_r_p0.9"] - r_["ind_sp_i_p0.9"] for r_ in rows) / len(rows)
    print(f"H1 preview: mean induced-sparsity gap (r - i) at p=0.9 = {gap:+.4f} "
          f"({'GO signal' if gap > 0 else 'NO-GO signal'})")
