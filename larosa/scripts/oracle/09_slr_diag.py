# coding=utf-8
# E0 offline diagnostics for the H4 sparse+low-rank (SLR) compensation
# (gist Next Experiments 1; steer card 2026-07-27_pivot-c4-slr-compensation):
#
#   D1 (S1 gate, weights+stats only): per layer, spectra of M_cold after
#      removing the top-hot_n rank-1 neuron terms — r90 and Frobenius energy
#      captured at reference ranks, vs plain M (hot_n=0).
#   D2 (S2 gate + matched-budget screening, needs calibration x): per layer,
#      (a) concentration of per-token input-channel energy (top-k |x|^2 share);
#      (b) relative compensation error E||Mx - comp(x)|| / E||Mx|| for the
#          plain-LR baseline and the S1/S2 budget splits at equal MACs
#          (B_eff: rank 1 == 1 hot neuron == 2 input channels).
#
# SCREENING ONLY: this is an input-L2 metric, which the whitening round proved
# can invert vs downstream loss across geometries. Use it to prune arms and
# order variants within a family; PPL (E1) is the referee.
#
# Usage:
#   python scripts/oracle/09_slr_diag.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir .../stats/c4 --nsamples 8 --out_json .../results/slr_diag.json

import argparse
import json
import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

import torch
import transformers

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=parent_dir).decode().strip()
    except Exception:
        return "unknown"


def parse_splits(spec):
    """'768:256,512:512' -> [(768, 256), (512, 512)] as (rank, other) pairs."""
    return [tuple(int(v) for v in pair.split(":")) for pair in spec.split(",") if pair]


def spectrum_stats(S, energy_ranks):
    e = S.float().cpu().double() ** 2  # cpu: fp64 unsupported on mps
    c = torch.cumsum(e, 0) / e.sum()
    r90 = int((c < 0.90).sum().item()) + 1
    return {"r90": r90,
            "energy_at": {str(r): float(c[min(r, c.shape[0]) - 1]) for r in energy_ranks}}


def tail_error(x, U, S, Vh, r, denom):
    """mean ||(target - trunc_r(target)) x|| / denom via the SVD tail."""
    if r >= S.shape[0]:
        return 0.0
    xt = x @ Vh[r:, :].T
    err = (xt * S[r:]) @ U[:, r:].T
    return float(err.norm(dim=-1).mean().item() / denom)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--nsamples", type=int, default=8,
                    help="calibration sequences for the x-capture pass")
    ap.add_argument("--capture_tokens", type=int, default=16384,
                    help="tokens kept per layer for D2 (fp16 CPU buffer)")
    ap.add_argument("--budget", type=int, default=1024,
                    help="equivalent-rank MAC budget B_eff for the split arms")
    ap.add_argument("--hot_grid", type=str, default="0,256,512,768,1024,1376",
                    help="hot_n values for D1 spectra (0 = plain M)")
    ap.add_argument("--s1_splits", type=str, default="768:256,512:512,256:768,0:1024",
                    help="S1 arms as rank:hot_n at B_eff = rank + hot_n")
    ap.add_argument("--s2_splits", type=str, default="768:512,512:1024,256:1536,0:2048",
                    help="S2 arms as rank:k at B_eff = rank + k/2")
    ap.add_argument("--conc_ks", type=str, default="41,256,512,1024,1536,2048",
                    help="k values for the |x|^2 top-k concentration curve")
    ap.add_argument("--energy_ranks", type=str, default="256,512,1024,2048")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    hot_grid = [int(v) for v in args.hot_grid.split(",")]
    s1_splits = parse_splits(args.s1_splits)
    s2_splits = parse_splits(args.s2_splits)
    conc_ks = [int(v) for v in args.conc_ks.split(",")]
    energy_ranks = [int(v) for v in args.energy_ranks.split(",")]
    for r, hn in s1_splits:
        assert r + hn == args.budget, f"S1 split {r}:{hn} != budget {args.budget}"
    for r, k in s2_splits:
        assert r + k // 2 == args.budget, f"S2 split {r}:{k} != budget {args.budget}"
    # every S1 hot_n must have its M_cold SVD available from the D1 grid
    need_hot = sorted({hn for _, hn in s1_splits} | set(hot_grid))

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

    # ---- capture pass: x entering each MLP, fp16 on CPU -------------------
    buffers = {li: [] for li, _ in oracle_mlp.iter_mlps(model)}
    counts = {li: 0 for li in buffers}

    def make_hook(li):
        def hook(module, inp):
            if counts[li] < args.capture_tokens:
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                take = min(x.shape[0], args.capture_tokens - counts[li])
                buffers[li].append(x[:take].half().cpu())
                counts[li] += take
        return hook

    handles = [mlp.register_forward_pre_hook(make_hook(li))
               for li, mlp in oracle_mlp.iter_mlps(model)]
    tokens = torch.load(os.path.join(args.stats_dir, "calib_tokens.pt"))[:args.nsamples]
    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for b in range(tokens.shape[0]):
            model(tokens[b:b + 1].to(device))
            print(f"capture sample {b + 1}/{tokens.shape[0]}", flush=True)
    for h in handles:
        h.remove()

    # ---- per-layer offline analysis ---------------------------------------
    layers_out = {}
    with torch.no_grad():
        for li, mlp in oracle_mlp.iter_mlps(model):
            dev = mlp.down_proj.weight.device
            x = torch.cat(buffers[li]).float().to(dev)
            buffers[li] = None
            M = oracle_mlp.compute_M(mlp)
            ref = x @ M.T
            denom = ref.norm(dim=-1).mean().item()
            rec = {"denom_mean_Mx_norm": denom, "tokens": int(x.shape[0])}

            # D2a: per-token channel-energy concentration of the MLP input
            en = (x * x)
            sorted_en, _ = torch.sort(en, dim=-1, descending=True)
            csum = torch.cumsum(sorted_en, dim=-1)
            total = csum[:, -1:]
            rec["x_topk_energy_share"] = {
                str(k): float((csum[:, k - 1] / total.squeeze(1)).mean().item())
                for k in conc_ks}
            del en, sorted_en, csum, total

            # D1 + S1 arms: one SVD of M_cold per hot_n (hot_n=0 -> plain M)
            svd_cache = {}
            rec["d1"] = {}
            for hn in need_hot:
                if hn == 0:
                    target = M
                    hot_idx = None
                else:
                    hot_idx = torch.topk(oracle_mlp.hot_scores(mlp), hn).indices
                    target = oracle_mlp.compute_M_cold(mlp, hot_idx)
                U, S, Vh = torch.linalg.svd(target, full_matrices=False)
                svd_cache[hn] = (U, S, Vh)
                if hn in hot_grid:
                    rec["d1"][str(hn)] = spectrum_stats(S, energy_ranks)
            rec["arms"] = {}
            U0, S0, Vh0 = svd_cache[0]
            rec["arms"][f"lr_r{args.budget}"] = tail_error(x, U0, S0, Vh0,
                                                           args.budget, denom)
            for r, hn in s1_splits:
                U, S, Vh = svd_cache[hn]
                rec["arms"][f"s1_r{r}_h{hn}"] = tail_error(x, U, S, Vh, r, denom)

            # S2 arms: R_r = M - trunc_r(M); err = R_r ((1 - m_x) * x)
            xabs = x.abs()
            for r, k in s2_splits:
                Rr = (U0[:, r:] * S0[r:]) @ Vh0[r:, :] if r < S0.shape[0] \
                    else torch.zeros_like(M)
                for score_name in ("abs", "wnorm"):
                    score = xabs if score_name == "abs" else xabs * Rr.norm(dim=0)
                    m_x = oracle_mlp.top_count_mask(score, k).float()
                    err = ((1.0 - m_x) * x) @ Rr.T
                    rec["arms"][f"s2_r{r}_k{k}_{score_name}"] = \
                        float(err.norm(dim=-1).mean().item() / denom)
                del Rr
            del svd_cache, x, ref, xabs
            torch.cuda.empty_cache()
            layers_out[str(li)] = rec
            print(f"layer {li}: lr_r{args.budget} "
                  f"{rec['arms'][f'lr_r{args.budget}']:.4f} | " +
                  " ".join(f"{k} {v:.4f}" for k, v in rec["arms"].items()
                           if not k.startswith("lr")), flush=True)

    # ---- aggregate summary -------------------------------------------------
    arm_names = list(next(iter(layers_out.values()))["arms"].keys())
    mid = [str(i) for i in range(4, 18) if str(i) in layers_out]
    summary = {}
    for a in arm_names:
        vals = {li: rec["arms"][a] for li, rec in layers_out.items()}
        summary[a] = {
            "mean": sum(vals.values()) / len(vals),
            "mid_stack_mean": sum(vals[li] for li in mid) / max(len(mid), 1),
            "worst_layer": max(vals, key=vals.get),
            "worst": max(vals.values()),
        }
    d1_summary = {}
    e_key = str(max(energy_ranks))
    for hn in hot_grid:
        r90s = [rec["d1"][str(hn)]["r90"] for rec in layers_out.values()]
        e_top = [rec["d1"][str(hn)]["energy_at"][e_key]
                 for rec in layers_out.values()]
        d1_summary[str(hn)] = {"mean_r90": sum(r90s) / len(r90s),
                               "max_r90": max(r90s),
                               f"mean_energy_at_{e_key}": sum(e_top) / len(e_top)}

    out = {"model_name": args.model_name, "stats_dir": args.stats_dir,
           "budget": args.budget, "nsamples": args.nsamples,
           "capture_tokens": args.capture_tokens,
           "hot_grid": hot_grid, "s1_splits": s1_splits, "s2_splits": s2_splits,
           "git_commit": git_commit(),
           "summary": {"arms": summary, "d1": d1_summary},
           "layers": layers_out}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")

    print("=" * 72)
    print(f"D1 hot-set removal (mean r90 / max r90 / mean energy@{e_key}):")
    for hn, s in d1_summary.items():
        print(f"  hot_n={hn:>5}: {s['mean_r90']:7.1f} / {s['max_r90']:5d} / "
              f"{s[f'mean_energy_at_{e_key}']:.3f}")
    print(f"D2 matched-budget rel err (mean / mid-stack 4-17 / worst) — "
          f"SCREENING ONLY, PPL is the referee:")
    for a in arm_names:
        s = summary[a]
        print(f"  {a:<24} {s['mean']:.4f} / {s['mid_stack_mean']:.4f} / "
              f"{s['worst']:.4f} (L{s['worst_layer']})")
