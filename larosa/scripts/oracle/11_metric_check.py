# coding=utf-8
# E-W0 metric validation: can a gradient-weighted error metric rank the KNOWN
# factor variants in their measured-PPL order where plain input-L2 provably
# cannot (the whitening inversion)?
#
# For each variant v with factors A,B (and slr runtime pieces where present),
# on captured calibration inputs x:
#   e_v(x)     = comp_v(x) - Mx           (compensation error vector, R^h)
#   plain[v]   = mean_layers E_x ||e_v(x)||_2
#   weighted[v]= mean_layers E_x ||w ⊙ e_v(x)||_2 ,  w = sqrt(E[(dL/dy)^2])
# then Spearman rank correlation of each metric against the provided measured
# PPLs, plus the explicit whitening-inversion checks. Gradient stats from
# 10_grad_calibrate.py; PPLs are passed on the CLI so this script never
# invents numbers.
#
# Usage:
#   python scripts/oracle/11_metric_check.py --model_name /raid/LLM/llama2-7b \
#       --stats_dir .../stats/c4 --grad_dir .../grad_stats/c4 \
#       --grad_dir_alt .../grad_stats/wt103 \
#       --factors plain512=.../factors/r512 wht512=.../factors/wht_r512 ... \
#       --ppl plain512=8.7638 wht512=9.7606 ... \
#       --out_json .../results/metric_check.json

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

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp

grad_calibrate = __import__("10_grad_calibrate")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=parent_dir).decode().strip()
    except Exception:
        return "unknown"


def spearman(a, b):
    """Spearman rank correlation without scipy (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va * vb > 0 else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--grad_dir", type=str, required=True)
    ap.add_argument("--grad_dir_alt", type=str, default=None,
                    help="second-corpus grad stats for the stability check")
    ap.add_argument("--factors", type=str, nargs="+", required=True,
                    help="name=dir pairs of factor variants")
    ap.add_argument("--ppl", type=str, nargs="+", required=True,
                    help="name=measured_ppl pairs (same names as --factors)")
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--capture_tokens", type=int, default=16384)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    variant_dirs = dict(kv.split("=", 1) for kv in args.factors)
    ppls = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.ppl)}
    assert set(ppls) == set(variant_dirs), "names in --ppl must match --factors"
    names = sorted(variant_dirs)

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

    # ---- capture x entering each MLP (same pattern as 09) -----------------
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
    for h in handles:
        h.remove()

    # variant metas (slr runtime knobs)
    metas = {}
    for name, vdir in variant_dirs.items():
        mp = os.path.join(vdir, "factors_meta.json")
        metas[name] = json.load(open(mp)) if os.path.exists(mp) else {}

    # ---- per-layer errors --------------------------------------------------
    plain_sum = {n: 0.0 for n in names}
    weighted_sum = {n: 0.0 for n in names}
    stability = {}
    nlayers = 0
    with torch.no_grad():
        for li, mlp in oracle_mlp.iter_mlps(model):
            dev = mlp.down_proj.weight.device
            x = torch.cat(buffers[li]).float().to(dev)
            buffers[li] = None
            M = oracle_mlp.compute_M(mlp)
            w = grad_calibrate.load_w(args.grad_dir, li, device=dev)
            if args.grad_dir_alt:
                w_alt = grad_calibrate.load_w(args.grad_dir_alt, li, device=dev)
                lw, lwa = w.log(), w_alt.log()
                stability[str(li)] = float(torch.corrcoef(
                    torch.stack([lw, lwa]))[0, 1].item())
            for name in names:
                d = torch.load(os.path.join(variant_dirs[name], f"layer_{li}.pt"),
                               map_location=dev)
                A, B = d["A"].float(), d["B"].float()
                R = M - B @ A
                meta = metas[name]
                if meta.get("comp_mode", "lr") == "slr_input":
                    k = meta["sparse_k"]
                    m_x = oracle_mlp.top_count_mask(x.abs(), k).float()
                    e = ((1.0 - m_x) * x) @ R.T   # sign-flipped err; norms only
                else:
                    e = x @ R.T
                plain_sum[name] += float(e.norm(dim=-1).mean().item())
                weighted_sum[name] += float((e * w).norm(dim=-1).mean().item())
                del e, R, A, B
            del x, M
            torch.cuda.empty_cache()
            nlayers += 1
            print(f"layer {li} done", flush=True)

    plain = {n: plain_sum[n] / nlayers for n in names}
    weighted = {n: weighted_sum[n] / nlayers for n in names}
    ppl_v = [ppls[n] for n in names]
    result = {
        "names": names,
        "measured_ppl": ppls,
        "plain_err": plain,
        "weighted_err": weighted,
        "spearman_plain_vs_ppl": spearman([plain[n] for n in names], ppl_v),
        "spearman_weighted_vs_ppl": spearman([weighted[n] for n in names], ppl_v),
        "stability_log_w_corr_per_layer": stability,
        "nsamples": args.nsamples, "capture_tokens": args.capture_tokens,
        "grad_dir": args.grad_dir, "git_commit": git_commit(),
    }

    # explicit inversion checks: PPL says whitened is WORSE than plain at the
    # same rank; a loss-aligned metric must agree (plain L2 is known to invert)
    checks = {}
    for a, b in (("wht512", "plain512"), ("wht1024", "plain1024")):
        if a in names and b in names:
            checks[f"{a}>{b}"] = {
                "ppl_says": ppls[a] > ppls[b],
                "plain_metric_says": plain[a] > plain[b],
                "weighted_metric_says": weighted[a] > weighted[b],
            }
    result["inversion_checks"] = checks

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out_json}")

    print("=" * 72)
    print(f"{'variant':<14} {'PPL@0.9':>8} {'plain_err':>10} {'weighted_err':>13}")
    for n in sorted(names, key=lambda n: ppls[n]):
        print(f"{n:<14} {ppls[n]:>8.4f} {plain[n]:>10.4f} {weighted[n]:>13.6f}")
    print(f"Spearman vs PPL: plain {result['spearman_plain_vs_ppl']:+.3f}  "
          f"weighted {result['spearman_weighted_vs_ppl']:+.3f}")
    for k, v in checks.items():
        print(f"inversion {k}: ppl {v['ppl_says']} | plain {v['plain_metric_says']} "
              f"| weighted {v['weighted_metric_says']}")
    if stability:
        vals = list(stability.values())
        print(f"w stability (log-corr c4 vs alt): mean {sum(vals)/len(vals):.3f} "
              f"min {min(vals):.3f}")
