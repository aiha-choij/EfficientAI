# coding=utf-8
# GC-refit E0 -- gate diagnostic driver (see
# requests/active/20260804-171529-gc-refit-e0/prompt.md and
# inference/gc_refit.py's module docstring for the full design).
#
# Part 1 (--part part1): does the optimal anchor-ridge refit of down_proj
# actually depend on which neuron-blocks a token-block's P3' mask cut?
# Collects mask patterns (mu) on 448 fit sequences, balanced-2-clusters
# them per layer (Hamming k-means), solves marginal + per-cluster refit
# (lambda in {0.01,0.1}), and cross-evaluates all three solutions on 64
# held-out sequences (sequence-level split, no leakage; held-out blocks
# get nearest-centroid cluster labels from the FIT centroids, never
# re-clustered). Reports per-layer M[c,c'] MSE matrices and the
# heterogeneity gain = M[cross]/M[within] - 1.
#
# Part 2 (--part part2): oracle block-average compensation wikitext-2 PPL
# (exact per-block average of the masked-out contribution, not a per-token
# estimate) -- same (g,B,sparsity,score) setting as Part 1 and as
# block-sparse-compensation's P3' (C7a/C7/C8a/C8) anchors.
#
# Usage:
#   python scripts/gcrefit/01_run_e0.py --part part1 \
#       --model_name /raid/LLM/llama2-7b --g 16 --B 64 --sparsity 0.9 \
#       --lambdas 0.01 0.1 \
#       --partitions ~/workspace/analysis/llama2_p3_partitions_s09.pt \
#       --stats_dir ~/workspace/oracle/llama2-7b/stats/wikitext103 \
#       --out_dir ~/workspace/gcrefit/llama2-7b
#
#   python scripts/gcrefit/01_run_e0.py --part part2 \
#       --model_name /raid/LLM/llama2-7b --g 16 --B 64 --sparsity 0.9 \
#       --partitions ~/workspace/analysis/llama2_p3_partitions_s09.pt \
#       --stats_dir ~/workspace/oracle/llama2-7b/stats/wikitext103 \
#       --out_dir ~/workspace/gcrefit/llama2-7b

import argparse
import json
import os
import subprocess
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "scripts", "refit"))  # for common.py

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, block_comp_mlp, refit_mlp, gc_refit
from utils.eval_ppl import eval_ppl_wikitext_with_inference_sparsity
import common


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=parent_dir).decode().strip()
    except Exception:
        return "unknown"


def layers_per_pass_gc(num_layers, d, h, mult, budget_gb):
    """mult=1 for marginal (G,C) accumulation, mult=2 for cluster-split
    (G0,C0,G1,C1). Same sizing discipline as
    scripts/refit/01_build_l1.py's layers_per_pass."""
    bytes_per_layer = 4 * (d * d + h * d) * mult
    lpp = max(1, int(budget_gb * (1024 ** 3) // bytes_per_layer))
    return min(lpp, num_layers)


def layers_per_pass_eval(num_layers, d, h, n_candidates, budget_gb):
    """Eval only needs candidate weight matrices [h,d] resident, not the
    much larger [d,d] G accumulators."""
    bytes_per_layer = 4 * h * d * n_candidates
    lpp = max(1, int(budget_gb * (1024 ** 3) // bytes_per_layer))
    return min(lpp, num_layers)


def load_model(args):
    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "gc_refit"
    config.blk_g = args.g

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # same precondition as block_comp/oracle C3-C5: dense calibration stats
    # (g_bar, col_norm) must already exist -- this script does not
    # recalibrate them, it reuses the existing oracle stats artifact.
    oracle_mlp.attach_col_norms(model)
    oracle_mlp.load_stats(model, args.stats_dir)

    device = next(model.parameters()).device
    d = model.config.intermediate_size
    K = round((1.0 - args.sparsity) * d)
    m_keep = max(1, round(K / args.B))
    gc_refit.attach_partitions(model, args.partitions, args.B, m_keep, device)
    return model, tokenizer, m_keep, K, device


def load_or_build_tokens(args, tokenizer):
    tok_path = args.calib_tokens or os.path.join(args.out_dir, "calib_tokens.pt")
    if os.path.exists(tok_path):
        tokens = refit_mlp.load_calib_tokens(tok_path, args.nsamples, args.seqlen)
        print(f"reusing saved calibration tokens: {tok_path}")
    else:
        tokens = common.build_calib_tokens(args.dataset, tokenizer, args.nsamples, args.seqlen, args.seed)
        refit_mlp.save_calib_tokens(tokens, tok_path)
        print(f"saved calibration tokens: {tok_path}")
    return tokens, tok_path


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def run_part1(args):
    model, tokenizer, m_keep, K, device = load_model(args)
    tokens, tok_path = load_or_build_tokens(args, tokenizer)
    assert args.fit_n + args.held_n <= args.nsamples, \
        f"fit_n({args.fit_n}) + held_n({args.held_n}) > nsamples({args.nsamples})"
    fit_tokens = tokens[:args.fit_n]
    held_tokens = tokens[args.fit_n:args.fit_n + args.held_n]  # sequence-level split, no overlap
    run_part1_core(model, fit_tokens, held_tokens, device, args, m_keep, K, tok_path)


def run_part1_core(model, fit_tokens, held_tokens, device, args, m_keep, K, tok_path):
    """Split out from run_part1 so the exact same orchestration logic
    (chunking, clustering, held-out cross-eval, gain computation) can be
    exercised by a CPU integration smoke test (test_gcrefit_units.py)
    against a tiny synthetic model, without going through
    AutoConfig.from_pretrained / a real 7B checkout."""
    num_layers = model.config.num_hidden_layers
    d, h = model.config.intermediate_size, model.config.hidden_size
    nb = args.seqlen // args.g
    all_layers = list(range(num_layers))

    print(f"Part1: g={args.g} B={args.B} sparsity={args.sparsity} K={K} m_keep={m_keep} "
          f"fit_n={args.fit_n} held_n={args.held_n} lambdas={args.lambdas}")

    # ---- mu collection pass (fit set, all layers, no (G,C) accumulators) ----
    t0 = time.time()
    gc_refit.enable_collect_a(model, layers=[])
    mu_by_layer = {li: [] for li in all_layers}
    with torch.no_grad():
        for si in range(args.fit_n):
            model(fit_tokens[si:si + 1].to(device))
            for layer_idx, mlp in oracle_mlp.iter_mlps(model):
                mu_by_layer[layer_idx].append(mlp.gc_last_mu)
            if si % 64 == 0:
                print(f"mu-collect (fit) {si}/{args.fit_n}", flush=True)
    mu_by_layer = {li: torch.cat(v, dim=0).reshape(-1, mu_by_layer[li][0].shape[-1]) for li, v in mu_by_layer.items()}
    print(f"mu-collect (fit) done in {time.time() - t0:.1f}s, "
          f"{mu_by_layer[0].shape[0]} blocks/layer (expect {args.fit_n * nb})")

    # ---- balanced 2-way clustering per layer ----
    labels_by_layer, centroids_by_layer, balance_report = {}, {}, {}
    for li in all_layers:
        labels, centroids = gc_refit.balanced_hamming_kmeans(mu_by_layer[li], seed=args.seed)
        labels_by_layer[li] = labels
        centroids_by_layer[li] = centroids
        n0 = int((labels == 0).sum())
        n1 = len(labels) - n0
        hd = (centroids[0] - centroids[1]).abs().sum().item()
        balance_report[li] = {"n0": n0, "n1": n1, "centroid_hamming_dist": hd}
    print(f"clustering done: median centroid Hamming dist = "
          f"{torch.tensor([v['centroid_hamming_dist'] for v in balance_report.values()]).median().item():.2f} "
          f"(out of B={args.B})")

    # ---- marginal (G,C,Y2,n) accumulation, chunked over layers ----
    lpp_a = layers_per_pass_gc(num_layers, d, h, mult=1, budget_gb=args.layers_budget_gb)
    chunks_a = [all_layers[i:i + lpp_a] for i in range(0, num_layers, lpp_a)]
    print(f"collect_a (marginal): layers_per_pass={lpp_a} -> {len(chunks_a)} sweep(s)")
    stats_marg = {}
    for ci, chunk in enumerate(chunks_a):
        gc_refit.enable_collect_a(model, layers=chunk)
        with torch.no_grad():
            for si in range(args.fit_n):
                model(fit_tokens[si:si + 1].to(device))
                if si % 128 == 0:
                    print(f"collect_a chunk {ci + 1}/{len(chunks_a)} (layers {chunk[0]}-{chunk[-1]}) "
                          f"sample {si}/{args.fit_n}", flush=True)
        stats_marg.update(gc_refit.finalize_collect_a(model))

    # ---- cluster-split (G,C,Y2,n) accumulation, chunked (2x memory) ----
    lpp_b = layers_per_pass_gc(num_layers, d, h, mult=2, budget_gb=args.layers_budget_gb)
    chunks_b = [all_layers[i:i + lpp_b] for i in range(0, num_layers, lpp_b)]
    print(f"collect_b (cluster split): layers_per_pass={lpp_b} -> {len(chunks_b)} sweep(s)")
    stats_cluster = {}
    for ci, chunk in enumerate(chunks_b):
        gc_refit.enable_collect_b(model, labels_by_layer, blocks_per_seq=nb, layers=chunk)
        with torch.no_grad():
            for si in range(args.fit_n):
                gc_refit.set_seq_idx(model, si)
                model(fit_tokens[si:si + 1].to(device))
                if si % 128 == 0:
                    print(f"collect_b chunk {ci + 1}/{len(chunks_b)} (layers {chunk[0]}-{chunk[-1]}) "
                          f"sample {si}/{args.fit_n}", flush=True)
        stats_cluster.update(gc_refit.finalize_collect_b(model))

    w_anchors = {li: mlp.down_proj.weight.detach().float().cpu() for li, mlp in oracle_mlp.iter_mlps(model)}

    # ---- solve marginal + per-cluster, each lambda ----
    solved = {}
    for lam in args.lambdas:
        solved[lam] = {}
        for li in all_layers:
            Wm = refit_mlp.solve_refit(stats_marg[li]["G"], stats_marg[li]["C"], w_anchors[li], lam=lam)
            W0 = refit_mlp.solve_refit(stats_cluster[li][0]["G"], stats_cluster[li][0]["C"], w_anchors[li], lam=lam)
            W1 = refit_mlp.solve_refit(stats_cluster[li][1]["G"], stats_cluster[li][1]["C"], w_anchors[li], lam=lam)
            solved[lam][li] = {"marg": Wm, 0: W0, 1: W1}
    print("solve_refit done for all layers/lambdas")

    # ---- sanity (a): marginal refit fit-set MSE must never exceed masking-only (anchor) fit-set MSE ----
    sanity = {}
    for lam in args.lambdas:
        viol = []
        for li in all_layers:
            st = stats_marg[li]
            mse_anchor = gc_refit.closed_form_mse(w_anchors[li].to(st["G"].device), st["G"], st["C"], st["Y2"], st["n"])
            mse_refit = gc_refit.closed_form_mse(solved[lam][li]["marg"], st["G"], st["C"], st["Y2"], st["n"])
            if mse_refit > mse_anchor + 1e-6:
                viol.append({"layer": li, "mse_anchor": mse_anchor, "mse_refit": mse_refit})
        sanity[lam] = {"violations": viol, "ok": len(viol) == 0}
        print(f"sanity (marginal refit MSE <= masking-only MSE), lam={lam}: "
              f"{'OK' if not viol else 'VIOLATION ' + str(viol)}")

    # ---- held-out: mu collection + nearest-centroid cluster assignment (never re-clustered) ----
    gc_refit.enable_collect_a(model, layers=[])
    held_mu_by_layer = {li: [] for li in all_layers}
    with torch.no_grad():
        for si in range(args.held_n):
            model(held_tokens[si:si + 1].to(device))
            for layer_idx, mlp in oracle_mlp.iter_mlps(model):
                held_mu_by_layer[layer_idx].append(mlp.gc_last_mu)
            if si % 32 == 0:
                print(f"mu-collect (held-out) {si}/{args.held_n}", flush=True)
    held_mu_by_layer = {li: torch.cat(v, dim=0).reshape(-1, v[0].shape[-1]) for li, v in held_mu_by_layer.items()}
    held_labels_by_layer = {li: gc_refit.assign_nearest_centroid(held_mu_by_layer[li], centroids_by_layer[li])
                            for li in all_layers}

    # ---- held-out cross-eval: SSE per candidate (marg/c0/c1 x lambda), split by TRUE held-out cluster ----
    n_cand = len(args.lambdas) * 3
    lpp_eval = layers_per_pass_eval(num_layers, d, h, n_cand, budget_gb=args.layers_budget_gb)
    chunks_eval = [all_layers[i:i + lpp_eval] for i in range(0, num_layers, lpp_eval)]
    print(f"eval (held-out cross): layers_per_pass={lpp_eval} -> {len(chunks_eval)} sweep(s)")
    eval_results = {}
    for ci, chunk in enumerate(chunks_eval):
        candidates_by_layer = {}
        for li in chunk:
            cands = {}
            for lam in args.lambdas:
                cands[f"lam{lam}_marg"] = solved[lam][li]["marg"]
                cands[f"lam{lam}_c0"] = solved[lam][li][0]
                cands[f"lam{lam}_c1"] = solved[lam][li][1]
            candidates_by_layer[li] = cands
        gc_refit.enable_eval(model, candidates_by_layer, held_labels_by_layer, blocks_per_seq=nb, layers=chunk)
        with torch.no_grad():
            for si in range(args.held_n):
                gc_refit.set_seq_idx(model, si)
                model(held_tokens[si:si + 1].to(device))
                if si % 32 == 0:
                    print(f"eval chunk {ci + 1}/{len(chunks_eval)} (layers {chunk[0]}-{chunk[-1]}) "
                          f"sample {si}/{args.held_n}", flush=True)
        eval_results.update(gc_refit.finalize_eval(model))

    # ---- heterogeneity gain + within-vs-marginal improvement, per layer, per lambda ----
    report = {}
    for lam in args.lambdas:
        per_layer = {}
        gains = []
        for li in all_layers:
            mse = eval_results[li]["mse"]
            M00, M01 = mse[f"lam{lam}_c0"][0], mse[f"lam{lam}_c0"][1]
            M10, M11 = mse[f"lam{lam}_c1"][0], mse[f"lam{lam}_c1"][1]
            Mmarg0, Mmarg1 = mse[f"lam{lam}_marg"][0], mse[f"lam{lam}_marg"][1]
            within = (M00 + M11) / 2
            cross = (M01 + M10) / 2
            gain = (cross / within - 1) if within > 0 else float("nan")
            imp0 = (Mmarg0 - M00) / Mmarg0 if Mmarg0 > 0 else float("nan")
            imp1 = (Mmarg1 - M11) / Mmarg1 if Mmarg1 > 0 else float("nan")
            per_layer[li] = {
                "M00": M00, "M01": M01, "M10": M10, "M11": M11, "Mmarg0": Mmarg0, "Mmarg1": Mmarg1,
                "within": within, "cross": cross, "heterogeneity_gain": gain,
                "within_vs_marginal_improve_c0": imp0, "within_vs_marginal_improve_c1": imp1,
                "held_out_count": eval_results[li]["count"],
            }
            gains.append(gain)
        gains_t = torch.tensor(gains)
        median_gain = gains_t.median().item()
        verdict = "positive" if median_gain >= 0.15 else ("negative" if median_gain < 0.05 else "gray-zone")
        report[lam] = {
            "per_layer": per_layer,
            "median_heterogeneity_gain": median_gain,
            "mean_heterogeneity_gain": gains_t.mean().item(),
            "verdict": verdict,
        }
        print(f"lam={lam}: median heterogeneity gain = {median_gain:.4f} -> {verdict}")

    out = {
        "meta": {
            "model_name": args.model_name, "seed": args.seed, "dataset": args.dataset,
            "nsamples": args.nsamples, "seqlen": args.seqlen, "fit_n": args.fit_n, "held_n": args.held_n,
            "calib_tokens": tok_path, "g": args.g, "B": args.B, "sparsity": args.sparsity,
            "K": K, "m_keep": m_keep, "lambdas": args.lambdas,
            "score": "residual |u*(g-g_bar)|*col_norm (block_comp_mlp._resid_score)",
            "partitions": args.partitions, "stats_dir": args.stats_dir, "git_commit": git_commit(),
        },
        "balance": balance_report,
        "sanity": {str(lam): v for lam, v in sanity.items()},
        "part1": {str(lam): v for lam, v in report.items()},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "part1_e0_report.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    torch.save(
        {"labels_by_layer": labels_by_layer, "centroids_by_layer": centroids_by_layer,
         "held_labels_by_layer": held_labels_by_layer, "balance_report": balance_report},
        os.path.join(args.out_dir, "part1_clusters.pt"))
    print("RESULT_JSON " + json.dumps({
        "part": "part1",
        "median_heterogeneity_gain_by_lambda": {str(l): report[l]["median_heterogeneity_gain"] for l in args.lambdas},
        "verdict_by_lambda": {str(l): report[l]["verdict"] for l in args.lambdas},
        "sanity_ok": {str(l): sanity[l]["ok"] for l in args.lambdas},
        "git_commit": git_commit(),
    }))
    print(f"wrote {out_json}")
    return out


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

def run_part2(args):
    model, tokenizer, m_keep, K, device = load_model(args)
    gc_refit.enable_oracle_avg(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print(f"Part2 (oracle block-average): g={args.g} B={args.B} sparsity={args.sparsity} "
          f"K={K} m={m_keep} (budget {m_keep * args.B}/{model.config.intermediate_size}="
          f"{m_keep * args.B / model.config.intermediate_size:.4f})")
    with torch.no_grad():
        ppl = eval_ppl_wikitext_with_inference_sparsity(model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"oracle block-average PPL: {ppl}")

    per_layer = block_comp_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)
    print(f"achieved s_block mean={mean_sp:.4f}")

    result = {
        "model_name": args.model_name, "condition": "oracle_blockavg", "g": args.g, "B": args.B,
        "sparsity": args.sparsity, "K": K, "m_keep": m_keep, "partitions": args.partitions,
        "stats_dir": args.stats_dir, "ppl": ppl, "achieved_s_block_mean": mean_sp,
        "achieved_s_block_per_layer": per_layer, "git_commit": git_commit(),
    }
    print("RESULT_JSON " + json.dumps({k: v for k, v in result.items() if k != "achieved_s_block_per_layer"}))
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "part2_oracle_blockavg.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["part1", "part2"], required=True)
    ap.add_argument("--model_name", type=str, default="/raid/LLM/llama2-7b")
    ap.add_argument("--dataset", type=str, default="c4", choices=["c4", "wikitext103"])
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--fit_n", type=int, default=448)
    ap.add_argument("--held_n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--g", type=int, default=16)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1])
    ap.add_argument("--calib_tokens", type=str, default=None)
    ap.add_argument("--partitions", type=str, required=True)
    ap.add_argument("--stats_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--layers_budget_gb", type=float, default=12.0,
                    help="target GPU memory ceiling for (G,C)/candidate accumulators per calibration sweep "
                         "-- more layers resident at once means fewer repeated calibration passes")
    args = ap.parse_args()
    args.partitions = os.path.expanduser(args.partitions)
    args.stats_dir = os.path.expanduser(args.stats_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    if args.part == "part1":
        run_part1(args)
    else:
        run_part2(args)
