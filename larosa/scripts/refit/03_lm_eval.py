# coding=utf-8
# lm-eval-harness zero-shot evaluation for L0/L1/L2 conditions. Wraps the
# ALREADY-loaded, already-configured model (mask installed via
# refit_mlp.set_condition, refit weights loaded via load_refit_weights if
# l1/l2) directly into lm_eval's HFLM, so the harness exercises the exact
# same masked/refit forward path as the PPL eval (02_eval_ppl.py) -- not a
# freshly re-loaded stock model, which is what `lm_eval --model hf
# --model_args pretrained=...` would give you.
#
# Usage:
#   python scripts/refit/03_lm_eval.py --model_name /raid/LLM/llama2-7b \
#       --mode l1 --s 0.9 --g 1 \
#       --weights_dir refit_out/llama2-7b/weights/l1_s0.9_g1_lam0.01 \
#       --limit 1000 --out_json refit_out/llama2-7b/results/harness_l1_s0.9_g1.json

import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(current_dir)

import torch
import transformers
from transformers import AutoTokenizer

from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, refit_mlp
import common

DEFAULT_TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag",
                 "winogrande", "boolq", "sciq", "lambada_openai"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--mode", type=str, required=True, choices=list(refit_mlp.MODES))
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--g", type=int, required=True)
    ap.add_argument("--weights_dir", type=str, default=None, help="required for l1/l2")
    ap.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--batch_size", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    if args.mode in ("l1", "l2") and not args.weights_dir:
        ap.error(f"--weights_dir is required for mode={args.mode}")

    # imported here: lm_eval pulls in a heavy dependency tree not needed by
    # the PPL-only scripts, and this keeps their import time light.
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = True  # harness generation/loglikelihood benefits from KV cache
    config._attn_implementation = oracle_mlp.best_attn_impl()
    config.torch_dtype = "bfloat16"
    config.sparse_mode = "refit"
    config.refit_mode = args.mode
    config.refit_s = args.s
    config.refit_g = args.g

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    refit_mlp.attach_col_norms(model)  # ALWAYS from the original weight, before any load
    if args.weights_dir:
        refit_mlp.load_refit_weights(model, args.weights_dir)
    refit_mlp.set_condition(model, args.mode, s=args.s, g=args.g)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    print(f"lm-eval: mode={args.mode} s={args.s} g={args.g} tasks={args.tasks} limit={args.limit}")
    with torch.no_grad():
        out = simple_evaluate(
            model=lm, tasks=args.tasks, limit=args.limit,
            random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
            log_samples=False, bootstrap_iters=1000,
        )

    per_layer = refit_mlp.achieved_sparsity_per_layer(model)
    mean_sp = sum(per_layer.values()) / max(len(per_layer), 1)

    def to_jsonable(d):
        if isinstance(d, dict):
            return {k: to_jsonable(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [to_jsonable(v) for v in d]
        if isinstance(d, (int, float, str, bool)) or d is None:
            return d
        return str(d)  # numpy scalars, etc.

    result = {
        "model_name": args.model_name,
        "mode": args.mode,
        "s": args.s,
        "g": args.g,
        "weights_dir": args.weights_dir,
        "tasks": args.tasks,
        "limit": args.limit,
        "seed": args.seed,
        "results": to_jsonable(out["results"]),
        "achieved_sparsity_mean": mean_sp,
        "git_commit": common.git_commit(parent_dir),
    }
    print("RESULT_JSON " + json.dumps({k: v for k, v in result.items() if k != "results"}))
    for task, metrics in result["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none"))
        print(f"  {task}: {metrics}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
