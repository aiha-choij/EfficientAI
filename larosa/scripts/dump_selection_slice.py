# coding=utf-8
# Qualitative visualisation support for the coactivation-block-structure topic.
#
# Dumps the raw per-token Top-K survival pattern of the FFN intermediate
# activation for a contiguous slice of tokens, at several sparsity levels,
# so that the token-to-token variability of neuron selection can be shown
# directly (raster plots, pairwise overlap matrices) instead of only through
# aggregate statistics.
#
# For each sparsity level s and each sampled layer we store
#   sel[t, j]  (bool)  = neuron j survived Top-K for token t
# for the first `ntokens` tokens of one WikiText-2 sequence. The model runs
# in genuinely sparsified mode, so the selection at layer L reflects the
# sparsification applied in all earlier layers.

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
import torch


def main():
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from inference.modeling_llama_larosa import LlamaForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--sparsities', type=str, default="0.5,0.7,0.9")
    parser.add_argument('--layers', type=str, default="0,16,31")
    parser.add_argument('--ntokens', type=int, default=256)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--out', type=str, default="selection_slice.pt")
    parser.add_argument('--attn', type=str, default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    args = parser.parse_args()

    s_list = [float(x) for x in args.sparsities.split(',')]
    layers = [int(x) for x in args.layers.split(',')]

    config = transformers.AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = args.attn
    config.torch_dtype = 'bfloat16'
    config.sparse_mode = 'topk_intermediate'
    config.sparse_level = s_list[0]
    config.Q_path = None
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map='auto',
        config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(sample["text"] for sample in dataset)
    testenc = tokenizer(text, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    inputs = testenc[:, :args.ctx].to(device)

    store = {}
    hooks = []

    def make_hook(li, s):
        def hook(module, inp, out):
            sel = (inp[0][0] != 0)[:args.ntokens].cpu()      # [ntokens, d]
            store[(s, li)] = sel
        return hook

    for s in s_list:
        for layer in model.model.layers:
            layer.mlp.sparse_level_h2 = s
        hooks = [model.model.layers[li].mlp.down_proj.register_forward_hook(
            make_hook(li, s)) for li in layers]
        with torch.no_grad():
            model(inputs)
        for h in hooks:
            h.remove()
        for li in layers:
            sel = store[(s, li)]
            k = sel.sum(dim=1).float()
            print(f"s={s} layer={li}: K_eff={k.mean():.0f} "
                  f"tokens={sel.shape[0]} d={sel.shape[1]}")

    tok_ids = inputs[0, :args.ntokens].cpu()
    tok_str = [tokenizer.decode([int(t)]) for t in tok_ids]
    torch.save({'sparsities': s_list, 'layers': layers,
                'ntokens': args.ntokens, 'model': args.model_name,
                'token_ids': tok_ids, 'token_str': tok_str,
                'sel': {f"{s}_{li}": v for (s, li), v in store.items()}},
               args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
