# coding=utf-8
# Co-activation statistics collection for topk_intermediate mode (P1 of the
# coactivation-block-structure topic).
#
# Runs the actually-sparsified model and, via a forward hook on down_proj,
# accumulates for each sampled layer:
#   f_j       selection frequency of neuron j
#   A_jj'     same-token co-activation counts:  sum_t 1[j in S_t] 1[j' in S_t]
#   A^g_jj'   window-g co-activation counts:    sum_{|t-t'|<g} 1[j in S_t] 1[j' in S_t']
#             (within-sequence windows; includes the t'=t self pair)
#   union inflation |U_{t in group} S_t| / K_eff for contiguous token groups
#             (the P0 "group tax" curve, free in the same pass)
# Counts are accumulated as fp32 GPU matmuls (exact: all counts < 2^24) and
# saved raw with their normalization constants; PMI / Jaccard are derived
# downstream (P2) as A/tokens, A^g/window_pairs over f = freq/tokens.
#
# d x d fp32 is ~484 MB per matrix: 5 layers x (1 + len(windows)) matrices
# must fit in GPU memory alongside the model.

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
import torch


class CoactStats:
    def __init__(self, d, windows, group_sizes, device):
        self.d = d
        self.windows = windows
        self.group_sizes = group_sizes
        self.freq = torch.zeros(d, dtype=torch.float32, device=device)
        self.A = torch.zeros(d, d, dtype=torch.float32, device=device)
        self.Aw = {g: torch.zeros(d, d, dtype=torch.float32, device=device)
                   for g in windows}
        self.w_pairs = {g: 0 for g in windows}
        self.tokens = 0
        self.k_sum = 0.0
        self.union_sum = {g: 0.0 for g in group_sizes}
        self.union_groups = {g: 0 for g in group_sizes}

    def update(self, sel):
        # sel: [B, T, d] bool — True where the neuron survived Top-K
        B, T, d = sel.shape
        self.tokens += B * T
        self.k_sum += sel.sum().item()
        for b in range(B):
            s = sel[b].float()                      # [T, d]
            self.freq += s.sum(dim=0)
            self.A += s.T @ s
            cs = torch.zeros(T + 1, d, device=s.device)
            torch.cumsum(s, dim=0, out=cs[1:])
            for g in self.windows:
                lo = torch.clamp(torch.arange(T, device=s.device) - (g - 1), min=0)
                hi = torch.clamp(torch.arange(T, device=s.device) + g, max=T)
                W = cs[hi] - cs[lo]                 # [T, d] window sums
                self.Aw[g] += s.T @ W
                self.w_pairs[g] += int((hi - lo).sum().item())
            for g in self.group_sizes:
                ng = T // g
                grp = sel[b, :ng * g].reshape(ng, g, d)
                self.union_sum[g] += grp.any(dim=1).sum().item()
                self.union_groups[g] += ng

    def summary(self):
        k_eff = self.k_sum / self.tokens
        out = {
            'freq': self.freq.cpu(),                 # counts; f_j = freq/tokens
            'A': self.A.cpu(),                       # counts; pairs = tokens
            'Aw': {g: m.cpu() for g, m in self.Aw.items()},
            'w_pairs': dict(self.w_pairs),
            'tokens': self.tokens,
            'k_eff': k_eff,
            'chance': k_eff / self.d,
            'union_inflation': {                     # E[|union|] / K_eff
                g: self.union_sum[g] / (self.union_groups[g] * k_eff)
                for g in self.group_sizes},
        }
        return out


def main():
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from inference.modeling_llama_larosa import LlamaForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--layers', type=str, default="0,8,16,24,31")
    parser.add_argument('--windows', type=str, default="16,64")
    parser.add_argument('--group_sizes', type=str, default="16,32,64")
    parser.add_argument('--nsamples', type=int, default=32)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--out', type=str, default="coactivation_stats.pt")
    parser.add_argument('--attn', type=str, default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager'],
                        help='attention backend (use sdpa on hosts without flash-attn)')
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    windows = [int(x) for x in args.windows.split(',')]
    group_sizes = [int(x) for x in args.group_sizes.split(',')]

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = args.attn
    config.torch_dtype = 'bfloat16'
    config.sparse_mode = 'topk_intermediate'
    config.sparse_level = args.sparsity
    config.Q_path = None

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map='auto', config=config)
    model.eval()
    for layer in model.model.layers:
        layer.mlp.sparse_level_h2 = args.sparsity
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(sample["text"] for sample in dataset)
    testenc = tokenizer(text, return_tensors="pt").input_ids
    nsamples = min(args.nsamples, testenc.numel() // args.ctx)
    device = next(model.parameters()).device

    stats = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            sel = (inp[0] != 0)
            if layer_idx not in stats:
                stats[layer_idx] = CoactStats(sel.shape[-1], windows, group_sizes, sel.device)
            stats[layer_idx].update(sel)
        return hook

    for li in layers:
        hooks.append(model.model.layers[li].mlp.down_proj.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        for i in range(nsamples):
            if i % 8 == 0:
                print(f"sample {i}/{nsamples}")
            inputs = testenc[:, i * args.ctx:(i + 1) * args.ctx].to(device)
            model(inputs)

    for h in hooks:
        h.remove()

    results = {li: st.summary() for li, st in stats.items()}

    # sanity print: k_eff, chance, mean same-token co-activation lift, union tax
    for li in layers:
        r = results[li]
        f = r['freq'] / r['tokens']
        A_rate = r['A'] / r['tokens']
        # lift of same-token co-activation over independence, off-diagonal mean
        indep = torch.outer(f, f)
        mask = ~torch.eye(r['freq'].numel(), dtype=torch.bool)
        lift = (A_rate[mask].double().mean() / indep[mask].double().mean()).item()
        uni = "  ".join(f"g={g}: {v:.2f}x" for g, v in r['union_inflation'].items())
        print(f"layer {li:>2}: K_eff={r['k_eff']:.0f} chance={r['chance']:.3f} "
              f"mean off-diag co-act lift={lift:.3f}  union/K [{uni}]")

    torch.save({'sparsity': args.sparsity, 'nsamples': nsamples, 'ctx': args.ctx,
                'model': args.model_name, 'layers': layers, 'windows': windows,
                'group_sizes': group_sizes, 'results': results}, args.out)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
