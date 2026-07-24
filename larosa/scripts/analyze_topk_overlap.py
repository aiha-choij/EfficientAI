# coding=utf-8
# Cross-token neuron-selection overlap analysis for topk_intermediate mode.
#
# Runs the actually-sparsified model (selection at layer L reflects upstream
# sparsification) at each sparsity level and, via a forward hook on down_proj,
# records which intermediate neurons survive per token. Reports, per layer:
#   - containment overlap |S_t ∩ S_t'| / K_eff for token pairs at several
#     distances and for random pairs, vs the chance baseline K_eff/d
#   - neuron selection frequency: always-on (>0.9), rare (<0.01), never (=0),
#     and the Gini coefficient of the frequency distribution
#   - per-sequence union coverage |∪_t S_t| / d
# and, across sparsity levels, the overlap of top-frequency neuron sets
# (do the survivors at high s come from the same "important" pool?).
#
# Results are printed and saved as a torch .pt for later plotting.

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
import torch


DISTANCES = [1, 4, 16, 64, 256]


class LayerStats:
    def __init__(self, d, device):
        self.d = d
        self.freq = torch.zeros(d, dtype=torch.float64, device=device)
        self.tokens = 0
        self.k_sum = 0.0
        self.inter = {dist: 0.0 for dist in DISTANCES}
        self.inter_pairs = {dist: 0 for dist in DISTANCES}
        self.rand_inter = 0.0
        self.rand_pairs = 0
        self.union_frac_sum = 0.0
        self.seqs = 0

    def update(self, sel):
        # sel: [B, T, d] bool — True where the neuron survived Top-K
        B, T, d = sel.shape
        self.tokens += B * T
        self.k_sum += sel.sum().item()
        self.freq += sel.sum(dim=(0, 1)).double()
        for dist in DISTANCES:
            if T > dist:
                inter = (sel[:, dist:] & sel[:, :-dist]).sum().item()
                self.inter[dist] += inter
                self.inter_pairs[dist] += B * (T - dist)
        perm = torch.randperm(T, device=sel.device)
        self.rand_inter += (sel & sel[:, perm]).sum().item()
        self.rand_pairs += B * T
        self.union_frac_sum += (sel.any(dim=1).sum(dim=-1).double() / d).sum().item()
        self.seqs += B

    def summary(self):
        k_eff = self.k_sum / self.tokens
        freq = (self.freq / self.tokens).cpu()
        sorted_f, _ = torch.sort(freq)
        n = self.d
        idx = torch.arange(1, n + 1, dtype=torch.float64)
        gini = ((2 * idx - n - 1) * sorted_f).sum().item() / (n * sorted_f.sum().item() + 1e-12)
        out = {
            'k_eff': k_eff,
            'chance': k_eff / self.d,
            'freq': freq,
            'always_on': (freq > 0.9).float().mean().item(),
            'rare': (freq < 0.01).float().mean().item(),
            'never': (freq == 0).float().mean().item(),
            'gini': gini,
            'union_frac': self.union_frac_sum / self.seqs,
            'rand_overlap': self.rand_inter / (self.rand_pairs * k_eff),
        }
        for dist in DISTANCES:
            out[f'overlap_d{dist}'] = (
                self.inter[dist] / (self.inter_pairs[dist] * k_eff)
                if self.inter_pairs[dist] else float('nan'))
        return out


def run_sparsity(model, testenc, seqlen, nsamples, device):
    stats = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            i_masked = inp[0]
            sel = (i_masked != 0)
            if layer_idx not in stats:
                stats[layer_idx] = LayerStats(sel.shape[-1], sel.device)
            stats[layer_idx].update(sel)
        return hook

    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        for i in range(nsamples):
            if i % 8 == 0:
                print(f"  sample {i}/{nsamples}")
            inputs = testenc[:, i * seqlen:(i + 1) * seqlen].to(device)
            model(inputs)

    for h in hooks:
        h.remove()
    return {li: st.summary() for li, st in stats.items()}


def print_summary(s, layer_summaries):
    layers = sorted(layer_summaries)
    def mean(key):
        return sum(layer_summaries[li][key] for li in layers) / len(layers)
    print(f"\n===== sparsity={s} (mean over {len(layers)} layers) =====")
    print(f"K_eff/token: {mean('k_eff'):.0f}  chance overlap K/d: {mean('chance'):.3f}")
    for dist in DISTANCES:
        m = mean(f'overlap_d{dist}')
        print(f"overlap @dist={dist:>3}: {m:.3f}  ({m / mean('chance'):.2f}x chance)")
    m = mean('rand_overlap')
    print(f"overlap random  : {m:.3f}  ({m / mean('chance'):.2f}x chance)")
    print(f"union frac / 2048-token seq: {mean('union_frac'):.3f}")
    print(f"always-on(>0.9): {mean('always_on')*100:.2f}%  rare(<0.01): "
          f"{mean('rare')*100:.2f}%  never: {mean('never')*100:.2f}%  gini: {mean('gini'):.3f}")
    for li in (layers[0], layers[len(layers)//2], layers[-1]):
        ls = layer_summaries[li]
        print(f"  layer {li:>2}: d1={ls['overlap_d1']:.3f} rand={ls['rand_overlap']:.3f} "
              f"chance={ls['chance']:.3f} union={ls['union_frac']:.3f} "
              f"always-on={ls['always_on']*100:.2f}% gini={ls['gini']:.3f}")


def cross_s_survivors(all_results, top_frac=0.1):
    print(f"\n===== cross-sparsity survivor analysis (top {top_frac:.0%} most-frequent neurons) =====")
    s_values = sorted(all_results)
    layers = sorted(all_results[s_values[0]])
    for a in range(len(s_values)):
        for b in range(a + 1, len(s_values)):
            sa, sb = s_values[a], s_values[b]
            ovl, corr = 0.0, 0.0
            for li in layers:
                fa = all_results[sa][li]['freq']
                fb = all_results[sb][li]['freq']
                k = int(top_frac * fa.numel())
                ta = set(torch.topk(fa, k).indices.tolist())
                tb = set(torch.topk(fb, k).indices.tolist())
                ovl += len(ta & tb) / k
                corr += torch.corrcoef(torch.stack([fa.float(), fb.float()]))[0, 1].item()
            n = len(layers)
            print(f"s={sa} vs s={sb}: top-set overlap {ovl/n:.3f} "
                  f"(chance {top_frac:.3f}), freq corr {corr/n:.3f}")


if __name__ == "__main__":
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from inference.modeling_llama_larosa import LlamaForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--sparsities', type=str, default="0.5,0.7,0.9")
    parser.add_argument('--nsamples', type=int, default=32)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--out', type=str, default="topk_overlap_analysis.pt")
    args = parser.parse_args()

    s_list = [float(x) for x in args.sparsities.split(',')]

    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = 'bfloat16'
    config.sparse_mode = 'topk_intermediate'
    config.sparse_level = s_list[0]
    config.Q_path = None

    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map='auto', config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(sample["text"] for sample in dataset)
    testenc = tokenizer(text, return_tensors="pt").input_ids
    nsamples = min(args.nsamples, testenc.numel() // args.ctx)
    device = next(model.parameters()).device

    all_results = {}
    for s in s_list:
        print(f"\n########## sparsity {s} ##########")
        for layer in model.model.layers:
            layer.mlp.sparse_level_h2 = s
        all_results[s] = run_sparsity(model, testenc, args.ctx, nsamples, device)
        print_summary(s, all_results[s])

    cross_s_survivors(all_results, top_frac=0.1)

    torch.save({'sparsities': s_list, 'nsamples': nsamples, 'ctx': args.ctx,
                'model': args.model_name, 'results': all_results}, args.out)
    print(f"\nsaved: {args.out}")
