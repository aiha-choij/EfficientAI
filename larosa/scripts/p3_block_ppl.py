# coding=utf-8
# P3 of the coactivation-block-structure topic: block-mask oracle PPL.
#
# For each group of g contiguous tokens, blocks are scored with the
# gauge-fixed weight-aware score  s_b = sum_{t in group} sum_{j in b}
# ||W_d[:,j]||_2 * |i_{t,j}|,  the top-m blocks (m = round(K/B)) are kept,
# and every token in the group is masked to the union of those blocks
# ("oracle": scores use the group's actual activations, non-causal within
# the group). Arms per layer-partition file (from p3_collect_cluster_all.py):
#   dense            per-token top-K sparsity level 0 (validated == dense)
#   per_token        per-token unstructured top-K at s (in-protocol anchor)
#   block/clustered  PPMI-clustered partitions, sweep g x B
#   block/random     random balanced partitions   (control)
# All arms share one PPL protocol (non-overlapping ctx-token chunks over the
# full wikitext-2 test set), so comparisons are internally consistent.

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
import torch
import torch.nn.functional as F


def patched_mlp_forward(self, x):
    from inference.modeling_llama_larosa import top_k_new
    i = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    mode = getattr(self, 'p3_mode', 'per_token')
    if mode == 'per_token':
        i = top_k_new(i, self.sparse_level_h2)
    elif mode == 'block':
        Bsz, T, d = i.shape
        g = self.p3_group
        ng = T // g
        M = self.p3_M                                        # [d, nb] fp32
        score = i.abs().float() * self.p3_wnorm              # [B, T, d] fp32
        gs = score[:, :ng * g].reshape(Bsz, ng, g, d).sum(dim=2)
        bscore = gs @ M                                      # [B, ng, nb]
        top = torch.topk(bscore, self.p3_m, dim=-1).indices
        keep_b = torch.zeros_like(bscore).scatter_(-1, top, 1.0)
        keep = (keep_b @ M.T) > 0                            # [B, ng, d]
        mask = keep.unsqueeze(2).expand(Bsz, ng, g, d).reshape(Bsz, ng * g, d)
        if ng * g < T:                                       # tail group
            tg = score[:, ng * g:].sum(dim=1)                # [B, d]
            tb = torch.topk(tg @ M, self.p3_m, dim=-1).indices
            tkeep_b = torch.zeros(Bsz, M.shape[1], device=i.device,
                                  dtype=torch.float32).scatter_(-1, tb, 1.0)
            tkeep = ((tkeep_b @ M.T) > 0).unsqueeze(1).expand(
                Bsz, T - ng * g, d)
            mask = torch.cat([mask, tkeep], dim=1)
        i = i * mask
    return self.down_proj(i)


@torch.no_grad()
def eval_ppl(model, testenc, ctx, nsamples, device):
    nll_sum, tok = 0.0, 0
    for k in range(nsamples):
        inp = testenc[:, k * ctx:(k + 1) * ctx].to(device)
        logits = model(inp).logits.float()
        loss = F.cross_entropy(logits[0, :-1], inp[0, 1:], reduction='sum')
        nll_sum += loss.item()
        tok += ctx - 1
    return float(torch.exp(torch.tensor(nll_sum / tok)))


def main():
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from inference.modeling_llama_larosa import LlamaForCausalLM, LlamaMLP

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--partitions', type=str, required=True)
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--group_sizes', type=str, default="16,64")
    parser.add_argument('--block_sizes', type=str, default="64,256")
    parser.add_argument('--nsamples', type=int, default=10000)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--out', type=str, default="p3_block_ppl.pt")
    parser.add_argument('--attn', type=str, default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    args = parser.parse_args()

    group_sizes = [int(x) for x in args.group_sizes.split(',')]
    block_sizes = [int(x) for x in args.block_sizes.split(',')]

    part = torch.load(args.partitions, map_location='cpu', weights_only=False)
    assert abs(part['sparsity'] - args.sparsity) < 1e-9, \
        "partition file was built at a different sparsity"

    config = transformers.AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = args.attn
    config.torch_dtype = 'bfloat16'
    config.sparse_mode = 'topk_intermediate'
    config.sparse_level = args.sparsity
    config.Q_path = None
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map='auto',
        config=config)
    model.eval()
    LlamaMLP.forward = patched_mlp_forward
    device = next(model.parameters()).device

    layers = model.model.layers
    d = config.intermediate_size
    for layer in layers:
        w = layer.mlp.down_proj.weight.float().norm(dim=0)   # [d]
        layer.mlp.p3_wnorm = w

    onehots = {}                                             # (li, B, name) -> M
    for li in range(len(layers)):
        for B in block_sizes:
            for name in ('clustered', 'random'):
                a = part['partitions'][li][B][name].to(device)
                M = torch.zeros(d, d // B, device=device, dtype=torch.float32)
                M[torch.arange(d, device=device), a] = 1.0
                onehots[(li, B, name)] = M

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(sample["text"] for sample in dataset)
    testenc = tokenizer(text, return_tensors="pt").input_ids
    nsamples = min(args.nsamples, testenc.numel() // args.ctx)
    print(f"PPL protocol: {nsamples} x {args.ctx} non-overlapping test tokens")

    K = round((1 - args.sparsity) * d)
    results = {}

    def set_mode(mode, g=None, B=None, name=None, level=None):
        for li, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.p3_mode = mode
            if level is not None:
                mlp.sparse_level_h2 = level
            if mode == 'block':
                mlp.p3_group = g
                mlp.p3_M = onehots[(li, B, name)]
                mlp.p3_m = max(1, round(K / B))

    set_mode('per_token', level=0.0)
    results['dense'] = eval_ppl(model, testenc, args.ctx, nsamples, device)
    print(f"dense: {results['dense']:.4f}")

    set_mode('per_token', level=args.sparsity)
    results['per_token'] = eval_ppl(model, testenc, args.ctx, nsamples, device)
    print(f"per_token s={args.sparsity}: {results['per_token']:.4f}")

    for B in block_sizes:
        m = max(1, round(K / B))
        for g in group_sizes:
            for name in ('clustered', 'random'):
                set_mode('block', g=g, B=B, name=name)
                key = f'block_{name}_B{B}_g{g}'
                results[key] = eval_ppl(
                    model, testenc, args.ctx, nsamples, device)
                print(f"{key} (m={m}, budget {m * B}/{d} = "
                      f"{m * B / d:.3f}): {results[key]:.4f}")

    print("\n===== P3 summary =====")
    print(f"dense {results['dense']:.4f} | per-token anchor "
          f"{results['per_token']:.4f}")
    for B in block_sizes:
        for g in group_sizes:
            c = results[f'block_clustered_B{B}_g{g}']
            r = results[f'block_random_B{B}_g{g}']
            print(f"B={B:>3} g={g:>2}: clustered {c:.4f}  random {r:.4f}  "
                  f"(gain {r - c:+.4f})  vs anchor "
                  f"{c - results['per_token']:+.4f}")

    torch.save({'args': vars(args), 'results': results,
                'K': K, 'nsamples': nsamples}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
