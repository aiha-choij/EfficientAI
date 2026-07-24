# coding=utf-8
# P3 prep for the coactivation-block-structure topic: collect same-token
# co-activation counts A for ALL 32 layers in one sparsified forward pass
# (A only — no window matrices, to fit 32 x 484 MB fp32 on one GPU), then
# cluster every layer into balanced blocks (PPMI -> spectral embedding ->
# balanced k-means, as validated in P2) for B in {64, 256}, plus one random
# balanced control partition per (layer, B). Assignments are saved for the
# P3 PPL evaluation (p3_block_ppl.py).

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
import torch


def balanced_kmeans(X, n_clusters, capacity, iters=25, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = X.shape[0]
    centroids = X[torch.randperm(n, generator=g)[:n_clusters]].clone()
    assign = None
    for _ in range(iters):
        d2 = torch.cdist(X, centroids)
        order = torch.argsort(d2.min(dim=1).values)
        counts = torch.zeros(n_clusters, dtype=torch.long, device=X.device)
        new_assign = torch.full((n,), -1, dtype=torch.long, device=X.device)
        pref = torch.argsort(d2, dim=1)
        for idx in order.tolist():
            for c in pref[idx].tolist():
                if counts[c] < capacity:
                    new_assign[idx] = c
                    counts[c] += 1
                    break
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(n_clusters):
            centroids[c] = X[assign == c].mean(dim=0)
    return assign


def random_assign_simple(d, capacity, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(d, generator=g)
    assign = torch.empty(d, dtype=torch.long)
    assign[perm] = torch.arange(d) // capacity
    return assign


def main():
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from inference.modeling_llama_larosa import LlamaForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--block_sizes', type=str, default="64,256")
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nsamples', type=int, default=32)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--out', type=str, default="p3_partitions.pt")
    parser.add_argument('--attn', type=str, default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(',')]

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
    for layer in model.model.layers:
        layer.mlp.sparse_level_h2 = args.sparsity
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(sample["text"] for sample in dataset)
    testenc = tokenizer(text, return_tensors="pt").input_ids
    nsamples = min(args.nsamples, testenc.numel() // args.ctx)
    device = next(model.parameters()).device

    n_layers = len(model.model.layers)
    acc = {}

    def make_hook(li):
        def hook(module, inp, out):
            sel = (inp[0] != 0)[0].float()          # [T, d]
            if li not in acc:
                d = sel.shape[-1]
                acc[li] = {'A': torch.zeros(d, d, device=sel.device),
                           'freq': torch.zeros(d, device=sel.device),
                           'tokens': 0, 'k_sum': 0.0}
            a = acc[li]
            a['A'] += sel.T @ sel
            a['freq'] += sel.sum(dim=0)
            a['tokens'] += sel.shape[0]
            a['k_sum'] += sel.sum().item()
        return hook

    hooks = [model.model.layers[li].mlp.down_proj.register_forward_hook(
        make_hook(li)) for li in range(n_layers)]
    with torch.no_grad():
        for i in range(nsamples):
            if i % 8 == 0:
                print(f"collect {i}/{nsamples}")
            model(testenc[:, i * args.ctx:(i + 1) * args.ctx].to(device))
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    eps = 1e-12
    partitions = {}
    k_eff_all = {}
    for li in range(n_layers):
        a = acc[li]
        tokens = a['tokens']
        d = a['freq'].numel()
        f = a['freq'] / tokens
        A = a['A'] / tokens
        k_eff_all[li] = a['k_sum'] / tokens
        W = torch.clamp(torch.log((A + eps) / (torch.outer(f, f) + eps)),
                        min=0.0)
        W.fill_diagonal_(0.0)
        deg = W.sum(dim=1).clamp(min=eps)
        Dr = deg.rsqrt()
        Wn = W * Dr[:, None] * Dr[None, :]
        _, evecs = torch.linalg.eigh(Wn)
        X = evecs[:, -args.embed_dim:]
        X = (X / X.norm(dim=1, keepdim=True).clamp(min=eps)).cpu()
        del W, Wn, evecs
        partitions[li] = {}
        for B in block_sizes:
            n_clusters = d // B
            partitions[li][B] = {
                'clustered': balanced_kmeans(X, n_clusters, B, seed=0),
                'random': random_assign_simple(d, B, seed=1)}
        acc[li]['A'] = None                       # free the 484 MB matrix
        acc[li] = {'freq': f.cpu(), 'k_eff': k_eff_all[li]}
        torch.cuda.empty_cache()
        print(f"layer {li}: clustered (k_eff={k_eff_all[li]:.0f})")

    torch.save({'sparsity': args.sparsity, 'nsamples': nsamples,
                'ctx': args.ctx, 'model': args.model_name,
                'block_sizes': block_sizes,
                'k_eff': k_eff_all,
                'partitions': partitions}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
