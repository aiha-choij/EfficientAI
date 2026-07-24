# coding=utf-8
# P2 of the coactivation-block-structure topic: cluster neurons into balanced
# blocks from the P1 co-activation statistics, then evaluate whether the
# clustered block structure beats random balanced partitions (the strong null).
#
# Part 1 (per layer, per block size B) — from the P1 .pt only:
#   PPMI_jj' = max(0, log(A_jj'/tokens / (f_j f_j')))  (diag zeroed)
#   spectral embedding: top-`embed_dim` eigenvectors of D^-1/2 PPMI D^-1/2
#   balanced k-means (capacity B per cluster) on row-normalized embedding
#   static metrics vs `seeds` random balanced partitions:
#     within-block mass on A, A^16, A^64:  sum_{z_j = z_j'} A_jj' / sum A_jj'
#         (off-diagonal only; higher = co-activation concentrated in blocks)
#     static coverage @ budget (from f): pick m = round(K_eff/B) blocks with
#         largest sum of f_j; coverage = sum_{j in chosen} f_j / K_eff
# Part 2 — one sparsified forward pass (32 x 2048 tokens), hooks on the same
# layers, evaluating every stored partition on real token data:
#   block-level union (per group of g tokens): touched_blocks * B / |union|
#         (>= 1; 1 = the union is perfectly tiled by blocks)
#   blocks touched per group, and union/K for reference
#   per-token dynamic coverage: top-m blocks by per-token selected count,
#         coverage = (selected neurons inside those blocks) / K_t
# All comparisons reported as clustered vs mean-over-seeds random ratio.

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
import torch


def balanced_kmeans(X, n_clusters, capacity, iters=25, seed=0):
    # X: [n, e] row-normalized embedding. Returns assignment [n] with each
    # cluster holding exactly `capacity` points (n = n_clusters * capacity).
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = X.shape[0]
    centroids = X[torch.randperm(n, generator=g)[:n_clusters]].clone()
    assign = None
    for _ in range(iters):
        d2 = torch.cdist(X, centroids)                     # [n, k]
        order = torch.argsort(d2.min(dim=1).values)        # confident first
        counts = torch.zeros(n_clusters, dtype=torch.long, device=X.device)
        new_assign = torch.full((n,), -1, dtype=torch.long, device=X.device)
        pref = torch.argsort(d2, dim=1)                    # [n, k]
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


def within_block_mass(A, assign, n_clusters):
    # off-diagonal co-activation mass captured inside blocks
    d = A.shape[0]
    onehot = torch.zeros(d, n_clusters, device=A.device)
    onehot[torch.arange(d, device=A.device), assign] = 1.0
    block_mass = torch.einsum('jc,jk,kc->', onehot, A, onehot)
    diag = A.diagonal().sum()
    total = A.sum() - diag
    return ((block_mass - diag) / total).item()


def random_assign_simple(d, capacity, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(d, generator=g)
    assign = torch.empty(d, dtype=torch.long)
    assign[perm] = torch.arange(d) // capacity
    return assign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--block_sizes', type=str, default="64,128,256")
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--group_sizes', type=str, default="16,64")
    parser.add_argument('--nsamples', type=int, default=32)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--out', type=str, default="coactivation_blocks.pt")
    parser.add_argument('--attn', type=str, default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(',')]
    group_sizes = [int(x) for x in args.group_sizes.split(',')]
    device = 'cuda'

    print(f"loading P1 stats: {args.stats}")
    p1 = torch.load(args.stats, map_location='cpu', weights_only=False)
    layers = p1['layers']
    sparsity = p1['sparsity']

    # ---------- Part 1: clustering + static metrics ----------
    partitions = {}   # (layer, B) -> {'clustered': assign, 'random': [assign]}
    static = {}       # (layer, B) -> metric dict
    for li in layers:
        r = p1['results'][li]
        tokens = r['tokens']
        d = r['freq'].numel()
        f = (r['freq'] / tokens).to(device)
        A = (r['A'] / tokens).to(device).float()
        k_eff = r['k_eff']

        eps = 1e-12
        pmi = torch.log((A + eps) / (torch.outer(f, f) + eps))
        W = torch.clamp(pmi, min=0.0)
        W.fill_diagonal_(0.0)
        deg = W.sum(dim=1).clamp(min=eps)
        Dr = deg.rsqrt()
        Wn = W * Dr[:, None] * Dr[None, :]
        print(f"layer {li}: eigh on {d}x{d} ...")
        evals, evecs = torch.linalg.eigh(Wn)
        X = evecs[:, -args.embed_dim:]
        X = X / X.norm(dim=1, keepdim=True).clamp(min=eps)
        X = X.cpu()

        Aw = {g: (r['Aw'][g] / r['w_pairs'][g]).float() for g in r['Aw']}
        for B in block_sizes:
            n_clusters = d // B
            assign_c = balanced_kmeans(X, n_clusters, B, seed=0)
            assign_r = [random_assign_simple(d, B, seed=s + 1)
                        for s in range(args.seeds)]
            partitions[(li, B)] = {'clustered': assign_c, 'random': assign_r}

            def stat_pack(assign):
                a_dev = assign.to(device)
                out = {'mass_A': within_block_mass(A, a_dev, n_clusters)}
                for g, Ag in Aw.items():
                    out[f'mass_A{g}'] = within_block_mass(
                        Ag.to(device), a_dev, n_clusters)
                m = max(1, round(k_eff / B))
                fb = torch.zeros(n_clusters, device=device).index_add_(
                    0, a_dev, f)
                top = torch.topk(fb, m).values.sum()
                out['static_cov'] = (top / k_eff).item()
                out['m_blocks'] = m
                return out

            sc = stat_pack(assign_c)
            sr = [stat_pack(a) for a in assign_r]
            static[(li, B)] = {'clustered': sc, 'random': sr}
            rm = lambda key: sum(x[key] for x in sr) / len(sr)
            print(f"  L{li} B={B}: mass_A {sc['mass_A']:.4f} vs rand "
                  f"{rm('mass_A'):.4f} ({sc['mass_A']/max(rm('mass_A'),1e-9):.2f}x)  "
                  f"static_cov {sc['static_cov']:.3f} vs {rm('static_cov'):.3f}")
        del A, W, Wn, pmi, evecs
        torch.cuda.empty_cache()

    # ---------- Part 2: dynamic metrics from one sparsified forward ----------
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from inference.modeling_llama_larosa import LlamaForCausalLM

    config = transformers.AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = args.attn
    config.torch_dtype = 'bfloat16'
    config.sparse_mode = 'topk_intermediate'
    config.sparse_level = sparsity
    config.Q_path = None
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map='auto',
        config=config)
    model.eval()
    for layer in model.model.layers:
        layer.mlp.sparse_level_h2 = sparsity
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(sample["text"] for sample in dataset)
    testenc = tokenizer(text, return_tensors="pt").input_ids
    nsamples = min(args.nsamples, testenc.numel() // args.ctx)
    mdev = next(model.parameters()).device

    onehots = {}
    for (li, B), p in partitions.items():
        n_clusters = p['clustered'].numel() // B
        packs = []
        for name, assigns in (('clustered', [p['clustered']]),
                              ('random', p['random'])):
            for a in assigns:
                a_dev = a.to(mdev)
                M = torch.zeros(a.numel(), n_clusters, device=mdev,
                                dtype=torch.float32)
                M[torch.arange(a.numel(), device=mdev), a_dev] = 1.0
                packs.append((name, M))
        onehots[(li, B)] = packs

    acc = {}  # (li, B, name) accumulators

    def make_hook(li):
        def hook(module, inp, out):
            sel = (inp[0] != 0)[0].float()               # [T, d]
            T = sel.shape[0]
            k_eff = p1['results'][li]['k_eff']
            for B in block_sizes:
                for name, M in onehots[(li, B)]:
                    n_clusters = M.shape[1]
                    counts = sel @ M                      # [T, n_clusters]
                    m = max(1, round(k_eff / B))
                    topv = torch.topk(counts, m, dim=1).values.sum(dim=1)
                    dyn_cov = (topv / sel.sum(dim=1).clamp(min=1)).sum().item()
                    key = (li, B, name)
                    a = acc.setdefault(key, {
                        'dyn_cov': 0.0, 'tok': 0,
                        **{f'bu_{g}': 0.0 for g in group_sizes},
                        **{f'grp_{g}': 0 for g in group_sizes}})
                    a['dyn_cov'] += dyn_cov
                    a['tok'] += T
                    for g in group_sizes:
                        ng = T // g
                        u = sel[:ng * g].reshape(ng, g, -1).amax(dim=1)  # [ng, d]
                        bc = ((u @ M) > 0).float().sum(dim=1)            # blocks touched
                        usz = u.sum(dim=1).clamp(min=1)
                        a[f'bu_{g}'] += (bc * B / usz).sum().item()
                        a[f'grp_{g}'] += ng
        return hook

    hooks = [model.model.layers[li].mlp.down_proj.register_forward_hook(
        make_hook(li)) for li in layers]
    with torch.no_grad():
        for i in range(nsamples):
            if i % 8 == 0:
                print(f"forward {i}/{nsamples}")
            model(testenc[:, i * args.ctx:(i + 1) * args.ctx].to(mdev))
    for h in hooks:
        h.remove()

    # average the random seeds (they share the name 'random' -> accumulated
    # together; divide by seeds via the tok/grp counters which also summed)
    dynamic = {}
    for (li, B, name), a in acc.items():
        dynamic[(li, B, name)] = {
            'dyn_cov': a['dyn_cov'] / a['tok'],
            **{f'block_union_{g}': a[f'bu_{g}'] / a[f'grp_{g}']
               for g in group_sizes}}

    print("\n===== P2 summary (clustered vs random-mean ratio) =====")
    summary = {}
    for li in layers:
        for B in block_sizes:
            c = dynamic[(li, B, 'clustered')]
            r = dynamic[(li, B, 'random')]
            sc = static[(li, B)]['clustered']
            sr = static[(li, B)]['random']
            rmean = lambda key: sum(x[key] for x in sr) / len(sr)
            row = {
                'mass_A_ratio': sc['mass_A'] / max(rmean('mass_A'), 1e-9),
                'mass_A_clustered': sc['mass_A'],
                'mass_A_random': rmean('mass_A'),
                'static_cov_ratio': sc['static_cov'] / max(rmean('static_cov'), 1e-9),
                'dyn_cov_clustered': c['dyn_cov'],
                'dyn_cov_random': r['dyn_cov'],
                'dyn_cov_ratio': c['dyn_cov'] / max(r['dyn_cov'], 1e-9),
            }
            for g in group_sizes:
                row[f'block_union_{g}_clustered'] = c[f'block_union_{g}']
                row[f'block_union_{g}_random'] = r[f'block_union_{g}']
                row[f'block_union_{g}_ratio'] = (
                    r[f'block_union_{g}'] / max(c[f'block_union_{g}'], 1e-9))
            for g in p1['results'][li]['Aw']:
                row[f'mass_A{g}_ratio'] = (
                    sc[f'mass_A{g}'] / max(rmean(f'mass_A{g}'), 1e-9))
            summary[(li, B)] = row
            print(f"L{li:>2} B={B:>3}: mass_A {row['mass_A_ratio']:.2f}x  "
                  f"dyn_cov {c['dyn_cov']:.3f}/{r['dyn_cov']:.3f} "
                  f"({row['dyn_cov_ratio']:.2f}x)  "
                  + "  ".join(
                      f"blkU{g} {c[f'block_union_{g}']:.2f}/"
                      f"{r[f'block_union_{g}']:.2f} "
                      f"({row[f'block_union_{g}_ratio']:.2f}x)"
                      for g in group_sizes))

    torch.save({'args': vars(args), 'sparsity': sparsity, 'layers': layers,
                'block_sizes': block_sizes, 'group_sizes': group_sizes,
                'partitions': {f"{li}_{B}": {
                    'clustered': p['clustered'],
                    'random': p['random']}
                    for (li, B), p in partitions.items()},
                'static': {f"{li}_{B}": v for (li, B), v in static.items()},
                'dynamic': {f"{li}_{B}_{n}": v
                            for (li, B, n), v in dynamic.items()},
                'summary': {f"{li}_{B}": v for (li, B), v in summary.items()},
                }, args.out)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
