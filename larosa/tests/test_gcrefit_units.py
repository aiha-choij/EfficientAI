# coding=utf-8
# Unit tests for the GC-refit E0 diagnostic (see inference/gc_refit.py).
# CPU, tiny random LLaMA in fp32, same discipline as
# test_block_comp_units.py / test_oracle_units.py. Run directly:
#   python tests/test_gcrefit_units.py
#
# Tests:
# 1. mu/mask consistency  : gc_refit's _block_p3_mask_mu m_bool matches
#                            block_comp_mlp.block_p3_mask bit-exactly, and
#                            mu correctly expands back to m_bool via M
# 2. collect_a accumulation: gc_G/gc_C/gc_Y2/gc_n match a brute-force
#                            reference computed directly from z/y* outside
#                            the model
# 3. balanced_hamming_kmeans: recovers a synthetic separable 2-cluster
#                            structure, exact 50/50 balance
# 4. assign_nearest_centroid: fit-set self-assignment via its own centroids
#                            mostly recovers the original balanced labels
# 5. collect_a/collect_b agreement: G0+G1 == G_marginal (same underlying
#                            forward computation, split two different ways)
# 6. closed_form_mse       : matches a brute-force SSE computed directly
#                            from z/y*
# 7. ridge sanity           : solve_refit's fit-set MSE never exceeds the
#                            anchor's fit-set MSE (any lambda >= 0) -- the
#                            same guarantee the E0 driver script checks at
#                            full (LLaMA2-7B) scale
# 8. oracle_avg formula     : block-average compensation matches a
#                            brute-force per-block average computed outside
#                            the model; full-keep (m=all blocks kept)
#                            reduces to dense

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch

from inference.configuration_llama import LlamaConfig
from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, block_comp_mlp, gc_refit, refit_mlp

H, D, LAYERS, VOCAB = 64, 176, 2, 128


def build_model():
    """Same two-step build as test_block_comp_units.py: calibrate g_bar/
    col_norm under sparse_mode='oracle', then flip to 'gc_refit' in place."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=H, intermediate_size=D,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, use_cache=False,
    )
    config._attn_implementation = "eager"
    config.sparse_mode = "oracle"
    config.oracle_condition = "dense"
    model = LlamaForCausalLM(config).float().eval()

    oracle_mlp.enable_stats_mode(model)
    with torch.no_grad():
        for _ in range(4):
            model(torch.randint(0, VOCAB, (2, 64)))
    oracle_mlp.finalize_stats(model)
    oracle_mlp.attach_col_norms(model)
    return model


def to_gc_refit(model, g=8):
    for layer in model.model.layers:
        layer.sparse_mode = "gc_refit"
        layer.self_attn.sparse_mode = "gc_refit"
        layer.mlp.sparse_mode = "gc_refit"
        layer.mlp.blk_g = g
        layer.mlp.blk_seq_mask = None


def make_partition_onehot(Dd, B):
    """Same synthetic partition convention as
    test_block_comp_units.py's _make_partition_onehot: consecutive-neuron
    blocks of size B."""
    assign = torch.arange(Dd) // B
    nb = int(assign[-1].item()) + 1
    M = torch.zeros(Dd, nb, dtype=torch.float32)
    M[torch.arange(Dd), assign] = 1.0
    return M


def attach_synthetic_partitions(model, B, m_keep):
    for _, mlp in oracle_mlp.iter_mlps(model):
        mlp.blk_neuron_M = make_partition_onehot(D, B)
        mlp.blk_neuron_m = m_keep


def test_1_mu_mask_consistency(model):
    to_gc_refit(model, g=4)
    B, m_keep = 8, 3
    attach_synthetic_partitions(model, B, m_keep)
    _, mlp = next(oracle_mlp.iter_mlps(model))
    torch.manual_seed(3)
    x = torch.randn(2, 20, H)
    with torch.no_grad():
        u = mlp.up_proj(x)
        g = mlp.act_fn(mlp.gate_proj(x))
        m_bool, mu, bid = gc_refit._block_p3_mask_mu(mlp, u, g)

        score = block_comp_mlp._resid_score(mlp, u, g)
        m_ref = block_comp_mlp.block_p3_mask(score, mlp.blk_g, mlp.blk_neuron_M, mlp.blk_neuron_m)
    assert torch.equal(m_bool, m_ref), "gc_refit's m_bool diverges from block_comp_mlp.block_p3_mask"
    assert mu.shape[-1] == mlp.blk_neuron_M.shape[-1]
    assert (mu.sum(-1) == m_keep).all(), "every token-block should keep exactly m_keep neuron-blocks"
    re_expand = (mu.float() @ mlp.blk_neuron_M.T) > 0
    re_expand = re_expand.index_select(-2, bid)
    assert torch.equal(re_expand, m_bool), "mu does not re-expand to m_bool via M"
    print("PASS mu/mask consistency: gc_refit's m_bool == block_comp_mlp.block_p3_mask, mu re-expands correctly")


def test_2_collect_a_accumulation(model):
    to_gc_refit(model, g=4)
    B, m_keep = 8, 3
    attach_synthetic_partitions(model, B, m_keep)
    gc_refit.enable_collect_a(model)
    torch.manual_seed(4)
    ids = torch.randint(0, VOCAB, (1, 32))

    # brute-force reference computed independently, outside the model's
    # accumulation path
    ref = {}
    for layer_idx, mlp in oracle_mlp.iter_mlps(model):
        ref[layer_idx] = {"G": torch.zeros(D, D), "C": torch.zeros(H, D), "Y2": torch.zeros(())}

    hooks = []

    def make_hook(layer_idx, mlp):
        def hook(module, inp):
            x = inp[0]
            with torch.no_grad():
                u = mlp.up_proj(x)
                g = mlp.act_fn(mlp.gate_proj(x))
                i = u * g
                m_bool, mu, bid = gc_refit._block_p3_mask_mu(mlp, u, g)
                y = mlp.down_proj(i).float().reshape(-1, H)
                z = (m_bool.to(i.dtype) * i).float().reshape(-1, D)
                ref[layer_idx]["G"] += z.T @ z
                ref[layer_idx]["C"] += y.T @ z
                ref[layer_idx]["Y2"] += (y * y).sum()
        return hook

    for layer_idx, mlp in oracle_mlp.iter_mlps(model):
        hooks.append(mlp.register_forward_pre_hook(make_hook(layer_idx, mlp)))

    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    stats = gc_refit.finalize_collect_a(model)
    for layer_idx, st in stats.items():
        assert torch.allclose(st["G"], ref[layer_idx]["G"], atol=1e-3), f"layer {layer_idx}: G mismatch"
        assert torch.allclose(st["C"], ref[layer_idx]["C"], atol=1e-3), f"layer {layer_idx}: C mismatch"
        assert torch.allclose(st["Y2"], ref[layer_idx]["Y2"], atol=1e-2), f"layer {layer_idx}: Y2 mismatch"
        assert st["n"] == ids.shape[0] * ids.shape[1]
    print(f"PASS collect_a accumulation matches brute-force reference ({len(stats)} layers)")


def test_3_balanced_hamming_kmeans():
    torch.manual_seed(6)
    Bfeat = 20
    n_each = 150
    center_a = torch.zeros(Bfeat)
    center_a[:10] = 1.0
    center_b = torch.zeros(Bfeat)
    center_b[10:] = 1.0
    flip = torch.rand(n_each, Bfeat) < 0.05  # 5% bit-flip noise
    a = center_a.unsqueeze(0).expand(n_each, -1).clone()
    a[flip[:n_each]] = 1.0 - a[flip[:n_each]]
    flip2 = torch.rand(n_each, Bfeat) < 0.05
    b = center_b.unsqueeze(0).expand(n_each, -1).clone()
    b[flip2] = 1.0 - b[flip2]
    mu = torch.cat([a, b], dim=0).bool()
    true_labels = torch.cat([torch.zeros(n_each, dtype=torch.long), torch.ones(n_each, dtype=torch.long)])

    labels, centroids = gc_refit.balanced_hamming_kmeans(mu, seed=0)
    n0 = int((labels == 0).sum())
    assert abs(n0 - n_each) <= 1, f"expected ~balanced split, got n0={n0} of {2 * n_each}"

    # cluster identity (0 vs 1) is arbitrary -- check agreement with true
    # labels either directly or flipped, whichever is higher
    agree = (labels == true_labels).float().mean().item()
    agree_flipped = (labels == (1 - true_labels)).float().mean().item()
    best_agree = max(agree, agree_flipped)
    assert best_agree > 0.9, f"expected recovery of the synthetic 2-cluster structure, got agreement {best_agree}"
    print(f"PASS balanced_hamming_kmeans: n0={n0}/{2 * n_each} (balanced), label agreement {best_agree:.3f}")


def test_4_assign_nearest_centroid():
    torch.manual_seed(7)
    Bfeat = 16
    n = 100
    center_a = torch.zeros(Bfeat)
    center_a[:8] = 1.0
    center_b = 1.0 - center_a
    a = center_a.unsqueeze(0).expand(n, -1).clone()
    b = center_b.unsqueeze(0).expand(n, -1).clone()
    mu = torch.cat([a, b], dim=0).bool()
    labels, centroids = gc_refit.balanced_hamming_kmeans(mu, seed=1)
    self_labels = gc_refit.assign_nearest_centroid(mu, centroids)
    agree = (self_labels == labels).float().mean().item()
    assert agree > 0.95, f"self-assignment via fit centroids should mostly recover the fit labels, got {agree}"
    print(f"PASS assign_nearest_centroid: self-assignment agreement {agree:.3f}")


def test_5_collect_a_b_agree(model):
    to_gc_refit(model, g=4)
    B, m_keep = 8, 3
    attach_synthetic_partitions(model, B, m_keep)

    gc_refit.enable_collect_a(model)
    torch.manual_seed(9)
    seqlen = 32
    nb = seqlen // 4
    ids = torch.randint(0, VOCAB, (1, seqlen))
    with torch.no_grad():
        model(ids)
    stats_a = gc_refit.finalize_collect_a(model)

    torch.manual_seed(21)
    cluster_ids_by_layer = {li: (torch.rand(nb) < 0.5).long() for li in stats_a}
    gc_refit.enable_collect_b(model, cluster_ids_by_layer, blocks_per_seq=nb)
    attach_synthetic_partitions(model, B, m_keep)  # enable_collect_b doesn't touch blk_neuron_M
    gc_refit.set_seq_idx(model, 0)
    with torch.no_grad():
        model(ids)
    stats_b = gc_refit.finalize_collect_b(model)

    for layer_idx in stats_a:
        G_sum = stats_b[layer_idx][0]["G"] + stats_b[layer_idx][1]["G"]
        C_sum = stats_b[layer_idx][0]["C"] + stats_b[layer_idx][1]["C"]
        n_sum = stats_b[layer_idx][0]["n"] + stats_b[layer_idx][1]["n"]
        assert torch.allclose(G_sum, stats_a[layer_idx]["G"], atol=1e-2), f"layer {layer_idx}: G0+G1 != G_marginal"
        assert torch.allclose(C_sum, stats_a[layer_idx]["C"], atol=1e-2), f"layer {layer_idx}: C0+C1 != C_marginal"
        assert n_sum == stats_a[layer_idx]["n"], f"layer {layer_idx}: n0+n1 != n_marginal"
    print(f"PASS collect_a/collect_b agreement: G0+G1==G_marginal, C0+C1==C_marginal, n0+n1==n_marginal ({len(stats_a)} layers)")


def test_6_closed_form_mse_matches_bruteforce():
    torch.manual_seed(10)
    d, h, n = 12, 5, 40
    Z = torch.randn(n, d)
    Y = torch.randn(n, h)
    W = torch.randn(h, d)
    G = Z.T @ Z
    C = Y.T @ Z
    Y2 = (Y * Y).sum()
    ref_mse = ((Y - Z @ W.T) ** 2).sum().item() / (n * h)
    got_mse = gc_refit.closed_form_mse(W, G, C, Y2, n)
    assert abs(ref_mse - got_mse) < 1e-4, f"closed_form_mse {got_mse} != brute-force {ref_mse}"
    print(f"PASS closed_form_mse matches brute-force SSE/(-n*h): {got_mse:.6f} vs {ref_mse:.6f}")


def test_7_ridge_never_worse_than_anchor():
    torch.manual_seed(11)
    d, h, n = 14, 6, 60
    Z = torch.randn(n, d)
    W_true = torch.randn(h, d)
    Y = Z @ W_true.T + 0.1 * torch.randn(n, h)
    W_anchor = torch.randn(h, d)  # deliberately NOT the true weight
    G = Z.T @ Z
    C = Y.T @ Z
    Y2 = (Y * Y).sum()
    anchor_mse = gc_refit.closed_form_mse(W_anchor, G, C, Y2, n)
    for lam in (0.0001, 0.01, 0.1, 1.0, 10.0):
        W_tilde = refit_mlp.solve_refit(G, C, W_anchor, lam=lam)
        tilde_mse = gc_refit.closed_form_mse(W_tilde, G, C, Y2, n)
        assert tilde_mse <= anchor_mse + 1e-6, \
            f"lam={lam}: refit fit-MSE {tilde_mse} exceeds anchor fit-MSE {anchor_mse} -- BUG"
    print(f"PASS ridge sanity: refit fit-MSE <= anchor fit-MSE for all lambda (anchor_mse={anchor_mse:.4f})")


def test_8_oracle_avg_formula(model):
    to_gc_refit(model, g=4)
    B, m_keep = 8, 3
    attach_synthetic_partitions(model, B, m_keep)
    gc_refit.enable_oracle_avg(model)
    _, mlp = next(oracle_mlp.iter_mlps(model))
    torch.manual_seed(12)
    x = torch.randn(1, 16, H)
    g_blk = mlp.blk_g
    with torch.no_grad():
        y = gc_refit._oracle_avg_forward(mlp, x)

        u = mlp.up_proj(x)
        gate = mlp.act_fn(mlp.gate_proj(x))
        i = u * gate
        m_bool, mu, bid = gc_refit._block_p3_mask_mu(mlp, u, gate)
        m = m_bool.float()
        kept = mlp.down_proj(m * i)
        tail = mlp.down_proj((1.0 - m) * i)
        nb = int(bid[-1].item()) + 1
        ref = torch.zeros_like(y)
        for blk in range(nb):
            idx = (bid == blk).nonzero(as_tuple=True)[0]
            avg = tail[:, idx].mean(dim=1, keepdim=True)
            ref[:, idx] = kept[:, idx] + avg
    diff = (y - ref).abs().max().item()
    assert diff < 1e-4, f"oracle_avg formula mismatch: max diff {diff}"
    print(f"PASS oracle_avg formula matches brute-force per-block average: max diff {diff:.2e}")

    # full-keep identity: all neuron-blocks kept -> tail is exactly 0 -> comp is 0 -> reduces to dense
    nb_neuron = mlp.blk_neuron_M.shape[-1]
    mlp.blk_neuron_m = nb_neuron
    with torch.no_grad():
        y_full = gc_refit._oracle_avg_forward(mlp, x)
        y_dense = mlp.down_proj(i)
    diff_dense = (y_full - y_dense).abs().max().item()
    assert diff_dense < 1e-4, f"oracle_avg full-keep vs dense: max diff {diff_dense}"
    print(f"PASS oracle_avg full-keep == dense: max diff {diff_dense:.2e}")


def _load_e0_driver():
    """scripts/gcrefit/01_run_e0.py starts with digits -- not a valid module
    name for a plain `import` -- load it by path instead."""
    import importlib.util
    driver_path = os.path.join(parent_dir, "scripts", "gcrefit", "01_run_e0.py")
    spec = importlib.util.spec_from_file_location("gcrefit_e0_driver", driver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_9_full_part1_pipeline_smoke(model):
    """End-to-end smoke test of run_part1_core (mu-collect -> balanced
    clustering -> chunked marginal/cluster (G,C) accumulation -> solve ->
    sanity check -> held-out mu-collect -> nearest-centroid assignment ->
    chunked cross-eval -> heterogeneity-gain report) against the tiny
    synthetic model, with layers_budget_gb forced tiny so BOTH the marginal
    (mult=1) and cluster-split (mult=2) accumulation passes are actually
    exercised as multi-chunk sweeps (num_layers=2 layers, forced
    layers_per_pass=1 -> 2 chunks each) -- this is the orchestration logic
    that isn't covered by the smaller unit tests above, and is exactly what
    the real LLaMA2-7B run also does (just at a much bigger layers_per_pass).
    Only checks that the pipeline runs to completion and produces
    self-consistent, well-formed output -- not that any particular
    heterogeneity-gain number comes out (this is random tiny synthetic
    data, no expected signal)."""
    import argparse
    import tempfile

    driver = _load_e0_driver()
    to_gc_refit(model, g=4)
    B, m_keep = 8, 3
    attach_synthetic_partitions(model, B, m_keep)

    torch.manual_seed(42)
    seqlen, fit_n, held_n = 16, 6, 2
    fit_tokens = torch.randint(0, VOCAB, (fit_n, seqlen))
    held_tokens = torch.randint(0, VOCAB, (held_n, seqlen))

    out_dir = tempfile.mkdtemp(prefix="gcrefit_e0_smoke_")
    args = argparse.Namespace(
        model_name="dummy", dataset="c4", nsamples=fit_n + held_n, seqlen=seqlen,
        fit_n=fit_n, held_n=held_n, seed=0, g=4, B=B, sparsity=0.9,
        lambdas=[0.01, 0.1], partitions="dummy", stats_dir="dummy",
        out_dir=out_dir, layers_budget_gb=1e-6,  # forces layers_per_pass=1
    )
    out = driver.run_part1_core(model, fit_tokens, held_tokens, torch.device("cpu"),
                                args, m_keep=m_keep, K=m_keep * B, tok_path="dummy")

    assert os.path.exists(os.path.join(out_dir, "part1_e0_report.json"))
    assert os.path.exists(os.path.join(out_dir, "part1_clusters.pt"))
    for lam_str in ("0.01", "0.1"):
        assert lam_str in out["part1"], f"missing lambda={lam_str} in report"
        per_layer = out["part1"][lam_str]["per_layer"]
        assert set(per_layer.keys()) == {0, 1}, f"expected 2 layers, got {set(per_layer.keys())}"
        for li, row in per_layer.items():
            n0, n1 = row["held_out_count"][0], row["held_out_count"][1]
            assert n0 + n1 == held_n * seqlen, \
                f"layer {li}: held-out token count {n0}+{n1} != {held_n * seqlen}"
            for key in ("M00", "M01", "M10", "M11", "Mmarg0", "Mmarg1"):
                assert row[key] >= 0, f"layer {li} {key}: negative MSE {row[key]}"
        assert out["sanity"][lam_str]["ok"], f"lambda={lam_str}: ridge sanity violated -- {out['sanity'][lam_str]}"
    print(f"PASS full Part1 pipeline smoke test: 2 layers x 2 lambdas, chunked (layers_per_pass=1), "
          f"outputs written to {out_dir}, sanity OK for both lambdas")


if __name__ == "__main__":
    model = build_model()
    test_1_mu_mask_consistency(model)
    test_2_collect_a_accumulation(build_model())
    test_3_balanced_hamming_kmeans()
    test_4_assign_nearest_centroid()
    test_5_collect_a_b_agree(build_model())
    test_6_closed_form_mse_matches_bruteforce()
    test_7_ridge_never_worse_than_anchor()
    test_9_full_part1_pipeline_smoke(build_model())
    test_8_oracle_avg_formula(build_model())
    print("ALL GC-REFIT UNIT TESTS PASSED")
