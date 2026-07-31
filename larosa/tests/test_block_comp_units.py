# coding=utf-8
# Unit tests for the block-shared-mask compensation conditions C7a/C7/C8a/C8
# (see inference/block_comp_mlp.py; spec at
# .labtool/topics/block-sparse-compensation/spec.md). Runs on CPU with a
# tiny random LLaMA in fp32, same discipline as test_oracle_units.py. Run
# directly: python tests/test_block_comp_units.py
#
# Required unit tests (spec section 2 "필수 unit test"):
# 1. g=1 reduction    : C7 at g=1 matches oracle C4 bit-near-exactly
# 2. p=1 identity     : C7 (full-rank comp) and C8/C8a (any sketch rank)
#                       match dense
# 3. C8 full-rank      : r_sk = full -> C8 == dense (any p, any g)
# 4. block sharing     : mask is uniform within every block; sequence
#                        (batch-row) boundaries never mixed; padding excluded
# 5. mask-vs-slice     : full-compute-then-mask == column-sliced skip
#                        (oracle spec unit test 4, one layer)
#
# Phase 4 (P3') tests -- neuron-block-quantized mask selection combined
# with the same token-block sharing (g) above (block_p3_mask/
# neuron_block_topm_mask in inference/block_comp_mlp.py):
# 6. B=1 reduction     : neuron-block selection with block size 1 (each
#                        neuron its own block) matches plain top-count
# 7. full-keep identity: neuron_m = nb_neuron (all blocks kept) == dense
# 8. neuron-block sharing: mask constant within every neuron-block
# 9. c8 full-rank + P3' mask == dense (compensation formulas are
#    unaffected by neuron-block-quantized vs unstructured mask selection)

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch

from inference.configuration_llama import LlamaConfig
from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, block_comp_mlp

H, D, LAYERS, VOCAB = 64, 176, 2, 128


def build_model():
    """Build + calibrate with sparse_mode='oracle' (so the existing oracle
    dense-calibration path accumulates g_bar/col_norm), then flip every
    layer's mlp/self_attn to sparse_mode='block_comp' in place. The g_bar/
    col_norm buffers stay attached regardless of the sparse_mode string --
    same model instance, same convention oracle's own tests use for
    building shared calibration state once."""
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


def to_block_comp(model):
    for layer in model.model.layers:
        layer.sparse_mode = "block_comp"
        layer.self_attn.sparse_mode = "block_comp"
        layer.mlp.sparse_mode = "block_comp"
        layer.mlp.blk_seq_mask = None


def to_oracle(model):
    for layer in model.model.layers:
        layer.sparse_mode = "oracle"
        layer.self_attn.sparse_mode = "oracle"
        layer.mlp.sparse_mode = "oracle"


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def test_1_g1_reduces_to_c4(model, ids):
    r = H // 4
    p = 0.6
    to_oracle(model)
    oracle_mlp.attach_factors_inplace(model, rank=r)
    oracle_mlp.set_condition(model, "c4", p=p)
    y_c4 = logits(model, ids)

    to_block_comp(model)
    # same rank, same code path (build_M_factors) -> bit-identical A/B
    block_comp_mlp.attach_block_factors_inplace(model, rank=r, r_sk=1)
    block_comp_mlp.set_condition(model, "c7", p=p, g=1)
    y_c7 = logits(model, ids)

    diff = (y_c7 - y_c4).abs().max().item()
    assert diff < 1e-3, f"c7 g=1 vs c4: max diff {diff}"
    print(f"PASS g=1 reduction: c7(g=1) == c4, max logit diff {diff:.2e}")


def test_2_p1_identity(model, ids):
    to_block_comp(model)
    # dense reference: any block_comp condition at p=1 should match this
    block_comp_mlp.set_condition(model, "c7a", p=1.0, g=16)
    y_dense_ref = logits(model, ids)  # c7a at p=1 (all-kept) IS dense too

    # c7 needs full-rank comp_lr for exact p=1 identity (comp_lr(x)
    # approximates M over ALL neurons, not just the tail that p=1 makes
    # empty -- see module docstring); c8/c8a's tail is gated to exactly
    # zero at p=1 regardless of rank, so any small sketch rank works there.
    block_comp_mlp.attach_block_factors_inplace(model, rank=H, r_sk=D // 8)
    for cond, g in (("c7", 16), ("c8", 64), ("c8a", 16)):
        block_comp_mlp.set_condition(model, cond, p=1.0, g=g)
        y = logits(model, ids)
        diff = (y - y_dense_ref).abs().max().item()
        assert diff < 1e-3, f"{cond} p=1 vs dense: max diff {diff}"
        print(f"PASS p=1 identity ({cond}, g={g}): max logit diff {diff:.2e}")


def test_3_c8_full_rank(model, ids):
    to_block_comp(model)
    block_comp_mlp.set_condition(model, "c7a", p=1.0, g=1)
    y_dense = logits(model, ids)

    full_r_sk = min(H, D)
    block_comp_mlp.attach_block_factors_inplace(model, rank=H, r_sk=full_r_sk)
    for p, g in ((0.7, 16), (0.5, 64)):
        block_comp_mlp.set_condition(model, "c8", p=p, g=g)
        y_c8 = logits(model, ids)
        diff = (y_c8 - y_dense).abs().max().item()
        assert diff < 1e-3, f"c8 full-rank (p={p},g={g}) vs dense: max diff {diff}"
        print(f"PASS c8 full-rank == dense (p={p}, g={g}): max logit diff {diff:.2e}")


def test_4_block_sharing():
    torch.manual_seed(5)
    B, T, Dd, g = 3, 37, 40, 8  # seqlen not a multiple of g -> ragged last block
    score = torch.rand(B, T, Dd)
    m = block_comp_mlp.block_p_mask(score, p=0.5, g=g)
    assert m.shape == (B, T, Dd)

    bid = block_comp_mlp.block_ids(T, g)
    num_blocks = int(bid[-1].item()) + 1
    assert num_blocks == (T + g - 1) // g, (num_blocks, T, g)
    for b in range(B):
        for blk in range(num_blocks):
            idx = (bid == blk).nonzero(as_tuple=True)[0]
            block_rows = m[b, idx]
            assert torch.equal(block_rows, block_rows[0:1].expand_as(block_rows)), \
                f"batch {b} block {blk}: mask not uniform across tokens in the block"
    print(f"PASS block sharing: {num_blocks} blocks (last block ragged, "
          f"{T - (num_blocks - 1) * g} tokens), uniform mask within each")

    # padding exclusion: a fully-padded block must not corrupt other blocks'
    # scores (its contribution is zeroed before aggregation)
    last_block_size = T - (num_blocks - 1) * g
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[:, -last_block_size:] = False  # pad the ragged last block entirely
    score_block, _ = block_comp_mlp.aggregate_block_score(score, g, seq_mask=seq_mask)
    score_block_full, _ = block_comp_mlp.aggregate_block_score(score, g, seq_mask=None)
    assert torch.equal(score_block[:, :-1], score_block_full[:, :-1]), \
        "padding a later block changed an earlier block's aggregated score"
    assert (score_block[:, -1] == 0).all(), "fully-padded block should aggregate to 0"
    print("PASS padding exclusion: padded tokens contribute 0, earlier blocks unaffected")

    # sequence (batch-row) boundaries: different rows never mix into the
    # same block score (aggregate_block_score sums along dim -2 only)
    score2 = score.clone()
    score2[1] = torch.rand(T, Dd) * 100  # perturb only row 1
    agg1, _ = block_comp_mlp.aggregate_block_score(score, g)
    agg2, _ = block_comp_mlp.aggregate_block_score(score2, g)
    assert torch.equal(agg1[0], agg2[0]) and torch.equal(agg1[2], agg2[2]), \
        "perturbing row 1 changed another row's block scores -- rows are mixing"
    print("PASS sequence-boundary isolation: perturbing one row leaves others unchanged")


def test_5_mask_vs_slice(model):
    to_block_comp(model)
    _, mlp = next(oracle_mlp.iter_mlps(model))
    torch.manual_seed(2)
    x = torch.randn(2, 24, H)
    with torch.no_grad():
        u = mlp.up_proj(x)
        g = mlp.act_fn(mlp.gate_proj(x))
        i = (u * g)
        score = block_comp_mlp._resid_score(mlp, u, g)
        mask = block_comp_mlp.block_p_mask(score, p=0.6, g=8)
        i_flat, mask_flat = i.reshape(-1, D), mask.reshape(-1, D)
        y_masked = (mask_flat.float() * i_flat) @ mlp.down_proj.weight.T
        y_sliced = torch.zeros_like(y_masked)
        for t in range(i_flat.shape[0]):
            idx = mask_flat[t].nonzero(as_tuple=True)[0]
            y_sliced[t] = i_flat[t, idx] @ mlp.down_proj.weight[:, idx].T
    diff = (y_masked - y_sliced).abs().max().item()
    assert diff < 1e-4, f"mask vs slice: max diff {diff}"
    print(f"PASS mask-vs-slice equivalence (block mask): max diff {diff:.2e}")


def _make_partition_onehot(Dd, B, device=None):
    """Deterministic synthetic partition for tests: consecutive neurons
    grouped into blocks of size B (assign = arange(Dd)//B). NOT the real
    PPMI clustering (that comes from an external artifact -- see
    build_neuron_partition_onehots) -- these tests only need SOME valid
    one-hot M to exercise the block_p3_mask/neuron_block_topm_mask wiring,
    the clustering method itself is out of scope for this module."""
    assign = torch.arange(Dd, device=device) // B
    nb = int(assign[-1].item()) + 1
    M = torch.zeros(Dd, nb, device=device, dtype=torch.float32)
    M[torch.arange(Dd, device=device), assign] = 1.0
    return M


def test_6_p3_b1_reduces_to_topcount():
    torch.manual_seed(11)
    Bsz, T, Dd = 3, 20, 24
    score = torch.rand(Bsz, T, Dd)  # continuous -> ties have probability 0
    M = _make_partition_onehot(Dd, 1)  # nb_neuron == Dd (each neuron its own block)
    assert M.shape == (Dd, Dd)
    m_keep = 9
    block_score, _ = block_comp_mlp.aggregate_block_score(score, g=1)
    keep_p3 = block_comp_mlp.neuron_block_topm_mask(block_score, M, m_keep)
    keep_ref = oracle_mlp.top_count_mask(block_score, m_keep)
    assert torch.equal(keep_p3, keep_ref), "B=1 neuron-block selection should match plain top-count"
    print(f"PASS P3' B=1 reduces to top_count_mask (m={m_keep})")


def test_7_p3_full_neuron_keep_identity(model, ids):
    to_block_comp(model)
    block_comp_mlp.set_condition(model, "c7a", p=1.0, g=1)
    y_dense = logits(model, ids)

    B = 8
    nb = D // B
    parts = {li: _make_partition_onehot(D, B) for li, _ in oracle_mlp.iter_mlps(model)}
    block_comp_mlp.set_condition(model, "c7a", g=16, neuron_partitions=parts, neuron_m=nb)
    y = logits(model, ids)
    diff = (y - y_dense).abs().max().item()
    assert diff < 1e-3, f"P3' full-neuron-keep vs dense: max diff {diff}"
    print(f"PASS P3' full neuron-block keep ({nb}/{nb} blocks) == dense: max logit diff {diff:.2e}")


def test_8_p3_neuron_block_sharing():
    torch.manual_seed(13)
    Bsz, T, Dd, B, g = 2, 12, 24, 4, 3
    score = torch.rand(Bsz, T, Dd)
    M = _make_partition_onehot(Dd, B)
    nb = Dd // B
    m_keep = nb // 2
    mask = block_comp_mlp.block_p3_mask(score, g, M, m_keep)
    assert mask.shape == (Bsz, T, Dd)
    for k in range(nb):
        cols = mask[:, :, k * B:(k + 1) * B]
        assert torch.equal(cols, cols[:, :, 0:1].expand_as(cols)), \
            f"neuron block {k}: mask not uniform across its neurons"
    print(f"PASS P3' neuron-block sharing: {nb} neuron-blocks (B={B}), uniform mask within each")


def test_9_p3_c8_full_rank_dense_equivalence(model, ids):
    to_block_comp(model)
    block_comp_mlp.set_condition(model, "c7a", p=1.0, g=1)
    y_dense = logits(model, ids)

    B = 8
    nb = D // B
    parts = {li: _make_partition_onehot(D, B) for li, _ in oracle_mlp.iter_mlps(model)}
    full_r_sk = min(H, D)
    block_comp_mlp.attach_block_factors_inplace(model, rank=H, r_sk=full_r_sk)
    for g, m_keep in ((16, nb // 2), (64, nb - 3)):
        block_comp_mlp.set_condition(model, "c8", g=g, neuron_partitions=parts, neuron_m=m_keep)
        y_c8 = logits(model, ids)
        diff = (y_c8 - y_dense).abs().max().item()
        assert diff < 1e-3, f"P3' c8 full-rank (g={g}, m={m_keep}) vs dense: max diff {diff}"
        print(f"PASS P3' c8 full-rank == dense (g={g}, m={m_keep}/{nb}): max logit diff {diff:.2e}")


if __name__ == "__main__":
    model = build_model()
    ids = torch.randint(0, VOCAB, (2, 64), generator=torch.Generator().manual_seed(7))
    test_1_g1_reduces_to_c4(model, ids)
    test_2_p1_identity(model, ids)
    test_3_c8_full_rank(model, ids)
    test_4_block_sharing()
    test_5_mask_vs_slice(model)
    test_6_p3_b1_reduces_to_topcount()
    test_7_p3_full_neuron_keep_identity(model, ids)
    test_8_p3_neuron_block_sharing()
    test_9_p3_c8_full_rank_dense_equivalence(model, ids)
    print("ALL BLOCK-COMP UNIT TESTS PASSED")
