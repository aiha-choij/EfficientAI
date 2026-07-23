# coding=utf-8
# Unit tests for the oracle intermediate-sparsity conditions (spec section 2).
# Runs on CPU with a tiny random LLaMA in fp32 (same discipline as the
# topk_intermediate CPU verification). Run directly:
#   python tests/test_oracle_units.py
#
# 1. p=1 identity      : C3 (and C5) at p=1 match dense
# 2. C4 full-rank      : with r = h, C4 == C3
# 3. rank diagnostic   : at p=1, per-layer C4-vs-dense error == (M_hat - M) x
# 4. mask-vs-slice     : full-compute-then-mask == column-sliced skip

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch

from inference.configuration_llama import LlamaConfig
from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp

H, D, LAYERS, VOCAB = 64, 176, 2, 128


def build_model():
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

    # calibration on random tokens -> g_bar / g_bar_star buffers
    oracle_mlp.enable_stats_mode(model)
    with torch.no_grad():
        for _ in range(4):
            model(torch.randint(0, VOCAB, (2, 64)))
    oracle_mlp.finalize_stats(model)
    oracle_mlp.attach_col_norms(model)
    return model


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def test_top_p_mask_semantics():
    score = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    m = oracle_mlp.top_p_mask(score, 0.7)
    assert m.tolist() == [[True, True, False, False]], m
    m = oracle_mlp.top_p_mask(score, 1.0)
    assert m.all(), m
    print("PASS top-p mask semantics")


def test_1_p1_identity(model, ids):
    oracle_mlp.set_condition(model, "dense", 1.0)
    y_dense = logits(model, ids)
    for cond in ("c3", "c5"):
        oracle_mlp.set_condition(model, cond, 1.0)
        y = logits(model, ids)
        diff = (y - y_dense).abs().max().item()
        assert diff < 1e-3, f"{cond} p=1 vs dense: max diff {diff}"
        print(f"PASS p=1 identity ({cond}): max logit diff {diff:.2e}")
    return y_dense


def test_2_c4_full_rank(model, ids):
    oracle_mlp.set_condition(model, "c3", 0.85)
    y_c3 = logits(model, ids)
    oracle_mlp.attach_factors_inplace(model, rank=H)  # full rank
    oracle_mlp.set_condition(model, "c4", 0.85)
    y_c4 = logits(model, ids)
    assert torch.allclose(y_c4, y_c3, rtol=1e-3, atol=1e-3), \
        f"C4 full-rank vs C3: max diff {(y_c4 - y_c3).abs().max().item()}"
    print(f"PASS C4 full-rank == C3: max logit diff {(y_c4 - y_c3).abs().max().item():.2e}")


def test_3_rank_diagnostic(model):
    rank = H // 8
    oracle_mlp.attach_factors_inplace(model, rank=rank)
    torch.manual_seed(1)
    for layer_idx, mlp in oracle_mlp.iter_mlps(model):
        x = torch.randn(1, 16, H)
        mlp.oracle_condition, mlp.oracle_p, mlp.oracle_layer_dense = "dense", 1.0, False
        with torch.no_grad():
            y_dense = oracle_mlp.oracle_mlp_forward(mlp, x)
            mlp.oracle_condition = "c4"
            y_c4 = oracle_mlp.oracle_mlp_forward(mlp, x)
            # expected error at p=1: (M_hat - M) x
            w_down, w_up = mlp.down_proj.weight, mlp.up_proj.weight
            M = (w_down * mlp.oracle_g_bar.unsqueeze(0)) @ w_up
            M_hat = mlp.oracle_B @ mlp.oracle_A
            expected = x @ (M_hat - M).T
        err = (y_c4 - y_dense) - expected
        norm = (y_c4 - y_dense).norm().item()
        assert torch.isfinite(y_c4).all()
        assert err.abs().max().item() < 1e-3, \
            f"layer {layer_idx}: c4 p=1 error != (M_hat-M)x, resid {err.abs().max().item()}"
        print(f"PASS rank diagnostic layer {layer_idx}: r={rank}, "
              f"||(M_hat-M)x|| = {norm:.4e} (algebra residual {err.abs().max().item():.2e})")


def test_4_mask_vs_slice(model):
    _, mlp = next(oracle_mlp.iter_mlps(model))
    torch.manual_seed(2)
    x = torch.randn(1, 8, H)
    with torch.no_grad():
        u = mlp.up_proj(x)
        g = mlp.act_fn(mlp.gate_proj(x))
        i = (u * g).reshape(-1, D)
        mask = torch.rand(i.shape) > 0.5
        y_masked = (mask.float() * i) @ mlp.down_proj.weight.T
        y_sliced = torch.zeros_like(y_masked)
        for t in range(i.shape[0]):
            idx = mask[t].nonzero(as_tuple=True)[0]
            y_sliced[t] = i[t, idx] @ mlp.down_proj.weight[:, idx].T
    diff = (y_masked - y_sliced).abs().max().item()
    assert diff < 1e-4, f"mask vs slice: max diff {diff}"
    print(f"PASS mask-vs-slice equivalence: max diff {diff:.2e}")


if __name__ == "__main__":
    test_top_p_mask_semantics()
    model = build_model()
    ids = torch.randint(0, VOCAB, (2, 64), generator=torch.Generator().manual_seed(7))
    test_1_p1_identity(model, ids)
    test_2_c4_full_rank(model, ids)
    test_3_rank_diagnostic(model)
    test_4_mask_vs_slice(model)
    print("ALL ORACLE UNIT TESTS PASSED")
