# coding=utf-8
# Unit tests for Local Loss Refit (L0/L1 masking + closed-form refit).
# Runs on CPU with a tiny random LLaMA in fp32 (same discipline as
# test_oracle_units.py). Run directly:
#   python tests/test_refit_units.py
#
# Required checks (request spec section "필수 unit test"):
#   1. s=0 restoration: mask all-keep + lambda=1e-6 -> W_tilde ~= W_down
#   2. g=1 block code path == top_k_mask directly (mask + output)
#   3. calibration reconstruction MSE: L1 <= L0
#   4. block-rule asserts: mask constant within a block, no cross-sequence
#      bleed, padding excluded from aggregation
#   5. seed + calibration token list save/reuse (reproducibility)

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch

from inference.configuration_llama import LlamaConfig
from inference.modeling_llama_larosa import LlamaForCausalLM
from inference import oracle_mlp, refit_mlp

H, D, LAYERS, VOCAB = 32, 48, 2, 64


def build_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=H, intermediate_size=D,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, use_cache=False,
    )
    config._attn_implementation = "eager"
    config.sparse_mode = "refit"
    config.refit_mode = "l0"
    model = LlamaForCausalLM(config).float().eval()
    refit_mlp.attach_col_norms(model)
    return model


def test_1_s0_restoration(model):
    # enough tokens (>> D) for G to be well-conditioned so ridge with a
    # near-zero lambda recovers W_down almost exactly
    refit_mlp.enable_l1_collect_mode(model, s=0.0, g=1)
    torch.manual_seed(11)
    with torch.no_grad():
        for _ in range(16):
            model(torch.randint(0, VOCAB, (4, 64)))
    stats = refit_mlp.finalize_l1(model)
    for layer_idx, mlp in oracle_mlp.iter_mlps(model):
        st = stats[layer_idx]
        w_tilde = refit_mlp.solve_refit(st["G"], st["C"], lam=1e-6)
        w_orig = mlp.down_proj.weight.detach()
        diff = (w_tilde - w_orig).abs()
        rel = diff.max().item() / w_orig.abs().max().item()
        assert torch.allclose(w_tilde, w_orig, rtol=1e-2, atol=1e-2), \
            f"layer {layer_idx}: s=0 refit vs W_down max rel diff {rel:.4e}"
        print(f"PASS s=0 restoration layer {layer_idx}: max rel diff {rel:.2e}")


def test_2_g1_equals_topk(model):
    torch.manual_seed(12)
    score = torch.rand(3, 20, D)
    for s in (0.3, 0.5, 0.8):
        m_direct = oracle_mlp.top_k_mask(score, s)
        m_block = refit_mlp.block_mask(score, s, g=1)
        assert torch.equal(m_direct, m_block), f"s={s}: g=1 block path != top_k_mask"
    # forward-level check: refit_mlp_forward with g=1 reproduces manual C2 mask
    _, mlp = next(oracle_mlp.iter_mlps(model))
    refit_mlp.set_condition(model, "l0", s=0.6, g=1)
    x = torch.randn(2, 16, H)
    with torch.no_grad():
        y = refit_mlp.refit_mlp_forward(mlp, x)
        u = mlp.up_proj(x)
        gg = mlp.act_fn(mlp.gate_proj(x))
        i = u * gg
        score2 = i.abs().float() * mlp.refit_col_norm
        m = oracle_mlp.top_k_mask(score2, 0.6)
        y_ref = mlp.down_proj(m.to(i.dtype) * i)
    assert torch.equal(y, y_ref), "refit_mlp_forward g=1 != direct top_k_mask computation"
    print("PASS g=1 block code path == top_k_mask (mask + forward output)")


def test_3_l1_beats_l0_insample(model):
    s, g = 0.5, 1
    refit_mlp.enable_l1_collect_mode(model, s=s, g=g)
    torch.manual_seed(13)
    batches = [torch.randint(0, VOCAB, (4, 64)) for _ in range(16)]
    with torch.no_grad():
        for b in batches:
            model(b)
    stats = refit_mlp.finalize_l1(model)
    for layer_idx, mlp in oracle_mlp.iter_mlps(model):
        st = stats[layer_idx]
        w_tilde = refit_mlp.solve_refit(st["G"], st["C"], lam=0.01)
        # recompute z, y* for the same calibration set to score in-sample MSE
        mse_l0, mse_l1, n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in batches:
                # run a dense forward up to capture this layer's (x -> mlp input)
                # cheaply: reuse the model's embed+layers up to `layer_idx` via a
                # hook-free approach is overkill for a unit test; instead verify
                # algebraically using the accumulated G/C directly (closed form
                # in-sample MSE = sum||y*||^2 - 2 tr(W C^T) + tr(W G W^T)).
                pass
        G, C, n = st["G"], st["C"], st["n"]
        w_orig = mlp.down_proj.weight.detach().float()

        def in_sample_sse(W):
            # sum_t ||y*_t - W z_t||^2 = sum||y*||^2 - 2 tr(W C^T) + tr(W G W^T)
            # sum||y*_t||^2 isn't tracked directly, but it's identical for both
            # W choices, so compare the W-dependent part only:
            return (-2.0 * torch.sum(W * C) + torch.sum((W @ G) * W)).item()

        sse_l0 = in_sample_sse(w_orig)
        sse_l1 = in_sample_sse(w_tilde)
        assert sse_l1 <= sse_l0 + 1e-3, \
            f"layer {layer_idx}: L1 in-sample SSE {sse_l1:.4f} > L0 {sse_l0:.4f}"
        print(f"PASS L1 <= L0 in-sample (layer {layer_idx}): "
              f"L1 W-term {sse_l1:.4f} <= L0 W-term {sse_l0:.4f}")


def test_4_block_rules():
    torch.manual_seed(14)
    B, T, Dl = 2, 20, 24
    score = torch.rand(B, T, Dl)
    g = 8
    m = refit_mlp.block_mask(score, 0.5, g)
    bid = refit_mlp.block_ids(T, g)
    # (a) mask constant within each block
    for b in range(B):
        for blk in bid.unique():
            rows = m[b, bid == blk]
            assert torch.all(rows == rows[0]), f"batch {b} block {blk}: mask not constant"
    # (b) last block ragged (T=20, g=8 -> blocks of size 8,8,4)
    assert bid.max().item() + 1 == 3 and (bid == 2).sum().item() == 4
    # (c) no cross-sequence bleed: perturbing batch row 1's score must not
    # change batch row 0's mask
    score2 = score.clone()
    score2[1] = torch.rand(T, Dl) * 100
    m2 = refit_mlp.block_mask(score2, 0.5, g)
    assert torch.equal(m[0], m2[0]), "batch row 0 mask changed when only row 1 perturbed"
    # (d) padding excluded from aggregation: inflate one padded position's
    # score to a huge value and confirm it does not change the block's mask
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[:, -1] = False  # last position of each row is "padding"
    score3 = score.clone()
    score3[:, -1, :] = 1e6  # huge score, but should be excluded
    m_nopad = refit_mlp.block_mask(score, 0.5, g)  # no padding, baseline (last pos real)
    m_withpad = refit_mlp.block_mask(score3, 0.5, g, seq_mask=seq_mask)
    # the block containing the padded position (last block) should match the
    # baseline mask computed WITHOUT the huge padded value at all (since it's
    # excluded from the sum either way -- compare against score with that
    # position zeroed instead of huge, which is what exclusion should equal)
    score_excl = score.clone()
    score_excl[:, -1, :] = 0.0
    m_excl_ref = refit_mlp.block_mask(score_excl, 0.5, g)
    assert torch.equal(m_withpad, m_excl_ref), \
        "padded position's inflated score leaked into the block aggregation"
    print("PASS block rules: constant-within-block, ragged last block, "
          "no cross-sequence bleed, padding excluded")


def test_5_calib_reproducibility(tmp_path_str):
    nsamples, seqlen, seed = 4, 16, 123
    g1 = torch.Generator().manual_seed(seed)
    buf1 = torch.randint(0, VOCAB, (nsamples * seqlen,), generator=g1).tolist()
    tokens1 = refit_mlp.reshape_calib_tokens(buf1, nsamples, seqlen)

    g2 = torch.Generator().manual_seed(seed)
    buf2 = torch.randint(0, VOCAB, (nsamples * seqlen,), generator=g2).tolist()
    tokens2 = refit_mlp.reshape_calib_tokens(buf2, nsamples, seqlen)
    assert torch.equal(tokens1, tokens2), "same seed did not reproduce the same calib tokens"

    path = os.path.join(tmp_path_str, "calib_tokens.pt")
    refit_mlp.save_calib_tokens(tokens1, path)
    reloaded = refit_mlp.load_calib_tokens(path, nsamples, seqlen)
    assert torch.equal(tokens1, reloaded), "save/load round trip changed calib tokens"
    try:
        refit_mlp.load_calib_tokens(path, nsamples, seqlen + 1)
        assert False, "shape mismatch should have raised"
    except AssertionError as e:
        assert "!=" in str(e)
    print("PASS calibration token reproducibility: seed-determinism + save/load round trip")


if __name__ == "__main__":
    import tempfile

    model = build_model()
    test_1_s0_restoration(model)
    test_2_g1_equals_topk(model)
    test_3_l1_beats_l0_insample(build_model())
    test_4_block_rules()
    with tempfile.TemporaryDirectory() as td:
        test_5_calib_reproducibility(td)
    print("ALL REFIT UNIT TESTS PASSED")
