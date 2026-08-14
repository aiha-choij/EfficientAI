"""GSLR stage-2.5, Lambda lookahead metric (topic: groupwise-flocking-tuning,
follow-up to gslr-stage2 / 5da111a).

Stage-2's diagnosis: independently retuning every layer with a layer-local
2nd-order approximation (the Gram matrix I^T I is a local Hessian) is only
valid for SMALL drift -- stage-2's l2,1 penalty forced large per-layer drift
and the model collapsed (PPL 8944-11182 vs untuned 30-60). Stage-2.5's fix
puts the group-mask STRUCTURE entirely in the (Delta w = 0) mask rule and
keeps weight changes at refit-level drift. Lambda is this design's residual-
stream "lookahead" importance weight: layer ell's down_proj output feeds
directly into the residual stream, which layer ell+1 then reads through its
own RMSNorm-gated attention (Wq/Wk/Wv) and MLP (W_gate/W_up) input
projections. A residual-stream dimension that many of those next-layer
matrices weight heavily is more consequential to perturb than one none of
them touch -- Lambda upweights the former in both the retuning objective and
the D_ell safety-budget measurement (gslr_layer_tune.measure_D), so a small
change in a "load-bearing" residual dimension counts for more than the same
change in a dimension the next layer mostly ignores.

Lambda_ell = diag(I + sum_{M in layer ell+1's residual-consuming input
projections} colsq(M @ diag(gamma))), where gamma is the RMSNorm weight
applied to the residual stream immediately before M consumes it:
  - input_layernorm.weight (gamma_attn) before self_attn.{q,k,v}_proj
  - post_attention_layernorm.weight (gamma_mlp) before mlp.{gate,up}_proj
colsq(A) is the per-INPUT-column sum of squares (A is (out, in) as an
nn.Linear weight always is in this codebase -- see gslr_layer_tune.py's
shape convention note) -- i.e. how much a unit perturbation of residual
dimension k is amplified into A's output, summed over A's output dims.

This is a DIAGONAL, first-order approximation of RMSNorm's true sensitivity:
RMSNorm normalizes by the whole vector's norm (a rank-1 coupling across all
h dims, not just a per-dim gamma scale), which this metric ignores by design
-- the stage-2.5 request's own spec defines Lambda_ell as this diagonal form
("Lambda_ell = diag(...)"), so the simplification is a request-mandated
choice, not an oversight. Recorded here explicitly since it's the kind of
thing a future reader would otherwise assume was missed.

The '+ I' term keeps Lambda >= 1 everywhere (a residual dimension is always
at least as important as "itself", i.e. the untouched no-lookahead case) so
D_ell (gslr_layer_tune.measure_D) never divides by/degenerates at a
lookahead-silent dimension.

Layer 30's lookahead is layer 31 (which exists in the base model and is
never excluded from masking, only from retuning -- gslr_group_ppl.py's
should_use_retuned). Layer 31 itself is never retuned (stage-2 spec, carried
over unchanged) so it never needs a Lambda of its own in this arm ladder.
"""

import torch


def lambda_from_mats(gamma_attn, gamma_mlp, Wq, Wk, Wv, Wgate, Wup):
    """Pure-tensor core (no model object needed) -- CPU-selftestable.
    gamma_attn, gamma_mlp: (h,). Wq/Wk/Wv/Wgate/Wup: (out, h) nn.Linear-style
    weights (any out dim, same in dim h). Returns Lambda (h,) fp32/64
    matching the input dtype family."""
    h = gamma_attn.shape[0]
    acc = torch.ones(h, dtype=torch.float64)
    for W, gamma in ((Wq, gamma_attn), (Wk, gamma_attn), (Wv, gamma_attn),
                     (Wgate, gamma_mlp), (Wup, gamma_mlp)):
        scaled = W.double() * gamma.double()[None, :]
        acc += scaled.pow(2).sum(0)
    return acc


def compute_layer_lambda(model, layer_idx, dtype=torch.float32):
    """Lambda_ell for retuning layer `layer_idx`, read from layer
    `layer_idx + 1`'s (frozen, original) weights. Caller must ensure
    layer_idx + 1 exists (stage-2.5's arm ladder only ever calls this for
    layer_idx in 0..30, and the base model has 32 layers, so 31 always
    exists)."""
    nxt = model.model.layers[layer_idx + 1]
    gamma_attn = nxt.input_layernorm.weight.detach().float().cpu()
    gamma_mlp = nxt.post_attention_layernorm.weight.detach().float().cpu()
    Wq = nxt.self_attn.q_proj.weight.detach().float().cpu()
    Wk = nxt.self_attn.k_proj.weight.detach().float().cpu()
    Wv = nxt.self_attn.v_proj.weight.detach().float().cpu()
    Wgate = nxt.mlp.gate_proj.weight.detach().float().cpu()
    Wup = nxt.mlp.up_proj.weight.detach().float().cpu()
    Lambda = lambda_from_mats(gamma_attn, gamma_mlp, Wq, Wk, Wv, Wgate, Wup)
    return Lambda.to(dtype)


def selftest():
    torch.manual_seed(0)
    h, dq, dm = 16, 20, 24

    # ---- test 1: all-zero next-layer weights -> Lambda == 1 everywhere
    # (the "+I" floor, nothing downstream reads this residual dim at all)
    gamma_attn = torch.randn(h)
    gamma_mlp = torch.randn(h)
    zeros_q = torch.zeros(dq, h)
    zeros_m = torch.zeros(dm, h)
    Lam0 = lambda_from_mats(gamma_attn, gamma_mlp, zeros_q, zeros_q, zeros_q, zeros_m, zeros_m)
    assert torch.allclose(Lam0, torch.ones(h, dtype=torch.float64)), "zero next-layer weights must give Lambda==1"
    print("selftest 1/3 OK (all-zero next-layer weights -> Lambda == 1, the +I floor)")

    # ---- test 2: scaling one matrix by c scales its colsq contribution by c^2
    Wq = torch.randn(dq, h)
    Wk = torch.randn(dq, h)
    Wv = torch.randn(dq, h)
    Wgate = torch.randn(dm, h)
    Wup = torch.randn(dm, h)
    Lam_a = lambda_from_mats(gamma_attn, gamma_mlp, Wq, Wk, Wv, Wgate, Wup)
    c = 3.0
    Lam_b = lambda_from_mats(gamma_attn, gamma_mlp, c * Wq, Wk, Wv, Wgate, Wup)
    expected_delta = (c ** 2 - 1) * (Wq.double() * gamma_attn.double()[None, :]).pow(2).sum(0)
    assert torch.allclose(Lam_b - Lam_a, expected_delta, rtol=1e-5, atol=1e-6), \
        "scaling Wq by c should scale its colsq contribution by c^2"
    print("selftest 2/3 OK (scaling one next-layer matrix by c scales its colsq contribution by c^2)")

    # ---- test 3: gamma==0 on one branch zeroes that branch's contribution
    # (RMSNorm gate closed -> that branch never reads the residual stream)
    gamma_mlp_zero = torch.zeros(h)
    Lam_c = lambda_from_mats(gamma_attn, gamma_mlp_zero, Wq, Wk, Wv, Wgate, Wup)
    attn_only = torch.ones(h, dtype=torch.float64)
    for W in (Wq, Wk, Wv):
        attn_only += (W.double() * gamma_attn.double()[None, :]).pow(2).sum(0)
    assert torch.allclose(Lam_c, attn_only, rtol=1e-5, atol=1e-6), \
        "gamma_mlp==0 should zero out the mlp branch's contribution to Lambda"
    print("selftest 3/3 OK (gamma_mlp==0 zeroes the mlp branch's contribution)")

    print("selftest ALL OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
