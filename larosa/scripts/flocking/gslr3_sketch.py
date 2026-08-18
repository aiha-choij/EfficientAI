"""GSLR stage-3 compensation-branch feature builders (topic: groupwise-
flocking-tuning, follow-up to gslr-stage2.5 / 69f4f63). Question: does
jointly fitting a "what the dropped neurons would have contributed" branch
on top of B1D (stage-2.5's only surviving arm -- W_down-only Lambda-weighted
group refit, W_up/W_gate frozen, Dead End: never touch them) close more of
the gap to the per-token anchor? See gslr3_tune.py for the joint solve
driver and the arm ladder (C0/C1/C2/C2a); this module only builds the phi
feature (and, for C2, its sketch factors).

  C1 phi_t = P_r (x_t - mu)                    -- PCA of calibration x
  C2 phi_t = A_d [(1-m_T) * (ghat_t * uhat_t)]  -- sketch-tail feature
  C2a       same as C2 but uhat_t = the EXACT up-projection (u_t), not the
            sketch -- diagnostic only (not deployable): isolates how much of
            C2's gap-closing is capped by the up-sketch's own quality.

svd_lr is ported from the EfficientAI-verify clone (read-only reference
checkout), larosa/scripts/oracle/06_refit_fusion.py:svd_lr (repo HEAD at
c215ce3 "fusion r5: r4lr..." when read for this port) -- byte-identical
formula to block_comp_mlp.py:_svd_factors (auto/block-sparse-compensation
branch, this repo) modulo variable names; this module reuses the simpler
(A,B)-only signature since C1/C2 don't need block_comp's per-condition
factor bookkeeping (save/load, multiple conditions sharing one sketch dict).
"""

import torch


def svd_lr(W, rank):
    """rank-r truncated SVD of W [out,in] (nn.Linear weight layout).
    Returns A [r,in], B [out,r] (fp32, W's device) with B @ A ~ W; exact
    (B @ A == W) once rank >= true rank(W) = min(out,in) -- see
    gslr3_tune.py's unit test 3 (r_sk=d + lam->0 => C2 tail == exact tail),
    which relies on this exactness, not an approximation, once r_sk clamps
    to the full min(out,in) rank."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    A = sq.unsqueeze(1) * Vh[:r, :]
    B = U[:, :r] * sq.unsqueeze(0)
    return A.contiguous(), B.contiguous()


def compute_pca_basis(X, r, chunk=8192):
    """Top-r principal directions of X (N,h), calibration activations.
    Computed via the (h x h) covariance's eigendecomposition rather than an
    SVD of the full (N x h) matrix -- N is O(1e5) tokens at this repo's
    calibration scale, h=4096, so eigh on a 4096x4096 covariance is far
    cheaper than an SVD of the tall matrix while giving the identical
    principal directions (same top-r eigenvectors of X_centered^T X_centered
    either way). mu, the calibration mean, is frozen the same way colnorm0/
    Lambda always are (anti-circularity: eval-time phi must use the SAME
    mu/P the arm was fit with -- both are persisted in the arm's layer_i.pt,
    see gslr3_tune.py).
    Returns (mu (h,), P (r_eff,h)) with r_eff = min(r,h), P's rows
    orthonormal; phi_t = P @ (x_t - mu)."""
    h = X.shape[1]
    mu = X.mean(0)
    cov = torch.zeros(h, h, dtype=torch.float32, device=X.device)
    for s in range(0, X.shape[0], chunk):
        Xc = X[s:s + chunk] - mu[None, :]
        cov += Xc.T @ Xc
    evals, evecs = torch.linalg.eigh(cov)          # ascending
    r_eff = min(r, h)
    idx = torch.argsort(evals, descending=True)[:r_eff]
    P = evecs[:, idx].T.contiguous()               # r_eff x h
    return mu, P


def apply_pca(X, mu, P):
    """phi = (X - mu) @ P.T, chunk-free (P is small, r x h)."""
    return (X - mu[None, :]) @ P.T


def build_c2_sketch(Wg0, Wu0, Wd0, r_sk):
    """Wg0/Wu0 [d,h] (gate_proj/up_proj weight layout), Wd0 [h,d]
    (down_proj). Returns (Ag,Bg,Au,Bu,Ad,Bd), all fp32 on Wg0's device.
    A_d [r_sk,d] is the rank-r_sk factor of Wd0 -- this is what the stage-3
    request's "A_d ... rank-r_sk projection" notation applies to a d-dim
    tail vector as phi = tail @ Ad.T (== A_d @ tail per-token). d < the
    request text's literal "좌특이(left-singular)" phrasing would suggest
    for a (h,d)-shaped Wd0 (whose LEFT singular vectors live in the h-dim
    output space, not the d-dim neuron space phi needs to land in) -- this
    is implemented as the RIGHT-singular-vector-derived factor instead
    (same convention as block_comp_mlp.py's _svd_factors and
    06_refit_fusion.py's svd_lr, both of which this stage explicitly builds
    on), because that is the only choice for which unit test 3 (r_sk=d,
    lam->0 => C2 tail recovers the EXACT dropped-neuron contribution) can
    hold: at r_sk >= rank(Wd0) = h (Wd0 is h x d with h<d for LLaMA2-7B,
    so its true rank is h, not d -- see svd_lr's docstring), Ad/Bd become an
    EXACT factorization (Bd @ Ad == Wd0 exactly, not just a good
    approximation), so Theta=Bd recovers Wd0's true action on ANY tail
    vector, matched-token-for-token, not merely on some subspace. The
    request's "좌특이" wording is treated as a terminology looseness rather
    than a literal spec to follow off a cliff; documented here so a future
    reader doesn't wonder whether this was missed."""
    Ag, Bg = svd_lr(Wg0, r_sk)
    Au, Bu = svd_lr(Wu0, r_sk)
    Ad, Bd = svd_lr(Wd0, r_sk)
    return Ag, Bg, Au, Bu, Ad, Bd


def compute_c2_phi(X, mask, act_fn, Ag, Bg, Au, Bu, Ad, chunk=8192, exact_u=None):
    """phi_t = A_d @ tail_t, tail_t = (1-mask_t) * (ghat_t * uhat_t).
    X (N,h), mask (N,d) bool (True = kept, matches compute_group_mask's
    token_mask convention). exact_u, optional (N,d): substitutes the EXACT
    up-projection (x @ Wu0.T) for the sketch uhat -- C2a's diagnostic-only
    "isolate the up-sketch's quality ceiling" arm (ghat stays sketched,
    per the stage-3 request: "C2에서 u-hat를 정확값으로", only u, not g)."""
    N = X.shape[0]
    r_sk = Ad.shape[0]
    out = torch.empty(N, r_sk, dtype=torch.float32, device=X.device)
    for s in range(0, N, chunk):
        Xc = X[s:s + chunk]
        ghat = act_fn((Xc @ Ag.T) @ Bg.T)
        uhat = exact_u[s:s + chunk] if exact_u is not None else (Xc @ Au.T) @ Bu.T
        tail = (1.0 - mask[s:s + chunk].float()) * (ghat * uhat)
        out[s:s + chunk] = tail @ Ad.T
    return out


# ---------------------------------------------------------------- selftest

def selftest():
    torch.manual_seed(0)

    # svd_lr: exact reconstruction once rank >= true rank(W)
    out, in_ = 10, 6
    W = torch.randn(out, in_)
    A, B = svd_lr(W, min(out, in_))
    assert (B @ A - W).norm() / W.norm() < 1e-5, "svd_lr must be exact at full rank"
    A2, B2 = svd_lr(W, 2)
    err_full = (B @ A - W).norm()
    err_trunc = (B2 @ A2 - W).norm()
    assert err_trunc > err_full + 1e-6, "truncated sketch must be strictly worse than full-rank"
    print("selftest 1/3 OK (svd_lr: exact at full rank, strictly lossy when truncated)")

    # PCA: top-1 direction on a rank-1-dominated cloud must match the known axis
    h, N = 8, 4000
    axis = torch.randn(h)
    axis = axis / axis.norm()
    X = 5.0 * torch.randn(N, 1) @ axis[None, :] + 0.01 * torch.randn(N, h) + torch.randn(h)[None, :]
    mu, P = compute_pca_basis(X, 1)
    cos = (P[0] @ axis).abs().item()
    assert cos > 0.99, f"PCA top-1 direction should recover the dominant axis, cos={cos:.4f}"
    phi = apply_pca(X, mu, P)
    assert phi.shape == (N, 1)
    assert torch.allclose(mu, X.mean(0), atol=1e-4)
    print(f"selftest 2/3 OK (PCA top-1 direction recovers dominant synthetic axis, cos={cos:.4f})")

    # C2 phi: exact_u=None vs exact_u=the true up-projection must differ
    # whenever the up-sketch is lossy, and phi must be exactly 0 wherever
    # mask==1 everywhere (fully-kept group -> no tail to compensate).
    d, r_sk = 12, 3
    Wg0, Wu0, Wd0 = torch.randn(d, h), torch.randn(d, h), torch.randn(h, d)
    Ag, Bg, Au, Bu, Ad, Bd = build_c2_sketch(Wg0, Wu0, Wd0, r_sk)
    Xc = torch.randn(20, h)
    f = torch.nn.functional.silu
    mask_all_kept = torch.ones(20, d, dtype=torch.bool)
    phi_kept = compute_c2_phi(Xc, mask_all_kept, f, Ag, Bg, Au, Bu, Ad)
    assert torch.allclose(phi_kept, torch.zeros_like(phi_kept)), \
        "phi must be exactly 0 when nothing is masked out (empty tail)"
    mask_half = torch.zeros(20, d, dtype=torch.bool)
    mask_half[:, : d // 2] = True
    u_exact = Xc @ Wu0.T
    phi_sketch = compute_c2_phi(Xc, mask_half, f, Ag, Bg, Au, Bu, Ad)
    phi_exact = compute_c2_phi(Xc, mask_half, f, Ag, Bg, Au, Bu, Ad, exact_u=u_exact)
    assert (phi_sketch - phi_exact).norm() > 1e-6, \
        "C2a (exact u) must differ from C2 (sketched u) when the up-sketch is lossy"
    # r_sk = full rank -> uhat sketch becomes exact -> phi_sketch == phi_exact
    Ag2, Bg2, Au2, Bu2, Ad2, Bd2 = build_c2_sketch(Wg0, Wu0, Wd0, h)  # clamps to min(d,h)
    phi_sketch_full = compute_c2_phi(Xc, mask_half, f, Ag2, Bg2, Au2, Bu2, Ad2)
    phi_exact_full = compute_c2_phi(Xc, mask_half, f, Ag2, Bg2, Au2, Bu2, Ad2, exact_u=u_exact)
    relerr = (phi_sketch_full - phi_exact_full).norm() / phi_exact_full.norm().clamp(min=1e-12)
    assert relerr < 1e-3, f"at full-rank sketch, C2 and C2a phi must coincide: relerr={relerr:.2e}"
    print("selftest 3/3 OK (C2 phi: 0 when nothing masked, C2a != C2 when sketch lossy, "
          "C2a == C2 at full-rank sketch)")

    print("selftest ALL OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
