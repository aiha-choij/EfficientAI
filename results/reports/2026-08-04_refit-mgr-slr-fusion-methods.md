# Closed-Form Local Compensation for FFN Activation Sparsity: Refit, Mean-Gated Residuals, and Their Fusion

*Methods report, 2026-08-04. Models: LLaMA2-7B (all numbers here unless
stated), with the standalone-refit section also citing Llama-3.1-8B and
Llama-3.2-3B-instruct. Evaluation: WikiText-2 perplexity. Code:
`larosa/scripts/oracle/06_refit_fusion.py`, `larosa/inference/refit_mlp.py`;
experiment records under `.labtool/topics/{local-loss-refit,
oracle-residual-sparsity}/`.*

---

## 1. Setup and notation

A LLaMA-family feed-forward network (FFN) layer with hidden width
h = 4096 and intermediate width d = 11008 computes, for a token input
x ∈ R^h:

$$
u = W_u x, \qquad
g = \sigma(W_g x), \qquad
i = u \odot g, \qquad
y = W_d\, i,
$$

where W_u, W_g ∈ R^{d×h} (up and gate projections), W_d ∈ R^{h×d}
(output projection), σ is SiLU, and ⊙ is the elementwise product; i is
the *intermediate activation* whose coordinates we call neurons.

**Activation sparsity.** For a sparsity level s, each token keeps only
the K = ⌊(1−s)·d⌋ neurons with the largest score and zeroes the rest,
expressed by a binary mask m ∈ {0,1}^d with Σ_j m_j = K. Every method
below differs only in three choices: (i) the score that ranks neurons,
(ii) what replaces the contribution of dropped neurons, and (iii) which
weights are re-solved against calibration data.

**Metric.** Perplexity (PPL) on the WikiText-2 test set:

$$
\mathrm{PPL} \;=\; \exp\!\Big( \tfrac{1}{N} \sum_{t=1}^{N}
-\log p\big(w_t \,\big|\, w_{<t}\big) \Big),
$$

the exponential of the mean token negative log-likelihood over N tokens
(2048-token non-overlapping chunks). Lower is better; the dense
(unmodified) model is the floor. Trusted anchors: dense = 5.4736
(LLaMA2-7B), 6.2394 (Llama-3.1-8B), 11.0489 (Llama-3.2-3B-instruct).

**Compute accounting.** The dense FFN costs 3hd multiply-accumulates
per token (three d×h matrices). We report each method's cost as a
fraction of 3hd.

---

## 2. The refit method on its own

### 2.1 Where the idea comes from

Masking is a distribution shift, not just an error. The output weight
W_d was trained to act on the *dense* intermediate vector i; under
masking it receives z = m ⊙ i instead. Dropped neurons do not merely
add noise — they remove, in a *systematic, mask-correlated* way, part
of the signal W_d expects. This reframing suggests a classical remedy:
keep the architecture and the mask fixed, and re-solve W_d so that it
maps the *masked* input distribution back to the *dense* output. The
template is the local-reconstruction family of post-training
quantization (GPTQ / AdaRound / BRECQ), which repairs quantization
error by adjusting the remaining continuous parameters against a frozen
teacher, layer by layer, in closed form. The research plan
("Sparse-BRECQ") transplants that template from weight quantization to
activation sparsity.

Two design commitments make the transplant clean:

1. **Optimize only downstream of the mask.** The mask is computed from
   the ORIGINAL weights, with score s_j = |i_j|·‖W_d[:,j]‖₂ (activation
   magnitude times the column norm of the original output weight), and
   is never recomputed from refit weights. Consequences: (a) the
   training objective contains no top-K selection that would need
   differentiating — it is a pure quadratic with a global optimum in
   closed form; (b) no circularity — the refit weight cannot alter its
   own mask.
2. **Zero runtime cost.** The refit only *replaces the values* of W_d;
   shapes are unchanged, so inference cost is identical to the unrefit
   sparse model.

### 2.2 Derivation, step by step

**Step 1 — objective.** Let y* = W_d⁰ i denote the dense teacher output
(original weight, unmasked activation) and z = m ⊙ i the masked
activation actually available at deployment. We seek W̃ ∈ R^{h×d}
minimizing the expected reconstruction error

$$
J(\widetilde{W}) \;=\; \mathbb{E}_x \,
\big\lVert\, \widetilde{W} z(x) - y^{*}(x) \,\big\rVert_2^2 .
$$

**Step 2 — empirical form and why plain least squares is not enough.**
Over calibration tokens {x_t}, t = 1..N, the empirical objective is
Σ_t ‖W̃ z_t − y*_t‖². Setting its derivative to zero gives the normal
equations W̃G = C with the (uncentered) second-moment matrices

$$
G \;=\; \sum_t z_t z_t^{\top} \;\in\; \mathbb{R}^{d\times d},
\qquad
C \;=\; \sum_t y^{*}_t z_t^{\top} \;\in\; \mathbb{R}^{h\times d}.
$$

G is rank-deficient in practice: a neuron that is rarely kept by the
mask contributes almost nothing to G, so the corresponding columns of
W̃ are underdetermined and a bare pseudo-inverse lets them drift
arbitrarily — fitting calibration noise.

**Step 3 — anchored ridge.** We add a proximity prior that expresses
the correct default belief: *where the data gives no evidence, keep the
trained weight.* With D = mean(diag(G)) (a scalar that makes λ
dimensionless by matching the ridge to the Gram's own scale):

$$
\mathcal{L}(\widetilde{W}) \;=\;
\sum_t \big\lVert \widetilde{W} z_t - y^{*}_t \big\rVert^2
\;+\; \lambda D\, \big\lVert \widetilde{W} - W_d^0 \big\rVert_F^2 .
$$

**Step 4 — closed form.** Setting ∂L/∂W̃ = 2(W̃G − C) + 2λD(W̃ − W_d⁰)
= 0 gives

$$
\boxed{\;\widetilde{W} \;=\;
\big( C + \lambda D\, W_d^0 \big)\,
\big( G + \lambda D\, I \big)^{-1}\;}
$$

solved with one Cholesky factorization of the d×d matrix per layer.
Layers are solved **independently**: every layer's statistics are
collected in a single dense forward pass (so each layer sees the true
dense input distribution, and its teacher is the dense output of that
same layer). Total preparation cost: one calibration forward pass
(512×2048 tokens in our runs) plus 32 Cholesky solves — minutes on one
GPU. Runtime cost: zero.

**Step 5 — properties worth stating.**

- *Global optimum.* L is strictly convex in W̃ (since G + λD·I is
  positive definite), so the closed form is the exact minimizer — no
  optimization error to debate.
- *Harmlessness limit.* As λ → ∞, W̃ → W_d⁰: the method degrades to the
  identity operation, bounding the worst case near "no change".
- *In-sample monotonicity.* At the optimum, calibration reconstruction
  error can only decrease relative to W_d⁰ (used as a unit test).
- *Statistics are reusable.* G and C depend only on the mask rule, so a
  λ sweep costs one extra solve each, and the statistics are stored.

### 2.3 What was verified experimentally (mechanism-level tests)

- *Exact recovery*: at s = 0 (no masking) the solve returns W_d⁰ to
  numerical precision, and the sparse and dense streams agree to ~1e−8
  in a multi-layer unit test — the machinery itself is exact.
- *Anchor behavior*: with synthetic rank-deficient data, unevidenced
  columns stay at the original weight instead of collapsing to zero.

### 2.4 What the experiments established (result-level)

All PPL numbers below use the anchored solve of §2.2 (the anchor term
is part of the method definition; an earlier variant that shrank toward
zero instead of W_d⁰ is superseded).

**(a) Refit is a high-sparsity tool: large gains at s = 0.9, neutral at
s = 0.5 — on two models.** (per-token masks, g = 1)

| model | s | mask only | + refit | Δ |
|---|---|---|---|---|
| Llama-3.1-8B | 0.5 | 6.387 | 6.380 | −0.1% (neutral) |
| Llama-3.1-8B | 0.9 | 12.935 | **9.147** | **−29.3%** |
| 3B-instruct | 0.5 | 11.210 | 11.207 | −0.03% (neutral) |
| 3B-instruct | 0.9 | 21.586 | **16.162** | **−25.1%** |

Reading: at low sparsity masking removes little, so there is little
systematic bias to correct and the anchored solve correctly does almost
nothing; at high sparsity the bias is large and the linear correction
recovers a substantial share of it.

**(b) The result replicates on an independent metric.** On a 7-task
zero-shot accuracy suite (lm-eval), refit at s = 0.9 improved 5/7 tasks
with an average of +3.97 accuracy points (Llama-3.1-8B), matching the
PPL story; at s = 0.5 accuracy was ~neutral — PPL and accuracy agree.

**(c) Token-block-shared masks: refit absorbs a minority of the sharing
tax.** When g consecutive tokens share one mask (the kernel-efficient
regime), the mask-only PPL explodes with g (Llama-3.1-8B, s = 0.9:
12.9 → 269.3 from g = 1 to g = 128); refit absorbs 12–22% of that
increase in log-PPL terms — a buffer, not a fix, motivating the
compensation methods of §3–§5.

**(d) Negative control — sequential refit (GPTQ-style) loses.** Solving
layers in order, feeding each layer the *previous refit layers'*
outputs (more faithful to deployment), was strictly worse than the
independent solve (3B, s = 0.9: 26.3 vs 19.95, and quadrupling
calibration data did not close it). Mechanism: approximation errors
compound through the stack, while the independent solve re-anchors
every layer to the true dense input. The mechanism was verified correct
by an exact-recovery unit test, so this is a property, not a bug —
recorded as a dead end.

**(e) λ sensitivity is weak.** Over λ ∈ {0.001, 0.01, 0.1}: at s = 0.5
all results are within 0.08%; at s = 0.9, λ = 0.1 is best by ~0.8%. We
use λ = 0.1 subsequently.

---

## 3. Mean-gated residual (MGR): an exact identity, then a choice

Split the gate around its calibration mean ḡ_j = E[g_j]:

$$
i \;=\; u \odot g
\;=\; u \odot \bar{g} \;+\; u \odot (g - \bar{g})
\;\equiv\; \text{(mean part)} + r,
\qquad r \;\triangleq\; u \odot (g - \bar{g}).
$$

Apply W_d and use u ⊙ ḡ = diag(ḡ)·W_u x:

$$
y \;=\; W_d (u \odot \bar{g}) + W_d\, r
\;=\; M x + W_d\, r,
\qquad
M \;\triangleq\; W_d \cdot \mathrm{diag}(\bar{g}) \cdot W_u
\;\in\; \mathbb{R}^{h\times h}.
$$

This is an *identity*, not an approximation: the average contribution
of **all** neurons is a fixed linear map Mx of the input, and
everything token-specific lives in the residual r. MGR sparsifies only
the residual:

$$
\text{score}_j = |r_j|\cdot\lVert W_d[:,j]\rVert_2,
\qquad
\hat{y} \;=\; W_d (m \odot r) + M x .
$$

Dropped neurons lose only their *residual*; their mean contribution is
fully retained inside Mx. With M computed exactly this is the
diagnostic condition "C3": PPL 6.638 at s = 0.9 (vs 8.11 mask-only) —
but Mx costs h² per token, which defeats the purpose. A deployable form
must approximate M.

---

## 4. SLR: a sparse-plus-low-rank approximation of M

$$
M \;\approx\; B A + R,
\qquad
A \in \mathbb{R}^{256\times h},\;
B \in \mathbb{R}^{h\times 256}\ \text{(rank-256 truncated SVD)},\;
R = M - BA,
$$

$$
\hat{y} \;=\; W_d (m \odot r) \;+\; B (A x) \;+\; R\, (m_x \odot x),
$$

where m_x keeps, per token, the 1536 input channels with largest |x| —
the part of M the SVD failed to capture is computed exactly, but only
on the channels that carry most of the input's energy. Cost:
(2·256 + 1536)/(3d) ≈ +6.2 percentage points of the dense FFN.
Measured: PPL 6.942 at s = 0.9 (best deployable compensation prior to
refit).

A lesson that shapes everything after: an earlier attempt to improve
the factorization by whitening reduced its own least-squares objective
by 13% yet *worsened* PPL at every rank — offline input-space
objectives are not aligned with downstream loss. The constructive use
of that lesson is §5: fit the factors by *regression against the
output*, not by SVD energy.

---

## 5. Fusion: SLR + refit, derived

**Step 1 — observe linearity.** In the SLR output

$$
\hat{y} \;=\;
\underbrace{W_d (m \odot r)}_{\text{linear in } W_d}
\;+\;
\underbrace{B (A x)}_{\text{linear in } B}
\;+\;
\underbrace{R (m_x \odot x)}_{\text{fixed: } h(x)},
$$

freeze A (the input projection) and the sparse correction
h(x) ≜ R(m_x ⊙ x). Then ŷ is *jointly linear* in the pair (W_d, B).

**Step 2 — stack into one regression.** Define the feature vector and
stacked parameter

$$
\varphi(x) = \begin{bmatrix} m \odot r \\ A x \end{bmatrix}
\in \mathbb{R}^{d+256},
\qquad
\Theta = [\, W_d ,\; B \,] \in \mathbb{R}^{h\times(d+256)},
\qquad
\hat{y} = \Theta\, \varphi(x) + h(x).
$$

**Step 3 — apply the anchored-ridge template of §2.2.** Move the fixed
term to the target, t_t ≜ y*_t − h(x_t), anchor at the originals
Θ₀ = [W_d⁰, B⁰]:

$$
G = \sum_t \varphi_t \varphi_t^{\top}
\in \mathbb{R}^{(d+256)\times(d+256)},
\qquad
C = \sum_t t_t\, \varphi_t^{\top},
$$

$$
\boxed{\;[\,\widetilde{W}_d,\ \widetilde{B}\,]
\;=\; \big( C + \lambda D\, \Theta_0 \big)
\big( G + \lambda D\, I \big)^{-1}\;}
\qquad D = \mathrm{mean}(\mathrm{diag}(G)).
$$

**Step 4 — why this is well-posed.** The mask m (and m_x) is
gauge-fixed to the original weights, so it does not depend on Θ; the
loss L(Θ) is a pure quadratic and the solve is its global optimum; the
refit weights cannot feed back into selection. The three properties of
§2.2 carry over, plus: the top-left d×d block of G alone yields the
"refit W_d only" solution (R1) from the same statistics, at no extra
cost.

**Interpretation of the two blocks.** W̃_d is the output weight
re-optimized *knowing masked residuals are coming* (absorbing the
systematic truncation bias); B̃ is the low-rank compensation re-aimed
at the directions that matter for the output error, replacing the
Frobenius-optimal SVD directions (the §4 lesson, applied).

**Results (LLaMA2-7B, λ = 0.1, exact achieved sparsity):**

| s | R0 = SLR (no refit) | R1 (W̃_d only) | R2 (joint) |
|---|---|---|---|
| 0.5 | 5.599 | 5.560 | 5.559 |
| 0.7 | 5.750 | 5.655 | 5.650 |
| 0.9 | 6.780 | 6.208 | **6.195** |

At s = 0.9 the fusion passes all three pre-registered thresholds — the
unrefit SLR (6.94), the plain mask at s = 0.85 (6.709, the old
compute-frontier reference), and even the exact-compensation diagnostic
C3 (6.638): refit corrects a bias in W_d that *exact* mean-gate
compensation cannot, because C3 keeps the original W_d. Two honest
qualifiers: (i) the gain is dominated by W̃_d (R1 ≈ R2 — the joint
B-refit adds ~0.01 PPL); (ii) applying the same free refit to the plain
mask at s = 0.85 gives 6.183, statistically tying R2 — so in
compute-matched terms the compensation family and the plain-mask family
are at parity, and the fusion's clear win is under *pinned* sparsity
(s = 0.9), which is this research line's target regime.

**Where the remaining headroom is.** A follow-up arm that solves the
most general *linear* compensation (a full h×h map T by the same
template) measures the ceiling of any linear-in-x method at 6.150 —
only 0.045 below R2. Progress beyond that requires signals nonlinear in
x: adding token-wise sketch estimates of the dropped neurons'
activations as extra regression features reached 5.946 at s = 0.9
(breaking the linear ceiling; neutral at s = 0.7), at a compute cost
that makes sketch-rank reduction the immediate engineering question.
Those experiments are in progress and will be reported separately.

---

## 6. Summary table (LLaMA2-7B, s = 0.9, WikiText-2 PPL; dense 5.474)

| method | formula sketch | PPL | FFN compute |
|---|---|---|---|
| plain top-K mask | W_d(m⊙i) | 8.11 | 0.10 |
| MGR exact (diagnostic) | W_d(m⊙r) + Mx | 6.638 | not deployable |
| SLR | W_d(m⊙r) + B(Ax) + R(m_x⊙x) | 6.942* | 0.162 |
| SLR + refit (this report) | [W̃_d, B̃]φ + h(x) | **6.195** | 0.162 |
| linear ceiling (diagnostic) | [W̃_d, T̃]·[m⊙r; x] | 6.150 | not deployable |
| + token-wise sketch features | + W̃_tail·(1−m)⊙(ĝ⊙û) | 5.946 | ~0.80 (unoptimized) |
| plain mask s=0.85 + refit (control) | W̃_d(m⊙i) | 6.183 | 0.15 |

\*6.942 is the original-session measurement; the fusion pipeline's own
re-measurement of the same arm is 6.780 (ḡ calibration sample differs);
all fusion comparisons are within one pipeline.
