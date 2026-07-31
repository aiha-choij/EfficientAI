# local-loss-refit — isolated effect of refitting W_down to a frozen mask

## Status
active

## Notation
Same FFN notation as oracle-residual-sparsity: i = u ⊙ g, u = up_proj(x),
g = act_fn(gate_proj(x)), y = down_proj(i). col_norm[j] = ‖W_down[:,j]‖₂.
C2-score = |i_j| · col_norm[j] (weight-aware, same as oracle-residual-sparsity's
C2 condition). Token block g: mask-sharing group of g consecutive tokens
within one sequence (never spans a sequence boundary; last block ragged;
g=1 = per-token). Refit: closed-form ridge regression fitting masked input
z = m ⊙ i to the DENSE teacher output y*, per layer, in fp32:
  G = Σ z zᵀ, C = Σ y* zᵀ, W̃_down = C (G + λ·mean(diag(G))·I)⁻¹
Not gradient training; weight shape is unchanged so runtime cost is 0.

## Hypothesis
After masking FFN intermediate activations with a fixed top-K rule (C2-score,
matches oracle-residual-sparsity's C2), how much of the accuracy loss does a
plain closed-form refit of down_proj alone recover — with NO other repair
mechanism (no mean-gate residual, no rank-r branch, no gate-sketch, no
Fisher weighting, no mask jitter)? This isolates the refit effect from every
other compensation idea already explored in oracle-residual-sparsity.

Ladder:
- L0: top-K(C2-score) mask, original W_down, dense input. (= oracle-residual-
  sparsity's C1-select topk / C2 condition — same mask rule, weight-aware.)
- L1: L0's mask (frozen, from ORIGINAL weight — never recomputed from the
  refit weight, to block circularity), W̃_down solved per layer independently
  from ONE dense forward pass over calibration (every layer sees dense x).
- L2: same mask rule but score recomputed against the SPARSE stream's actual
  activations; W̃_down solved layer-by-layer in sequence (GPTQ-style single
  sweep) — layer ℓ's calibration input is the OUTPUT of layers < ℓ already
  refit; layer ℓ's teacher y* is always the ORIGINAL dense model's output at
  that layer. Attention is dense and unmodified in both streams.

Primary readout: ΔL1 = L1 − L0, ΔL2 = L2 − L1 at matched (s, g), as
accuracy(%p) or ΔPPL. Go if ΔL1 at s=0.9, g=1 is clear (e.g. accuracy +3%p
or a PPL improvement well outside the ~1e-3 backend-noise floor established
in oracle-residual-sparsity's Phase-3 gate). A null result on ΔL1 is
informative too: it would say masking error is mostly token-idiosyncratic,
not a fixed linear bias down_proj can absorb — motivating the compensation-
branch line (oracle-residual-sparsity) rather than refit alone.

## Design decisions
- **Base infra reused as-is (2026-07-31)**: `larosa/inference/oracle_mlp.py`
  (top_k_mask, top_count_mask, iter_mlps, attach_col_norms) and
  `larosa/inference/modeling_llama_larosa.py`'s `sparse_mode` dispatch
  (mirrors the existing `oracle` mode wiring) — new sparse_mode `refit`
  added alongside it, oracle_mlp.py itself left untouched (oracle-residual-
  sparsity's active job must not be disturbed).
- **Scope deliberately narrow (from the request spec)**: no mean-gate
  residual, no rank-r compensation branch, no Fisher weighting, no mask
  jitter — this topic measures refit ALONE. Those ideas live in
  oracle-residual-sparsity; do not import them here even if tempting.
- Mask score always computed from the ORIGINAL (never refit) down_proj
  weight's column norms, in both L1 and L2 — refit must never feed back
  into the score (spec's explicit anti-circularity rule).
- Model plan: Llama-3.2-3B for dev/first-pass, Llama-3.1-8B for the main
  matrix — both already present locally at /raid/LLM (Llama-3.2-3B: only
  the instruct variant `llama3.2-3b-instruct` is on disk, not a plain
  pretrained checkpoint; decide before submitting whether that's
  acceptable for the dev pass or whether to pull the base checkpoint).
  Fallback anchor if a model is unavailable: LLaMA2-7B (`/raid/LLM/llama2-7b`)
  — trusted dense/C1 PPL anchors from oracle-residual-sparsity's Phase-3
  gate: dense 5.4736 (gateway A100) / 5.4738 (a6000-2); C1
  5.5210/5.7296/8.1083 (gateway) at s=0.5/0.7/0.9, wikitext-2.

## Key Findings
(none yet — implementation in progress)

## Dead Ends
(none yet)

## Open Questions
- L2's memory footprint: caching the full-calibration hidden-state tensor
  between sequential layer steps is O(N·h) per stream; needs a
  streaming/chunked implementation (planned: host-RAM bf16 cache, GPU only
  holds one chunk at a time) rather than holding all layers' activations at
  once (that would be O(L·N·h), too large).
- Whether `llama3.2-3b-instruct` (only 3.2-3B checkpoint on disk) is an
  acceptable dev-pass stand-in for a plain pretrained 3B, given the spec
  asked for the base model.
- lm-eval-harness availability on the gateway env — unconfirmed; PPL-first,
  harness as a follow-up if missing (per request's fallback clause).

## Next Experiments
1. Unit tests (CPU, tiny random model, spec's 5 required checks) — must
   pass before any GPU submission.
2. Single verification point: g=1, s=0.9, L0 vs L1, small model, PPL only —
   confirm the effect exists before the full matrix.
3. Full matrix: s ∈ {0.5, 0.7, 0.9} × g ∈ {1, 8, 32, 128} × {L0, L1, L2},
   cost-reduction ladder from the request spec if GPU time is short
   (restrict g ∈ {8,128} to s=0.9 first; L2 to g ∈ {1,32} first).

## Active Jobs
- (none yet)

## Pointers
- Request spec (self-contained, inlined the host wiki doc):
  `/home/choij/workspace/requests/active/20260731-090908-local-loss-refit/`
  (report.md tracks session-to-session judgment/rationale).
- Base infra: oracle-residual-sparsity topic — `gist.md` (C2 condition,
  anchors, sparse_mode wiring precedent), `larosa/inference/oracle_mlp.py`,
  `larosa/scripts/oracle/{01_calibrate,03_build_M,04_eval_ppl}.py`,
  `larosa/tests/test_oracle_units.py` (unit-test convention: CPU, tiny
  random LlamaForCausalLM, fp32).
- New code lands under `larosa/inference/refit_mlp.py`,
  `larosa/scripts/refit/`, `larosa/tests/test_refit_units.py` — all on
  branch `auto/local-loss-refit` (this topic's commits, and all code
  changes for this request, stay off `main`; proposed via `agent-pr`).
- Models: `/raid/LLM/Llama-3.1-8B`, `/raid/LLM/llama3.2-3b-instruct`,
  `/raid/LLM/llama2-7b` (fallback), all local — no download needed so far.
