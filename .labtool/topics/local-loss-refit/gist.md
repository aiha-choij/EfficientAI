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
- **[MAIN, TOPIC-CLOSING] Both halves of the core question CONFIRMED with
  two independent metrics (2026-07-31).** Llama-3.1-8B, lm-eval-harness
  (7 tasks, piqa dropped -- see Open Questions), g=1:
  - **s=0.9: Go, on both metrics.** PPL ΔL1 -20.1% relative; accuracy
    L1 beats L0 on 5/7 tasks (arc_easy +6.0%p, boolq +12.5%p, winogrande
    +3.3%p, sciq +3.7%p, lambada_openai +4.0%p; ties/slight-loss on
    arc_challenge -1.2%p, hellaswag -0.5%p), **average Δacc +3.97%p** --
    meets the request's own stated Go example ("accuracy +3%p") almost
    exactly.
  - **s=0.5: confirmed hurts, on both metrics.** PPL ΔL1 +13.7% relative;
    accuracy L1 loses to L0 on 6/7 tasks, **average Δacc -1.19%p** (only
    boolq improves, +1.6%p -- interesting that boolq is also the single
    biggest s=0.9 winner, an asymmetry not investigated further).
  This closes the loop the PPL-only matrix (3B+8B, 8 points each) couldn't
  close alone: the request's PRIMARY judgment metric is accuracy, and it
  now tells the same story as PPL on the main model. **The request's core
  question is answered: refit alone recovers a real, large share of
  masking's accuracy cost, but ONLY in the high-sparsity/coarse-block
  regime -- it is not a universal fix and actively hurts outside that
  regime.** No further harness runs are required to answer the stated
  Go/No-go question; remaining extensions (full g-sweep on harness, a
  true dense s=0 baseline for framing) are backlog, not blockers.
  Journal: 2026-07-31_experiment-refit-harness-8b.md
- **[MAIN] L0 vs L1 single-point verification (2026-07-31): GO.**
  llama3.2-3b-instruct (dev-pass stand-in, see Open Questions), s=0.9, g=1,
  wikitext-2 PPL: L0 21.5859 -> L1 19.9510, ΔL1 = **−1.635 PPL (−7.6%)**,
  same achieved sparsity (0.9000, mask identical by construction). Far
  outside any plausible noise floor. Closed-form down_proj refit alone,
  with NO other repair mechanism, recovers a real share of masking's cost.
  Proceeds to the reduced-cost matrix (rung 1 of the cost-reduction ladder)
  on both the dev model and the main model (Llama-3.1-8B, confirmed a real
  base checkpoint locally, no LLaMA2-7B fallback needed).
  When relevant: (a) L1 alone is now confirmed non-trivial — the open
  question shifts to how the effect scales with g (sharing tax) and
  whether L2 buys more on top of L1; (b) absolute PPL (19-22) is NOT
  comparable to the LLaMA2-7B anchors (5.5-8.1) — different model/scale/
  tuning/calibration corpus; only ΔL1 at matched (s,g) is a valid
  cross-run comparison.
  Journal: 2026-07-31_experiment-refit-l0l1-validate-3b.md
- **[MAIN, CONFIRMED on 2 models] L1 refit is a HIGH-SPARSITY tool, not a
  uniform fix — it HURTS at s=0.5/0.7 and helps big at s=0.9, growing
  strongly with g.** Reduced-cost matrix (2026-07-31) on BOTH
  llama3.2-3b-instruct (dev) and Llama-3.1-8B (main, base pretrained), same
  code/config, wikitext-2 PPL (calib wikitext103):

  | s | g | 3B ΔL1 (rel) | 8B ΔL1 (rel) |
  |---|---|---|---|
  | 0.5 | 1 | +18.3% | +13.7% |
  | 0.7 | 1 | +18.7% | +10.7% |
  | 0.9 | 1 | −7.6% | −20.1% |
  | 0.5 | 32 | +26.3% | +12.7% |
  | 0.7 | 32 | +18.7% | −0.8% |
  | 0.9 | 32 | −33.2% | −51.2% |
  | 0.9 | 8 | −34.1% | −42.3% |
  | 0.9 | 128 | −35.0% | −58.4% |

  Replicated cleanly across two different models/scales/tunings (instruct
  3B vs base 8B) — **not a dev-model or corpus-mismatch artifact**. At
  s=0.9, refit absorbs an increasing share of the g-driven "sharing tax" as
  g grows (3B: ~37% of the g=1→128 PPL increase; 8B: ~60%, i.e. the effect
  is STRONGER on the larger base model). At s=0.5 it consistently hurts on
  both models; s=0.7 is the crossover zone (still negative at g=1 on both,
  crosses to roughly neutral at g=32 on 8B but stays negative on 3B — exact
  crossover point looks model-dependent).
  Leading hypothesis (bias-variance tradeoff, unconfirmed mechanism but the
  pattern itself IS confirmed): at low s the mask removes little, so there
  is very little true systematic bias for a linear refit to correct: the
  closed-form fit still strictly reduces IN-SAMPLE calibration loss (unit
  test 3's guarantee) but with little real bias to remove it mostly
  captures calibration-corpus-specific noise that doesn't transfer to the
  held-out eval set. At s=0.9 the systematic masking bias is large enough
  that this correction dominates any variance cost.
  When relevant: (a) do NOT claim "refit helps" as a general property —
  it is confined to (and valuable in) the high-sparsity / coarse-block
  regime, and actively harmful outside it; (b) any deployment or follow-on
  compensation-branch design should gate L1 refit ON only past some
  (s, possibly g) threshold, not apply it uniformly; (c) the s=0.9, g=1 Go
  decision from the single-point verification is unaffected and now
  doubly confirmed (3B: −7.6%, 8B: −20.1%, both models agree in sign and
  both are large relative to any plausible noise floor).
  Journal: 2026-07-31_experiment-refit-l0l1-matrix-3b.md,
  2026-07-31_experiment-refit-l0l1-matrix-8b.md
- **[CONFIRMED] L2 LOSES to both L0 and L1 — a real property, NOT a
  calibration-budget artifact.** llama3.2-3b-instruct, s=0.9, g=1,
  wikitext-2 PPL, calib wikitext103: L0 21.586, L1 19.951, L2(n=128) 26.901,
  **L2(n=512, matched to L1's exact budget) 26.328** — 4x the calibration
  data moved PPL by only −0.57 (≈2%), nowhere near closing the ~6-7 PPL gap
  to L1. The sequential MECHANISM is independently verified correct (unit
  test 8: exact multi-layer restoration at s=0, dense/sparse streams agree
  to 1.86e-08), so this isn't an implementation bug — it's **error
  compounding**: L1 always regresses from the TRUE dense activations at
  every layer, while L2's layer L sees layer <L's own imperfect refit
  output, so approximation error accumulates through the stack. Same
  structural risk GPTQ-style quantization carries, and here it loses to the
  simpler dense-anchored alternative (L1), which is always available since
  refit doesn't require sequential treatment the way weight quantization
  does.
  When relevant: (a) do NOT add L2 to the broader (s,g) matrix — it already
  clearly loses at the one point tested, matched-budget, mechanism-
  verified; (b) this is informative for the compensation-branch line
  (oracle-residual-sparsity) too — matches that line's own experience that
  "more faithful to the deployed condition" objectives don't automatically
  win (whitening round's input-L2-vs-downstream-loss mismatch is the same
  shape of lesson); (c) L1 remains the confirmed, deployable form of this
  topic's method.
  Journal: 2026-07-31_experiment-refit-l2-validate-3b.md

- **[M1, honesty correction] Sharing-tax absorption headline was an
  exponential distortion — replaced with log-PPL (2026-07-31)**. The
  "3B 37% / 8B 60%" absorption figures in the topic-closing entry above
  were computed on raw PPL score, which distorts multiplicative
  quantities (PPL itself is exp(cross-entropy nats), so differencing raw
  PPL conflates "how much worse" with "how much worse, exponentiated").
  Recomputed on log-PPL (nats, i.e. mean cross-entropy) using the exact
  same underlying result JSONs: **3B absorption ≈ 12.4%, 8B ≈ 21.5%**
  (both roughly a third of the raw-PPL figures). Verified by direct
  recomputation from `~/workspace/refit/{llama3.2-3b-instruct,
  Llama-3.1-8B}/results/{l0,l1}_s0.9_g{1,128}*.json` — not re-estimated,
  the exact same PPL numbers, just a different (more honest) aggregation.
  **The old 37%/60% headline is demoted to "PPL-score basis" and must be
  reported alongside, not instead of, the log-PPL figures going forward**
  (per the requesting agent request's explicit instruction). Directional
  conclusion (refit absorbs more of the tax on the larger model, 8B > 3B)
  is UNCHANGED by this correction — only the magnitude was wrong.
- **[C1, critical correction, code done 2026-07-31 — GPU jobs in
  flight]**: `solve_refit`'s ridge was a 0-shrinkage prior (weak evidence
  pulls W_tilde toward the all-zero row). Fixed to anchor toward the
  ORIGINAL W_down instead (master design: `W_tilde = (C + lambda*diag_mean
  *W_anchor)(G + lambda*diag_mean*I)^-1`, reduces to the old formula at
  W_anchor=0). This is a real candidate alternate explanation for the
  s=0.5/0.7 "refit hurts" finding above (never ruled out before): if low-
  sparsity masking leaves little true bias to correct, the OLD ridge could
  have been actively pulling weak-evidence columns toward zero instead of
  leaving them at their trustworthy original value, adding pure noise on
  top of (or instead of) any real corpus-mismatch bias-variance story.
  Mathematically, the anchored solution is provably at least as good as
  the anchor on in-sample SSE for any lambda>=0 (same argument that shows
  0-anchored ridge always beats the zero vector). All refit+oracle unit
  tests re-verified, no regression. **Recalibration required** (G/C were
  never saved from the original s=0.5/0.7/0.9 runs) — build script now
  also saves raw (G,C,n) to disk per layer so this never has to happen
  again for a pure lambda/prior change. GPU jobs submitted (3B, s=0.9 and
  s=0.5, g=1) — not yet landed; **whether the s=0.5/0.7 "hurts" conclusion
  survives this fix is still open**, do not assume either outcome.
  Branch: `auto/refit-honesty-corrections` (off `auto/local-loss-refit`,
  PR #1 still unmerged) — separate from `block-sparse-compensation`'s own
  branch. Code not yet proposed via `agent-pr` (pending recalibration
  results so the PR description can report them).
- **[M1, DONE] Dense (s=0) PPL anchor (2026-07-31)**: 3B dense PPL
  **11.0489**, 8B dense PPL **6.2394** (no new code — `mode=l0,s=0.0` is
  the exact dense-forward identity, already unit-tested). Puts masking's
  OWN damage in frame for the first time: at s=0.9,g=1, masking alone
  (L0 vs dense) costs +95.4% PPL on 3B / +107.3% on 8B; refit (L1)
  recovers some of that but still lands +80.6% / +65.5% above dense —
  refit narrows the gap masking opens, it doesn't close it.

## Dead Ends
- 2026-07-31 — **L2 (sequential/GPTQ-style refit against the sparse
  stream)**: loses to both L0 and L1 at s=0.9,g=1 on llama3.2-3b-instruct,
  confirmed not a calibration-budget artifact (4x data barely moved it).
  Root cause: error compounding through the sequential layer chain — L1's
  dense-anchored independent-per-layer fitting doesn't have this problem
  and is simpler. Not pursuing further (no broader matrix, no lambda/
  partial-sequential variants) without a new idea for why it would turn
  around, and none is in hand. Journal:
  2026-07-31_experiment-refit-l2-validate-3b.md

## Open Questions
- ~~lm-eval-harness `piqa` task fails on this gateway env~~ → resolved by
  scoping down: `piqa` alone hits `datasets==5.0.0`'s removal of legacy
  HF-hub loading-script support (`lm_eval==0.4.3` defines `piqa` via one);
  the other 7 request-spec tasks (arc_easy, arc_challenge, boolq,
  hellaswag, winogrande, sciq, lambada_openai) all confirmed working
  (checked individually/in small groups, 2026-07-31). Suite going forward:
  7 tasks, `piqa` dropped, documented deviation from the request's 8-task
  list — not fixing by changing the shared conda env (blast radius beyond
  this topic).
- Exact s/g crossover point where ΔL1 flips sign looks model-dependent
  (3B: still negative at s=0.7,g=32; 8B: ~neutral there) — not pinned down,
  not blocking (s=0.9 Go stands either way), but relevant if this method
  is ever gated by a threshold rule.
- ~~Whether llama3.2-3b-instruct is an acceptable dev stand-in~~ → resolved:
  the low-s regression replicated on the base Llama-3.1-8B too, so the dev
  model's instruct tuning was not distorting the qualitative finding
  (magnitudes differ some, direction/pattern doesn't).
- ~~lm-eval-harness availability~~ → resolved: 0.4.3 confirmed installed in
  the gateway conda env (`~/miniconda3/envs/larosa`).

## Next Experiments
1. ~~Unit tests~~ — done, 5/5 pass (2026-07-31).
2. ~~Single verification point~~ — done, GO (2026-07-31).
3. ~~Reduced-cost L0/L1 matrix (rung 1)~~ — DONE (2026-07-31), both dev and
   main models. Confirmed finding: refit helps at s=0.9 (more so as g
   grows), hurts at s=0.5, crossover near s=0.7.
4. ~~Implement + validate L2~~ — DONE (2026-07-31), unit-tested correct,
   CONFIRMED to lose to L0/L1 at s=0.9,g=1 (not a calibration artifact —
   see Dead Ends). Not extending to the broader matrix.
5. ~~lm-eval-harness zero-shot suite~~ — DONE (2026-07-31), 4 jobs:
   Llama-3.1-8B, L0 vs L1 x s∈{0.9, 0.5} x g=1, 7 tasks (piqa dropped, see
   Open Questions), --limit 1000. CONFIRMS the PPL-based finding on the
   accuracy axis at both sparsity levels (s=0.9 Go +3.97%p avg; s=0.5
   hurts -1.19%p avg). Request's core Go/No-go question answered on two
   independent metrics.

**Request's core experimental question is now answered** (see Key
Findings, topic-closing entry). Backlog, not required for the request:
full g-sweep on the harness (only g=1 tested for accuracy so far, PPL
matrix covered g up to 128); a true dense (s=0) harness baseline for
framing "how much does masking cost, how much does refit recover" in
absolute terms; matching `piqa` (env-specific, low priority); L2 lambda/
partial-sequential variants (flagged as ideas in its Dead-End card, not
started).

## Active Jobs
- M1 dense anchors: DONE (`refit-dense-anchor-3b`, `refit-dense-anchor-8b`
  — see Key Findings).
- C1 recalibration: `refit-c1-build-3b-s09`, `refit-c1-build-3b-s05`
  running (a100-40-2, builds only so far — eval jobs to follow once
  these land). C2 (lambda sweep) queued for after C1's verdict is in.

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
