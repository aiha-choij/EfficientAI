# Init: local-loss-refit

Date: 2026-07-31

## Initial hypothesis
After freezing a top-K weight-aware mask (C2-score = |i|·col_norm) on the
FFN intermediate activation, a closed-form ridge-regression refit of
down_proj alone (no other repair mechanism) recovers a measurable share of
the accuracy lost to masking. Test at s ∈ {0.5, 0.7, 0.9} × token-sharing
block g ∈ {1, 8, 32, 128}, ladder L0 (mask only) / L1 (mask + per-layer
independent refit, dense input) / L2 (mask + sequential GPTQ-style refit,
sparse-stream input).

## Starting phase
Design/implementation. Request is self-contained (host research-wiki spec
inlined into the request text); no prior journal entries exist for this
exact question — it is a deliberately narrow spinoff of
oracle-residual-sparsity's C2 condition, isolating refit's own contribution
before any of that topic's compensation-branch machinery.

## Notes
- Base infra survey done: oracle_mlp.py / modeling_llama_larosa.py /
  scripts/oracle/*.py / test_oracle_units.py all read in full; new code
  will reuse oracle_mlp.py's mask/score/iter helpers without modifying that
  file (oracle-residual-sparsity's own active job must stay undisturbed).
- At first read, oracle-residual-sparsity's current.md said "🟢 active", not
  "wrapped/paused" as the request text claimed; a later re-read (after
  branching) showed commit 05ce2f7 "pivot: oracle-residual-sparsity wrapped"
  had since landed on main — repo state now matches the request's framing.
  Either way this topic only reads that infra, never touches
  oracle-residual-sparsity's files or jobs.
- All work for this request (topic init included) committed on
  `auto/local-loss-refit`, not `main`, per this request's operating
  contract (agent-pr only, no self-merge).
