# labtool config

- project: EfficientAI
- gateway_host: a100-40-2
- gateway_bin: ~/workspace/bin
- gateway_runs: ~/workspace/runs
- remote_workdir: ~/workspace/repos/EfficientAI/larosa
- sync_note: Local repo and gateway clone (a100-40-2:~/workspace/repos/EfficientAI) both track github.com:aiha-choij/EfficientAI. Sync = push from local, then `ssh a100-40-2 'cd ~/workspace/repos/EfficientAI && git pull'`.

## Environment (gateway)

- conda env: `~/miniconda3/envs/larosa` (python 3.10, conda-forge)
- key deps: torch 2.6.0+cu124, transformers 4.46.3, lm-eval 0.4.3, flash-attn
- models dir: `~/workspace/models/`
- driver: 550.163.01 (CUDA 12.4) — cu13 wheels do NOT work; pin cu124 builds.
