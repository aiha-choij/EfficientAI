# init — larosa-repro

- Date: 2026-07-22
- Initial hypothesis: Upstream LaRoSa code reproduces its README reference lm_eval numbers on Qwen2.5-7B at sparse_level 0.25.
- Starting phase: running experiments (reproduction).
- Notes: Repo forked to aiha-choij/EfficientAI; gateway clone at a100-40-2:~/workspace/repos/EfficientAI. Env constraint found during setup: driver 550 (CUDA 12.4) → torch must be cu124 build; default pip torch now ships cu13 wheels which fail. Anaconda default channels require ToS acceptance — env created from conda-forge instead.
