# Test_Iteration_10: evaluation fixes, inlier-scale robustness, accuracy experiments (IN PROGRESS)

Brief and full status: [../HANDOFF.md](../HANDOFF.md), sections 4-5.

| Part | Status | Where |
|---|---|---|
| 1a Jetson context from the 5 matched candidates | done: small effect (707 +0.80 -> +1.00, official TEST unchanged) | `It1_Evaluation_Fixes` |
| 1b empty shortlist -> skip | done + unit test | `candidate_judge.decide`, `test_iteration10_blocks.py` |
| 1c failed calibration -> answer nothing | done + unit test | `ltt_tau`, `NO_ANSWER` |
| 1d configuration manifest vs headless | done | `eval_tools.py`, `It1_Evaluation_Fixes` |
| 1e failure log per photo | done | `results/it1_failure_log.csv` |
| 2 inlier-scale robustness (V0-V5, x0.5-x2) | notebook written, run not finished | `It2_Scale_Robustness`, `scale_robust.py` |
| 3a targeted verification | headroom only: <= 0.4 / 60, not built | `It3_Targeted_Verification` |
| 3b coarse / DINO mixture, 3c stage cache | code written, GPU runs pending | `stage_cache.py`, `Rich_Dump_Cached`, `run_gpu.sh` |
| fresh NUDT 808 + pre-registration | pending (in `run_gpu.sh`) | — |
