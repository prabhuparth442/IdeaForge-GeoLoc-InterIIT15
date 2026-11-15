# Test_Iteration_10: evaluation fixes, inlier-scale robustness, accuracy experiments

Full write-up: [REPORT.md](REPORT.md). Handoff / context for another machine: [../HANDOFF.md](../HANDOFF.md).

## Result on the fresh NUDT 808 draw (300 photos, dumped before any rule was chosen, finalists pre-registered, scored once)

| | score | wrong | gain vs base (95 %) | wrong at inlier scale x0.5 / 0.75 / 1 / 1.25 / 1.5 / 2 | queries |
|---|---|---|---|---|---|
| PC base (bar 8 / 11) | 52.08 | 5 | — | 0 / 3 / 5 / 6 / 6 / 12 | 8.1 |
| PC J0 (T8 judge + 12-inlier override) | 54.80 | 0 | +2.72 [+0.92, +4.92] | 0 / 0 / 0 / 4 / 5 / 8 | 8.2 |
| **PC V5 (relative inliers + scale augmentation, no override)** | **54.48** | **0** | **+2.40 [+0.52, +4.64]** | **0 at every scale** | 8.3 |
| PC J2 (judge, no override, cut 0.98) | 54.20 | 0 | +2.12 [+0.16, +4.32] | 0 at every scale | 8.7 |
| Jetson today | 52.68 | 2 | — | 0 / 2 / 2 / 3 / 4 / 9 | 6.4 |
| Jetson J0 (+ early exit) | 53.32 | 0 | +0.64 [-0.48, +2.04] | 0 / 0 / 0 / 2 / 2 / 3 | 2.2 |
| **Jetson V5 / J1 (no override, early exit)** | 53.00 | **0** | +0.32 [-0.92, +1.80] | **0 at every scale** | **2.2** |

**Recommendation:** PC V5; Jetson V5 or J1 with early exit (no proven score gain, but 0 wrong at every scale and ~5 s instead of ~14 s per photo);
remove the 12-inlier override everywhere. Ask first whether NUDT-trained judges are allowed (satellite-only constraint?).

## Parts

| Part | Result | Notebook / block |
|---|---|---|
| 1a Jetson context from the 5 matched candidates | small effect (707 +0.80 -> +1.00; official TEST unchanged) | `It1_Evaluation_Fixes`, `candidate_judge.add_context(m)` |
| 1b empty shortlist -> skip; 1c failed calibration -> answer nothing | fixed + unit tests | `candidate_judge`, `test_iteration10_blocks.py` |
| 1d manifest vs headless | 4 differences (hint, query cap, 0.95 scale, matcher confidence to check) | `eval_tools.py` |
| 1e failure log | official: no retrieval losses, loss = retrieved but rejected; NUDT: 3-17 never retrieved per set | `eval_tools.failure_log` |
| 2 inlier-scale robustness | V5 flattens the scale failure (PC 1 wrong at every scale on design hold-outs, the same 606 look-alike; V0: 31 at x2); on 808 V5 / J2 / J1 0 wrong at every scale | `It2_Scale_Robustness`, `scale_robust.py` |
| 3a targeted verification | ceiling <= 0.4 / 60, not built | `It3_Targeted_Verification` |
| 3b coarse / DINO mixture | official: DINO never drops the right place; NUDT 101: 6 of 600 queries, +1 photo; not adopted | `It7_Coarse_DINO_Mixture` |
| 3c stage cache, shared map FFT | identical outputs; cache -41 % time per research run; FFT sharing -3 % (adopted for research) | `It4_Stage_Cache`, `stage_cache.py` |
| pre-registration, final | `results/preregistered_t10.json` (15:25, before 808 was read) | `It5_Preregister`, `It6_Final_808` |
