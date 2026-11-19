# Red team of PV-X and the robust version PV-X-R2 (2026-11-19)
Reviewers: leakage_audit.md, mechanism_audit.md, generalisation_plan.md (this folder). Server runs below: 166 dev photos (every 2nd; "odd" = the other half with a new noise draw), Main_Pipeline_V3, hint jitter 5000, seed 0. Per-photo details: runs/*.csv; scripts rt_*.py/sh, make_variants*.py. Scores /60, wrong in brackets. Sigma for height-biased arms is taken from the original height.

## Findings
- No ground-truth leakage and no cancelling sign error (mechanism_audit.md). The new-set gain is real (cluster-bootstrap CI +8 to +14.5); old official 60.00 and the NUDT wrong-answer drop are partly in-sample (leakage_audit.md).
- False-answer null test (hint moved so the truth is outside the 5.5 km disc, dev 332): 21 of 254 such photos still answered (8.3 %), 14 of 95 above 1000 m (15 %); inliers up to 25. Hint 25 km: 10 of 29 answers wrong. It only matters when the coarse stage misses the true place; it shows the bar of 12 has little margin at height.
- Black or empty image regions break it: side bands blacked out 3.2, shrunk-into-black-frame altitude emulation answers 6 of 180 (so that emulation says nothing about 3000 m).
- Cost: GPU peak 3.4 GB (coarse search level 3; most photos under 0.5 GB), 1.3 to 3.3 s per photo on the shared server. Nothing is trained, so there are no training stats.

## Metadata error (166 photos). PV-X vs base vs PV-X-R2
| arm | base | PV-X | PV-X-R2 |
|---|---|---|---|
| clean | 36.80 | 51.58 (0) | 52.41 (0) |
| noise 2x (2 deg tilt, 4 deg yaw, 4 % height) | 22.70 | 35.13 (4) | 45.00 (4) |
| noise 3x | 14.82 | 21.65 (10) | 35.60 (10) |
| height +10 % | 31.73 | 31.95 (9) | 37.48 (9) |
| height -10 % | 36.61 | 46.30 (1) | |
| tilt +2 deg | 18.87 | 35.60 (2) | 47.17 (2) |
| yaw +3 deg | 36.36 | 50.39 (1) | |
Unseen odd photos, new noise draw: clean 49.66 -> 48.61; noise 2x 33.14 -> 45.90; 3x 16.23 -> 33.69; height +10 % 33.47 -> 37.45; tilt +2 34.45 -> 46.16 (PV-X -> PV-X-R2). Old official 60.00 -> 59.66 (0 wrong). NUDT 52.08 -> 52.08 (identical, attitude is off there).
PV-X-R2 = FIT_ROUNDS 2, priors 2/4 deg and 5 %, move limit 6 sigma (presets/pv_x_r2.py). Prior width alone does nothing (noise 2x, prior 1x: 35.46); the second round plus the wide prior does it. Chosen on the even photos, confirmed on the odd photos with a different noise draw.
Photo changes: invert 48.29, low contrast 50.39, mild blur (1 px) 45.54, blur 2 px 37.30 (base 19.70), added noise (0.5 std) 30.47 with 98 of 166 answered. Time budget cut to 1/2.4: no loss (5x and 10x did not run, queue bug).

## Still open
Height +10 % recovers only about 4 points; 3x noise still has 10 wrong answers; altitude above 2017 m untested (a real test needs level-13 map data and photos); other region or camera untested; Jetson timing; the null-test false answers suggest a bar that grows with height or a rival-place check. The locked split was not rerun with PV-X-R2 (a fourth look).
