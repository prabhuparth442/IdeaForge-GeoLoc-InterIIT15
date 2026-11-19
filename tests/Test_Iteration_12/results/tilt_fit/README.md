# Tilt fix from the matches in the photo view (iteration 44), measured 2026-11-18

Notebook: tests/Test_Iteration_12/notebooks/Tilt_Fit_From_Matches.ipynb (executed copy here), chain scripts/server_chains/tiltfit_chain.sh. 125 working photos from 1000 m with at least 20 inliers; XoFTR matches in the photo view at a candidate 30-60 m from the truth (random direction, fixed seed); the truth is used only to measure the error.
V0 = median of the match votes (what the pipeline did), V1 = pose fit (position, pitch, roll, yaw, height) with the audit's assumed noise as the prior (1, 1, 2 degrees, 2 %), V2 = priors 3 times wider, V3 = position and height only.

| variant | median error / sigma | within 1 sigma | within 2 sigma | mean points per photo |
|---|---|---|---|---|
| V0 median | 1.58 | 19 % | 70 % | 5.76 |
| **V1 fit, nominal priors** | **0.93** | **54 %** | 90 % | **7.86** |
| V2 fit, priors x3 | 0.99 | 50 % | 83 % | 7.49 |
| V3 no tilt | 1.44 | 27 % | 74 % | 6.15 |

Kill rule (gain >= 0.3 points per photo, both odd/even halves, priors not tuned): V1 gain +2.10 (halves +1.91 / +2.26): useful. Why it works here when iteration 41 failed: the photo is wide (76 degrees) and the photo view gives about 80 matches over the whole picture, so the perspective change across the picture (about 2.7 % edge to edge per degree of tilt) is measurable.
Caveats: the 1/1/2/2 % prior is the noise assumed in the synthetic audit (it reproduced the real error there), not measured from the metadata; these are the development photos, not a locked set; the candidate offsets are simulated, not coarse peaks (the pipeline run is the real test).
The pipeline version (photo_view.fit_pose, numpy only, scipy is not a dependency) matches the scipy results (median difference 1.2 m on the same matches).
