# Map drawn in the photo's own view, above 1000 m (iteration 43), measured 2026-11-18

Notebook: tests/Test_Iteration_12/notebooks/Map_In_Photo_View.ipynb (executed copy here). Server chain: scripts/server_chains/photoview_chain.sh. 130 working photos at 1000 m and above, 3 heading queries, true place plus 4 wrong places 3 km away, bar per input from its own wrong-place counts, cross-validated (odd/even photos).

| input | LoFTR recall | XoFTR recall | XoFTR AUC | XoFTR true-place median inliers |
|---|---|---|---|---|
| canvas (today) | 0.43 | 0.60 | 0.90 | 14 |
| photo view (photo as it is, map drawn in its view) | 0.70 | **0.95** | 1.00 | 77 |
| photo view x2 | 0.32 | 0.69 | 0.85 | 8.5 |

XoFTR by band (canvas -> photo view recall): 1000-1300 m 0.74 -> 0.95; 1300-1600 m 0.53 -> 0.93; 1600-2100 m 0.52 -> 0.98. LoFTR above 1600 m: 0.05 -> 0.40.
Kill rule (>= 0.10 recall gain in >= 2 of 3 bands): passed by both matchers' plain photo view in 3 of 3 bands; the x2 variant is killed.

This is a diagnostic at the TRUE place with the metadata pose, not a pipeline score. Not yet shown: the coarse search gives a place only within about a map pixel, the answer position has to be recovered from the matches in the photo view, and the other two sets (old official, NUDT) and wrong-answer behaviour are untested. Next step: wire it into the fine stage behind a switch and run the pipeline on the three sets.

## In the pipeline (iteration 43b), bar 20, 332 working photos, measured 2026-11-18
Switches: XOFTR_FROM_HEIGHT_M=800, PHOTO_VIEW_FROM_HEIGHT_M=1000, PHOTO_VIEW_MIN_INLIERS=20 (code: Main_Pipeline_Headless/geoloc/photo_view.py, tests/test_photo_view.py, scripts/server_chains/photoview_run_chain.sh). Run: details_pipeline_bar20.csv.
Score 39.70 -> **42.04** of 60, answered 284 -> 321, 0 wrong, 1.78 -> 1.42 s per photo. On the 130 photos from 1000 m: 4.42 -> 5.41 points per photo, answered 86 -> 123, 0 wrong. 51 photos better, 29 worse (mostly a +10 answer replaced by a +6 or +3 one).
Precision is now the limit: on the 85 photos answered by both paths the median error is 1.45 sigma in the photo view and 1.26 sigma in the canvas, and it does not improve with more inliers (80-160: 1.25; above 160: 1.36), so it is the metadata pose that limits it (the drawn map is only as right as the pose).
Bars 16 and 24 were started as a sensitivity check (out_pv_b16, out_pv_b24 on the server). Old official (below 351 m) and NUDT (below 320 m) never use the photo view, so their scores are unchanged by construction. The locked test split has not been run with this.

## Pose fit in the pipeline (iteration 44b), 332 working photos, measured 2026-11-18
PHOTO_VIEW_FIT_POSE=True (the answer's position from a fit of the camera pose to the winner's matches, numpy only), with XoFTR from 800 m, photo view from 1000 m, bar 20. details_pipeline_bar20_posefit.csv; scoring script score.py (same rule as the headless pipeline, recomputed from the files).

| run | score / 60 | answered | wrong | within 1 sigma (answered) |
|---|---|---|---|---|
| XoFTR from 800 m (base) | 39.70 | 284 | 0 | 52 % |
| + photo view from 1000 m, bar 20 | 42.04 | 321 | 0 | 47 % |
| + photo view, bar 16 | 42.31 | 324 | 0 | 47 % |
| **+ photo view + pose fit, bar 20** | **46.07** | 321 | 0 | **60 %** |

On the 130 photos from 1000 m: 4.42 (base) -> 5.41 (view) -> **7.12** points per photo with the fit; the fit was used on 110 of them (median move 26.6 m from the median-vote answer; it is dropped when it moves the answer more than 3 sigma). The bar changes little (16: +0.27 over 20). The priors are the audit's assumed noise, not tuned on these photos. Not yet checked: locked test split, old official, NUDT (below the gate, unchanged), the gate height.

## Gate height, rounds, priors, other sets (iteration 45, night of 2026-11-18, interim)
Server runs (details in runs/), all with XoFTR from 800 m, pose fit on, bar 20, agreement rule as in settings (on) unless a name says otherwise. pvs_* = Main_Pipeline_Headless, pv2_* = the copy Main_Pipeline_V2 (adds fit diagnostics columns and the render-match-fit rounds). Score / 60 recomputed with score.py.

| set | run | score | answered | wrong | within 1 sigma |
|---|---|---|---|---|---|
| working 332 | XoFTR base | 39.70 | 284 | 0 | 52 % |
| working 332 | photo view + fit from 1000 m | 46.07 | 321 | 0 | 60 % |
| working 332 | ... from 800 m (pvs_dev_g800) | 46.10 | 322 | 0 | 59 % |
| working 332 | ... from 600 m (pvs_dev_g600) | 46.57 | 321 | 0 | 62 % |
| working 332 | ... on every photo (pvs_dev_g0) | 48.00 | 322 | 0 | 65 % |
| working 332 | ... every photo, priors x0.4 (pv2_dev_g0p04) | **48.70** | 322 | 0 | 67 % |
| working 332 | two render-match-fit rounds, from 1000 m (pv2_dev_r2) | 46.14 | 321 | 0 | 60 % |
| NUDT headless 300 | XoFTR base | 48.52 | 281 | 12 | 94 % |
| NUDT headless 300 | photo view + fit on every photo, bar 20 (pv2_nudt_g0) | **52.28** | 269 | 2 | 98 % |
| old official 500 | XoFTR base | 52.69 | 454 | 1 | 93 % |
| old official 500 | photo view + fit on every photo, bar 20 (pvs_official_g0b20) | 52.08 | 445 | 0 | 94 % |

Readings. (1) A second round of render-match-fit gives nothing (+0.07, noise): killed, PHOTO_VIEW_FIT_ROUNDS stays 1. (2) The lower the gate the better on the working set (the photo view helps below 1000 m too). (3) NUDT: wrong answers 12 -> 2 and +3.8 points; its two wrong answers have 11 inliers against a median of 68 for right ones (lookalike.py: inliers AUC 0.99 for wrong), so the agreement rule that answers on fewer inliers is the likely cause. (4) Old official: bar 20 is too high at low heights (9 fewer answers, -0.6); a bar grid with the agreement rule off is running (queue 2). (5) Priors x0.4: chosen by cross-validation (odd/even, offline, results/tilt_fit and offline_fit.py): +0.62 on the working set at gate 600, +0.7 at gate 0; NUDT result pending. The fit's own match noise (1.5 px) and robust scale (2) are at the cross-validated optimum; a free radial-distortion term adds only +0.17 per photo offline and is not used.
Not yet done: the bar x agreement grid (queue 2), the locked test split, Jetson timing, a freeze.

## Frozen configuration PV-X and the three sets (night of 2026-11-18 to 19)
PV-X = XoFTR for every photo + photo view for every photo + bar 12 + pose fit (priors x0.4) + agreement rule on. Definition and the order of events: FREEZE_PV-X.md. Score / 60, runs in runs/:

| set | base | PV-X | wrong (base -> PV-X) | answered |
|---|---|---|---|---|
| working 332 (new set, development) | 39.70 | **50.69** | 0 -> 1 | 284 -> 329 |
| old official 500 (development) | 52.69 | **60.00** | 1 -> 0 | 454 -> 500 |
| NUDT headless 300 (development) | 48.52 | **52.08** | 12 -> 6 | 281 -> 282 |
| locked test 168 (second look, report only) | 38.00 | **49.32** | 0 -> 1 | 140 -> 166 |

Other variants on the way (same code, runs/): bar 20 with the agreement rule: official 52.08, NUDT 52.28 (2 wrong), working 48.70 at all heights (priors x0.4); bar 12, LoFTR below 800 m: working 48.90, NUDT 52.28 (5 wrong), official 57.24; agreement rule off, bar 12: working 48.90, NUDT 52.48 (3 wrong), official 54.43 (the rule helps official); bar 16 with the rule off was clearly worse on official and stopped. A rival-place rule (drop an answer when another place has a comparable inlier count) removes NUDT's wrong answers but costs old official more than it gains: killed. XoFTR below 800 m: official 57.24 -> 60.00, working 48.90 -> 50.69, NUDT 52.28 -> 52.08 (no change).
Remaining loss per 60: working: imprecise 8.4, skipped 0.05, one wrong; NUDT: 6 wrong (about 3.6 points), imprecise 1.9, skipped 0.4; official: none.

## PV-X with a second hint-jitter seed (2026-11-18, night)
The hint (the GPS the pipeline starts from) is moved randomly up to 5 km from the truth to imitate the evaluation; seed 0 was used everywhere above. Seed 1, same configuration: old official 60.00 (0 wrong, identical), new set 50.26 (2 wrong; seed 0: 50.69, 1 wrong), NUDT 51.68 (8 wrong; seed 0: 52.08, 6 wrong). So the run-to-run spread from the hint is about 0.4 points on the two sets with errors, and NUDT's wrong answers are 6 to 8.

## Second opinion on borderline answers (iteration 47, log only, 2026-11-18, night)
When the winner of a photo has fewer than 20 inliers (PHOTO_VIEW_VERIFY_BELOW=20, nothing acts on it) the other matcher (LoFTR, since XoFTR answers) matches the same place in the photo view; runs/pvv_*.csv (Main_Pipeline_V4, columns verify_inliers, verify_distance_m). NUDT (14 borderline answers, 4 wrong): the other matcher finds 0 inliers at all four wrong answers and at 1 of 10 right ones (the others 6 to 26); old official (18 borderline answers, 0 wrong): right answers get 0 to 19 (one gets 0); the new set has none below 20 inliers. A rule "skip when the other matcher finds fewer than 3 inliers or lands more than 2 sigma away" would gain +1.48 on NUDT and cost 0.12 on old official (in-sample on 4 wrong answers: a hypothesis, not a result). A rival-place rule (second-best place with a comparable inlier count) did not generalise and is killed (README above). The same check on answers below 40 inliers, with both seeds, is the next run.

## PV-X2 = PV-X + second opinion on borderline answers (iteration 47, measured 2026-11-19, night)
Rule (PHOTO_VIEW_VERIFY_BELOW=20, _ACT=True, _MIN=3, _MAX_HEIGHT_M=1000; code in Main_Pipeline_V4 = Headless at this date): when the winner has fewer than 20 inliers and the photo is up to 1000 m high, the other matcher (LoFTR) matches the same place in the photo view; the photo is skipped when it finds fewer than 3 inliers or lands more than 2 sigma from the answer. Logged first (log-only runs pvv_*_v0, v40*, pvv_*_x2*), then run as a filter (runs/pvv_*_x3*).

| set (seed) | PV-X | PV-X2 | wrong | answered |
|---|---|---|---|---|
| NUDT seed 0 | 52.08 | **53.56** | 6 to 2 | 282 to 277 |
| NUDT seed 1 | 51.68 | **53.96** | 8 to 2 | 286 to 279 |
| old official seed 0 | 60.00 | 59.88 | 0 to 0 | 500 to 499 |
| new set (332 working) seed 0 | 50.69 | 50.64 | 1 to 1 | 329 to 328 |

Limit 40 instead of 20: the same gain on NUDT but old official loses about 1 point (59.04 seed 0, 58.92 seed 1) because LoFTR fails on right answers with 16 to 38 XoFTR inliers. Caution: chosen on 4 to 8 wrong answers of one set (hypothesis, in-sample); the 6 wrong answers left on NUDT seed 0 include one that both matchers confirm (a true look-alike). Report: Documents/Outputs/Progress_Report_v0.7_Photo_View_and_Pose_Fit.pdf. The locked test split was run only with PV-X, not PV-X2.
