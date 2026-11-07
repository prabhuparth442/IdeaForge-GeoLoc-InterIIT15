# Stage evaluation on the Cartosat map: what it says and which strategies are viable

6 Nov 2026. Evidence: `tests/Stage_Evaluation/` (per-photo CSV, K/M/weight sweep, bar/gamma/rescue sweep, map table), `Stage_Evaluation.ipynb` with its saved outputs, `Patch_v0.5.6_Decision_and_Bias`, and the older `Single_Image_Run` CSVs. All numbers below were recomputed from the files.

Tags: **[measured]** computed from the files; **[in-sample]** a rule chosen on the same 100 photos it is scored on; **[derived]** arithmetic or theory; **[guess]** an estimate with real uncertainty. Points are per 100 photos (max 1000); 1 point per 100 photos = 0.06 out of 60.

---

## 1. Read this first: the ground moved

1. **The base map is now Cartosat-3 PAN.** The new runs search `Dataset_Official_2` (scenes 268234931 and 268234921): `SatID=CARTOSAT-3`, `Sensor=PAN`, 1 band, ortho-rectified, 0.28 m, acquired 8 Feb 2025 near Mumbai. This is the evaluation imagery type, so the Stage_Evaluation run is the **first test on the real kind of map** (the old runs used the Pleiades Neo `small_sat_area`, 6 bands, 0.3 m).
2. **The matcher is also stricter**: `RANSAC_PIXELS = 3.0`, `MATCHER_CONFIDENCE = 0.3`. Median inliers of the chosen candidate fell from about 42 to **14**.
3. **Consequence: the old tables and the new tables are not comparable.** The v0.5.6 numbers (710 for LoFTR, 843 under its gate, the 12-inlier bar) came from the old map and the looser matcher. On the new run the same old rule ("answer if inliers >= 12") scores 552, not 710.
4. **File hygiene.** `Single_Image_Run/per_photo_official_fused_layer1_K50_M25_loftr_minima.csv` was overwritten at 02:03 UTC on 6 Nov by the new-map run (it is identical to the Stage_Evaluation CSV, and a copy `..._official_new_...` was added at 05:59). The old-map LoFTR file behind v0.5.6 is gone, and the LightGlue and NUDT files in that folder are still old-map / old-settings. Do not compare them with the LoFTR file. Keep dated copies of CSVs from now on.
5. A correction to my own earlier prompt: `z_chosen` is the **DINO dense z over the 50-place shortlist**, not the coarse correlation z (v0.5.6 had this right).

---

## 2. The funnel on the new map [measured] (100 photos, bar 8, K=50, M=25, DINO weight 100)

| step | photos | lost here | ceiling if the rest were perfect |
|---|---|---|---|
| all photos | 100 | | 60.0 |
| target in the 50 peaks (coarse) | 89 | **11** | 53.4 |
| target in the 25 sent to fine | 89 | 0 | 53.4 |
| right candidate chosen | 87 | 2 | 52.2 |
| answered (bar 8) | 62 | **25** | 37.2 |
| ... within 1 sigma | 59 | 3 | 35.4 |

Actual score 36.48 / 60. The two big buckets are the **decision (25 photos)** and **coarse recall (11 photos)**. Precision is a small bucket here.

Coarse maps, share of photos whose global maximum is at the target (4 sigma): edge ZNCC 0.76, fused 0.78, learned 0.73, MIND 0.72, whitened 0.75, cosine 0.57, magnitude 0.19. After the DINO re-order, top-1 (4 sigma): **edge ZNCC 0.76 -> 0.90** (14 fixed, 0 broken, +0.14 [+0.07, +0.21]); **fused 0.78 -> 0.85** (10 fixed, 3 broken, +0.07 [0.00, +0.14]). Recall at 50: edge 0.90, fused 0.89.

By flight height (bar 8): <=125 m: 0.41 answered, 4.1 points per photo; 125-200 m: 0.45, 4.5; 200-300 m: 0.96, 9.4; 300-400 m: 0.74, 7.0. **35-40 % of the photos at <=200 m have 0 inliers** (median 6.5), against 0 % at 200-300 m (median 67).

---

## 3. What happens to the v0.5.6 claims on the new map

| v0.5.6 claim | on the new map | verdict |
|---|---|---|
| Confirmation gate: an inlier-best candidate that is not the top-1 is always wrong (0 of 12) | **2 overrides, both right** (photos 3 and 5: 4 and 7 inliers, z 1.4-1.5). No wrong answer exists at any bar >= 1 | Not reproduced. Pooled with the old map: 2 of 14 overrides right. Keep a guard, do not hard-code the rule (S6) |
| 0-inlier top-1 answered when DINO z >= 2.8 | 8 of 8 right at z >= 3.0, 0 wrong. z >= 2.5 adds 2 wrong; z >= 2.0 adds 7 wrong | **Reproduced**, with the cut sitting in the gap 2.73-3.00 (S2) |
| The 12-inlier bar is the wrong variable; inlier count depends on the matcher | Confirmed more strongly: with the stricter matcher the useful bar is **1-3**; with the old one it was 5-12 | Holds. Bar and RANSAC settings must be tuned together |
| The 1-sigma floor is an angular bias (error = 0.042 h, 2.4 degrees), not a map offset | Median error of answers with >= 8 inliers is **0.58 sigma**; only **5 %** are above 1 sigma (old map: 26-35 %). On 45 photos answered with >= 15 inliers on both maps the median error dropped from 7.6 m to 5.6 m (correlation of the errors 0.56; LightGlue on the old map against LoFTR on the new one, but v0.5.6 found the two matchers agree to 0.8 m). A through-origin fit gives 0.0216 h (1.2 degrees) now, 0.0281 h on the old map for the same photos | **Weakened**: part of the floor was map-dependent. A smaller height-proportional term (about 1.2 degrees) remains, but unsigned errors cannot name its cause |
| 19-21 sigma near-misses (photos 38, 82) come from the shift limit | Photo 38 is now **0.14 sigma** (0 inliers, z 1.58). Photo 82 is now outside the 50 peaks (163 sigma) | The 156 m offset moved when only the map changed. Drop the shift-limit re-centring (v0.5.6 item 5) from the plan until a signed-error run says otherwise |
| Learned features, MIND, whitening and dense-DINO fusion are deprioritised because top-1 is 0.96 | Top-1 is **0.85** and recall is **0.89** here, and 11 photos are lost at the coarse stage | Recall matters again, but fusion is not the lever (see S5) |
| Honest total 52-56 / 60 | This sample's recall ceiling is 53.4 and the best rule below reaches 48.6 | 52+ needs a recall gain; do not plan on it yet |

---

## 4. Viable strategies

Rule definitions: **R1** = answer if inliers >= 3 (bars 1, 2 and 3 give identical results, 75 answered). **R2** = R1, plus answer the unrefined top-1 when it has 0 inliers and its DINO z >= 3.0.

### S1. Lower the answer bar to 3 [measured, in-sample]
- 608 -> **734** (36.5 -> 44.0 / 60), 75 answered, **0 wrong**. Paired bootstrap gain over bar 8: **+7.56 [+3.96, +11.76]** out of 60. Both halves improve (37.4 -> 48.2 and 35.5 -> 39.8).
- Answered precision is high: 71 within 1 sigma, 4 within 2 sigma. Bar 4 gives 718, bar 6 698, bar 12 552.
- Risk: the bar is only safe for the present matcher settings. With the looser old settings, 3-6 inliers included wrong answers. Re-validate every time RANSAC tolerance, confidence or the matcher changes, and log raw putative pairs so a significance measure (NFA, v0.5.6 Sec. 2.6) can replace the raw count.
- Cost: minutes (the setting is `MIN_INLIERS`).

### S2. Answer the 0-inlier top-1 when DINO z >= 3.0 [measured, in-sample]
- Adds **+76 points** (+4.6 / 60) on top of S1: 810 (**48.6 / 60**), 83 answered, 0 wrong; gain over bar 8 **+12.12 [+7.56, +16.92]**. Halves 51.4 and 45.8.
- The eight photos (all within 1.2 sigma): 61, 60, 90, 42, 6, 77, 79, 37. Among 0-inlier photos, the AUC of `z_chosen` for "top-1 is right" is **0.84**; contrast 0.66, valid share 0.56, edge strength 0.38, height 0.42.
- Why 3.0: [derived] if DINO had no information the best of 50 standardised scores averages 2.25 (sd 0.43); P(max >= 2.8) is about 12 %, P(>= 3.0) about 6.5 %. The wrong 0-inlier photos sit at 1.5-2.73, the right ones from 3.0 up (two near the cut: a wrong one at 2.73 scores 0, a right one at 3.00).
- Evidence limit: break-even for an answer worth about 9 is P(right) > 0.69, and 8 of 8 only gives a Wilson lower bound of 0.68. Pooling the old-map files adds 16 photos with z >= 2.9 that all scored >= 0 (lower bound about 0.81), but those are largely the same official photos.
- Do not lower the cut: 2.5 gives 770 (2 wrong), 2.0 gives 700 (7 wrong, interval includes zero).
- `RESCUE_LIST` in the notebook never tries a 0-inlier rescue; add `(0, 3.0)` if `decide_with_dino` accepts 0.

### S3. Rescue the 4 right, 0-inlier photos at low altitude [guess]
- Photos 13, 38, 59, 96 (heights 100-123 m, z 1.58-2.27) are right at the top-1 but have no inliers: **40 points (+2.4 / 60)**. 13 other 0-inlier photos at similar z are wrong, and a pure z rule cannot separate them.
- Options: a second matcher on the top-1 only (`MATCHERS` already supports `roma_minima` and `lightglue`); a bigger map context or second scale; accept only if two weak independent matches agree. The fine stage handles 25 candidates in about 2 s here, so there is room.
- Expected: +1 to +2 / 60 [guess]. Cost about 1 day.

### S4. Spend the fine-stage budget on fewer candidates [measured]
- At K = 50, DINO weight 3-10 puts the target among the **first 5 candidates** as often as it is in the 50 (target in M = 0.88-0.89 at M = 5; weight 100 gives 0.89), with top-1 0.88 against 0.85 at weight 100 and 0.78 with no DINO. Differences of 0.03 are 3 photos, so treat them as equal.
- So the fine stage could run on 5-10 candidates instead of 25, which frees most of the 2 s for S3 or a second matcher. The sweep's score column is useless for this (36.48 in almost every row) because it was run at bar 8; re-run it with R2 to see the score effect.

### S5. Recall: the 11 photos lost at the coarse stage [measured + guess]
- Ceiling: 11 photos x 10 = **110 points (+6.6 / 60)**, the largest remaining bucket. All 11 have 0 inliers and are skipped, so there are no wrong answers, which also satisfies the feature-based no-match rule.
- Why they are lost: the correlation maps do not show the target (median z near the truth 2.16 against 3.79 for found photos); three of them do have z >= 2.9 near the truth but rank below 50 peaks. Heights: 4 at <=125 m, 4 at 125-200 m, 3 at 350 m; valid share is the same as for found photos.
- Ideas, cheapest first:
  1. **Larger K with DINO as judge** (K = 100-200): the target-in-K curve is still rising (0.84, 0.86, 0.89 at K = 10, 25, 50) and DINO ranks the true place first almost always (separation 0.998). Set `K_MAX = 200` in the notebook.
  2. **Use edge ZNCC as the first stage and drop the six-map fusion**: after DINO it reaches top-1 0.90 against 0.85, with recall 0.90 against 0.89, and it is cheaper than six maps including a learned CNN. The difference is 5 photos, not significant, so run it on all 379 photos with the fine stage before changing anything. The fine stage was run on the fused peaks only.
  3. Small hypothesis grid (yaw / scale) for the coarse stage (v0.5.1 E1/E3).
  4. Adapt the learned features to the PAN map (F4/F5) only if the above fail: the learned map is currently below edge ZNCC here (0.73 vs 0.76).
- Expected: 0 to +4 / 60 [guess].

### S6. Overrides policy [measured, small]
- The fine stage picks a candidate other than the top-1 for photos 3 and 5, both right here, and 12 of 12 were wrong on the old map. A hard "top-1 only" gate costs 20 points here and protects against -20s elsewhere; break-even for an answer worth 10 is P(right) > 0.67 and pooled evidence is 2 of 14.
- Suggested: allow an override only when a second matcher agrees, and log `chosen` (it is missing from this CSV; only `chosen_right` is stored).

### S7. Precision and the height-proportional bias [measured at dev altitudes, derived above]
- At 100-350 m the whole precision loss is **20 points (+1.2 / 60)**, so v0.5.6's "+2.5 to +4.8 / 60 from a static bias correction" is an upper bound of about +1.2 on the new map.
- The remaining term (about 0.0216 h from unsigned errors) matters at altitude [derived, sigma = sqrt(64 + (3h/200)^2)]: 350 m 7.6 m vs 9.6 m (0.79 sigma, tier 10); 1000 m 21.6 m vs 17.0 m (1.27 sigma, tier 6); 3000 m 65 m vs 45.6 m (1.42 sigma, tier 6). Smaller than the 3-point cap v0.5.6 projected, but still 4 points per answer above about 700 m if this term is real.
- Do not build P0 yet. Log signed east/north error, tilt and the render's nadir offset, and fit the five-parameter model of v0.5.6 Sec. 5.3 on 379 photos first. P2 (pose solve with the IMU prior) remains the insurance for altitude.

### S8. What not to spend time on now
- More decision features and a learned gate: after R2 only 4 right photos (40 points) remain on the decision side (oracle gate 850 vs 810).
- Shift-limit re-centring (item 5 of the v0.5.6 plan): its motivating photos moved with the map.
- Dense DINO as a score map, pseudo-thermal retraining: already negative or inconclusive.

---

## 5. Where the points go after R2 [measured]

810 of 1000 (48.6 / 60). The 190 missing points:

| bucket | points | /60 |
|---|---|---|
| target not in the 50 peaks (11 photos) | 110 | 6.6 |
| right top-1 with 0 inliers and z < 3 (photos 13, 38, 59, 96) | 40 | 2.4 |
| answered at 2 sigma instead of 1 | 20 | 1.2 |
| two in-K photos with a wrong top-1 and no inliers (skipped, 93 and 43) | 20 | 1.2 |

By height, per photo (bar 8 -> R2): <=125 m 4.12 -> 7.24; 125-200 m 4.50 -> 7.30; 200-300 m 9.39 -> 9.83; 300-400 m 7.04 -> 8.35. The gain is concentrated at low altitude, where the matcher finds few inliers but the top-1 is usually right.

---

## 6. Order of work

| # | step | gain / 60 (this sample) | cost | note |
|---|---|---|---|---|
| 1 | S1 + S2: bar 3, 0-inlier fallback at z >= 3.0 | +12.1 [+7.6, +16.9] | hours | in-sample, halves agree |
| 2 | Run all 379 official photos, and NUDT, with the new matcher settings; paired bootstrap against bar 8 | | ~1 hour | **do before trusting anything above** |
| 3 | Add LightGlue and RoMa-MINIMA to `MATCHERS` on the new map; re-run the K/M/weight sweep under R2 | | hours | v0.5.6's matcher result is not tested on this map |
| 4 | S5.1 and S5.2: `K_MAX = 200`; edge ZNCC as first stage through the fine stage | 0 to +4 [guess] | 0.5 day | the biggest remaining bucket |
| 5 | S3 + S4: five candidates to the fine stage, second matcher on 0-inlier top-1 | +1 to +2 [guess] | 1 day | |
| 6 | Signed-error logging, bias fit (S7), altitude emulation | at altitude | 1-2 days | no data above 350 m |

Expected value, honestly: on this map and sample 36.5 -> 44.0 -> 48.6. Cross-dataset tests earlier showed in-sample rule gains shrinking by about a third, so I would plan on +8 rather than +12 until step 2 is done, i.e. about 44-47 out of 60, with the recall ceiling (53.4 here) limiting anything beyond.

---

## 7. Caveats

- 100 photos from the 379 available; all rules are fitted on them (the bootstrap quantifies resampling noise, not selection). One map, one matcher setting, one set of photos.
- The K/M/weight sweep is flat in score because it was run at bar 8; its top-1 and target-in-M columns are the informative ones.
- For 0-inlier photos `err_sigma` is the DINO-reordered top-1's unrefined position, identical regardless of matcher.
- No data above 350 m, and the evaluation altitude mix is unknown. The Jetson time for LoFTR has not been measured (2.04 s per photo for 50 candidates on the development GPU).
- Whether the Cartosat product used here is the one used at evaluation, and whether its georeferencing error resembles it, is still a question for the organisers.
