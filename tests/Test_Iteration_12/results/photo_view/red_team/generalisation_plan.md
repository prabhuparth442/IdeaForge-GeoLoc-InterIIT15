# PV-X generalisation: red-team stress plan (offline analysis, no server jobs run)

Date 2026-11-19. Scope: would PV-X (XoFTR everywhere + photo view + bar 12 + pose fit priors x0.4 + agreement rule) hold on photos and maps it was never tuned on? Everything below comes from the per-photo CSVs in `results/photo_view/runs/`, the dataset metadata and the existing diagnostics. Analysis scripts: `red_team/analysis/generalisation/` (load.py joins the CSVs to the metadata; boot.py, s2.py, s3.py, mc.py produce the tables). Score unit = points per 60 (6 x mean points per photo). Nothing in source was changed.

Short verdict: the evidence that PV-X works is strong on the data it has seen (gain +10 on 500 Rajasthan photos, CI [+8.5, +11.9]), but the data is far less independent than 1300 photos suggest, the wrong-answer protection is a fixed inlier bar of 12 whose margin shrinks with height, and nothing has been run above 2017 m or on terrain with relief. The claim "works to 3000 m" is not supported; the claim "works to about 2000 m on flat terrain similar to Rajasthan" is.

---------------------------------------------------------------------------------------------------

## 1. How the real evaluation can differ from every development set

What is actually known (sources: `Documents/Problem_Statement/23_Oct_ideaForge_Briefing_Organised.md`, `REQUIREMENTS_AND_LIMITS.md`, `MAIN_PIPELINE.md`; verbal, the revised statement is the authority): thermal LWIR only, Boson 320x256 (same camera model, per-camera calibration differs), day and night, multiple seasons, no gimbal (viewing angle varies), 100 m to 3000 m (planned testing to about 2000 m), 100 x 100 km region different from the dev region, 0.3 m Airbus RGB+NIR map acquired on other dates/seasons, hint within 5 km uniform, metadata yaw/pitch/roll/alt with "realistic measurement noise", altitude "above take-off" or AGL (unconfirmed), no DEM, wrong answer -20, 30 s per photo on an Orin NX 15 W.

What the development data cover (computed from metadata):

| property | new Rajasthan (332 + 168 locked) | old official (500) | NUDT headless (300) |
|---|---|---|---|
| heights | 100-2017 m, plateaus (only 4 photos between 1600 and 1800 m, 32 of 58 above 1800 m within 2000-2017 m) | 100-351 m | 212-320 m, one cruise height (~300 m) |
| ground extent of all photos | 2.8 x 13.6 km strip | about 1.5 x 1.5 km | 4 flights over Changsha |
| tilt (hypot pitch, roll) | 0.2-22 deg, median 7 | 0.1-18 deg | none (attitude switched off, nadir ortho) |
| terrain relief | flat (DEM at takeoff constant, 191 m) | flat (4 m) | city / village, flat |
| camera | fx 202.4 | fx 200.3 | other camera, other geometry preset |
| time | unknown (no timestamp in metadata) | unknown | day and night flights |
| map | one scene group, one acquisition | PNEO3 acquisition 2026-01-14 | Changsha ref map |

Per difference, the component at risk and why:

| # | difference | component at risk | why / evidence |
|---|---|---|---|
| D1 | Heights 2000-3000 m (dev stops at 2017 m) | coarse level choice, photo view level, inlier bar, precision | `level_for_height` gives level 13 at 3000 m, the dev store stops at level 11 (3.05 km tile vs a 4.75 km footprint at 3000 m: the canvas crops about 36 % of the photo width). The PV-X preset (`presets/pv_x.py`) sets none of LEVELS, GEOMETRY, so the code default LEVELS=[3..7] would clip every photo above about 470 m to level 7; the dev runs passed `--set LEVELS=[3..11]` on the command line. True inliers fall about e^-1.2 per km (section 2), the false-match level rises. |
| D2 | Terrain relief and AGL vs above-take-off | photo view drawing (height), pose-fit height prior, scale queries | All three dev sets are flat. `PHOTO_VIEW_PRIOR_HEIGHT = 0.008` (0.8 % of height). A 300 m hill under a 2000 m photo is a 15 % scale error, 19 times the prior. Queries only cover scale 1.0 / 1.05, RANSAC allows +-15 %. Untested. |
| D3 | Another region (100 x 100 km): terrain type, look-alikes | verification bar, agreement rule, coarse search | Look-alike density differs 6-60x between dev sets (below). The dev region is a 13 km strip; at most 2-3 independent scenes above 1600 m. |
| D4 | Another camera calibration | GEOMETRY constants (yaw_offset 180, scale 1.04, roll/pitch signs), fit priors | The constants were fitted on Rajasthan and official; NUDT needs a different preset (attitude False, yaw_offset 0, scale 0.8, LEVELS [3,4,5]) found by hand with ground truth. Nothing automatic finds them on a new camera. Same camera model per the briefing, so probably transferable, but unproven. |
| D5 | Thermal contrast: season, time of day, night, polarity, haze | XoFTR matching, local normalisation, bar | NUDT night flights have median inliers 120 vs 190 by day and the lowest set score (city7 night 48.2). Dev Rajasthan/official time of day unknown, "not measured: cloud, haze, night, seasonal change" (REQUIREMENTS section 3.7). |
| D6 | Map age and acquisition season | coarse (edge ZNCC), XoFTR | Dev map and photos are close in time; real map is from other dates. Not testable with current data except by perturbation. |
| D7 | Metadata noise larger than assumed | pose fit priors x0.4, drawn map, bar | The priors are 2.5x tighter than the audit's noise (1/1/2 deg, 2 %), which itself is an assumption that reproduced one dataset. Dev fit corrections are already larger than the priors (median |dyaw| 1.0-1.7 deg vs prior 0.8; |dheight| 1.1-2.4 % vs 0.8 %). Sibling red-team (`sensitivity_synth_summary.txt`) shows at 1600 m: x2 noise halves points (9.34 to 5.15), x3 gives 12.5 % wrong. |
| D8 | Tilt beyond 22 deg | photo view geometry, fit | Dev max 21.9 deg. Precision already drops: fraction of 10-point answers 0.98 (tilt < 3) to 0.60 (15-25 deg). |
| D9 | Search disc over a denser/larger map | wrong-answer rate | candidates per photo: M_FINE 5 x 6 queries (up to 30 places, 90 with second pass); per-candidate false pass 3 % at bar 12 (section 2). More dissimilar-but-textured places in the disc raise it linearly. |
| D10 | Speed on the Jetson | time guard (22 s), query count | PV-X mean 2.1-3.8 s on the server vs 1.28 s for the base (official). Jetson base was 13.06 s (about 10x server). A crude 10x scaling puts PV-X at 22-39 s mean per photo, over the 22 s guard on 36-100 % of photos: answers then depend on hardware speed. Not measured (README: "Jetson timing" open). |
| D11 | Number of effective test photos | the headline numbers | 1300 photos, but see section 2.5: independence is low. |

---------------------------------------------------------------------------------------------------

## 2. Stratified analysis of existing per-photo CSVs

PV-X runs used: working 332 `pv3_dev_x0`, official `pv3_official_x0`, NUDT `pv3_nudt_x0`, locked 168 `pvx_test_t0` (all seed 0). Base used for "gain": XoFTR from 800 m as in the README (working 39.70, locked 40.75, official 52.69, NUDT 48.52). Total score 1300 photos. All per-photo bootstrap CIs resample photos only.

### 2.1 Gain and wrong answers by height band (all four sets pooled)

| height | n | PV-X | base | gain | answered PV-X / base | wrong PV-X / base | share of 10-point answers | median error / sigma | median inliers |
|---|---|---|---|---|---|---|---|---|---|
| < 150 m | 327 | 58.9 | 50.0 | +8.9 | 1.00 / 0.88 | 0 / 1 | 0.95 | 0.39 | 69 |
| 150-250 m | 91 | 52.7 | 48.4 | +4.4 | 0.95 / 0.97 | 0 / 2 | 0.78 | 0.42 | 288 |
| 250-350 m | 467 | 55.1 | 52.1 | +3.0 | 0.97 / 0.96 | 6 / 10 | 0.92 | 0.36 | 252 |
| 350-600 m | 108 | 57.6 | 47.1 | +10.6 | 1.00 / 0.95 | 0 / 0 | 0.91 | 0.61 | 261 |
| 600-1000 m | 116 | 54.2 | 46.8 | +7.4 | 0.99 / 0.96 | 0 / 0 | 0.78 | 0.65 | 374 |
| 1000-1300 m | 61 | 45.3 | 37.8 | +7.6 | 0.98 / 0.84 | 2 / 0 | 0.69 | 0.80 | 140 |
| 1300-1600 m | 68 | 50.7 | 26.0 | +24.7 | 0.97 / 0.65 | 0 / 0 | 0.74 | 0.64 | 112 |
| 1600-2100 m | 62 | 35.5 | 20.2 | +15.3 | 0.98 / 0.66 | 0 / 0 | 0.26 | 1.38 | 54 |

Bootstrap gain (PV-X minus base), per height group: below 350 m +5.34 [3.84, 6.88]; 350-1000 +8.92 [6.67, 11.25]; 1000-1600 +16.6 [11.5, 21.5]; 1600-2100 +15.3 [11.5, 19.4]. Per set: working +10.99 [8.86, 13.18], locked +8.57 [5.64, 11.39], official +7.31 [5.62, 9.12], NUDT +3.56 [0.44, 7.08]. Rajasthan pooled +10.18 [8.50, 11.90].

Reading:
* The gain does not vanish with height; at 1000+ m it grows (recall: base answers fall from 0.96 to 0.65 while PV-X holds 0.97-0.98). This is a recall gain.
* There is a precision cliff, not an answering cliff: the share of full-point answers drops from 0.64-0.78 (1000-1700 m) to 0.30 (1750-2000) and 0.22 (2000-2250); median error / sigma 0.64 to 1.38. Score by 250 m band: 56.5 (250-500), 57.0, 50.1, 42.6, 52.7, 47.8, 39.1 (1750-2000), 32.8 (2000-2250). The 1600-1800 m range has only 4 photos, so the break is located between 1500 and 1800 m only loosely.
* Where gain is worst: NUDT (+3.6, the lower CI bound +0.4), working 200-250 m (-6.0 but n = 8), locked < 150 m and 250-350 m (0.00, n = 9 and 24). By photo, 29 of 332 working photos are worse than the base (mostly a +10 answer replaced by +6 or +3); the "worse" share peaks at 25 % in 1600-1800 m (n = 4, meaningless) and 5-8 % elsewhere above 350 m.
* Wrong answers are not removed by PV-X in Rajasthan; they are introduced: base 0 wrong in 500 photos, PV-X 2 (one in working at 1077 m with 12 inliers, one in locked at 1055 m with 13 inliers). On NUDT it removes 12 -> 6.

### 2.2 By pitch / roll / tilt

| tilt (deg) | n | gain | wrong | share of 10-pt answers | median error / sigma |
|---|---|---|---|---|---|
| 0-3 | 91 | +9.3 | 0 | 0.98 | 0.39 |
| 3-5 | 216 | +10.0 | 1 | 0.93 | 0.48 |
| 5-8 | 225 | +4.9 | 0 | 0.88 | 0.52 |
| 8-11 | 187 | +9.2 | 1 | 0.75 | 0.58 |
| 11-15 | 169 | +10.9 | 0 | 0.89 | 0.46 |
| 15-25 | 112 | +9.4 | 0 | 0.60 | 0.75 |

Rajasthan only, 1000 m and up (n = 191): tilt 0-4: gain +26.3, precision 0.90; 4-7: +7.8, 0.63; 7-10: +11.1, 0.29 (n = 14); 10-25: +16.1, 0.38. Gain is positive in every tilt stratum, but precision falls with tilt, and at 10 deg and above at 1000+ m only 38 % of answers are full-point. No tilt data above 22 deg and none at all for NUDT (attitude off). By signed roll the weakest Rajasthan stratum is -8 to -3 deg (n = 164, gain only +1.9, and it holds both Rajasthan wrong answers); signed pitch shows no pattern. Treat tilt as "no gain loss seen, precision cost, range untested above 22 deg".

### 2.3 By inlier count of the winner (the only wrong-answer guard)

Pooled over 1300 photos:

| inliers | answers | wrong | wrong rate | note |
|---|---|---|---|---|
| 12 (bar) | 5 | 1 | 20 % | mean points 2.4 vs base 12.0 |
| 13-15 | 13 | 4 | 31 % | |
| 16-20 | 27 | 1 | 3.7 % | |
| 21-30 | 58 | 2 | 3.4 % | both NUDT |
| 31-50 | 116 | 0 | 0 | |
| > 50 | 1056 | 0 | 0 | |

Both Rajasthan wrong answers sit at 12 and 13 inliers (1055 and 1077 m, one with second pass). All answers with 12-15 inliers pooled: 5 wrong of 18, expected value about zero or negative at -20. Counterfactual bar (skip answers below the bar, recomputed from the CSVs):

| set | bar 12 | 16 | 20 | 25 | 30 | wrong at 12 / 16 / 20 |
|---|---|---|---|---|---|---|
| working 332 | 50.69 | 50.95 | 50.40 | 49.46 | 48.83 | 1 / 0 / 0 |
| locked 168 | 49.32 | 49.93 | 49.57 | 48.89 | 48.54 | 1 / 0 / 0 |
| NUDT 300 | 52.08 | 52.76 | 51.76 | 51.96 | 51.24 | 6 / 3 / 2 |
| official 500 | 59.76 | 58.92 | 57.84 | 56.16 | 54.36 | 0 / 0 / 0 |

Bar 16 is better on three of four sets (+0.26, +0.61, +0.68) and worse on official (-0.84; official is a ceiling set at 60.00 and low, sharp 100 m photos have few but correct inliers). Pooled over sets weighted by size the difference between bar 12 and 16 is close to zero, so the bar choice is not settled by existing data; what is certain is that the bar is the only barrier, that the first wrong answers appear 0-1 inlier above it in Rajasthan, and that it does not scale with height or terrain.

Second pass photos (hard cases; 3 locked, 7 working, 22 NUDT): locked second-pass answered 1 of 3 and was the wrong one; working 4 of 7 answered, all right but 0.29 precision; NUDT second-pass answers 4, none wrong (base had 4 wrong there).

### 2.4 By coarse stage strength (rank proxy)

Coarse rank of the truth is not in the PV-X CSVs; the pipeline's coarse z (peak confidence) and the earlier coarse-confidence files are used.
* PV-X coarse z by height (Rajasthan): median 3.06 (< 300 m), 2.47 (600-1000), 1.83 (1000-1300), 1.50 (1300-1600), 1.61 (1800-2100); p10 falls from 2.3 to 0.83-1.2, minimum 0.14. Linear fit about -1.1 per km: it reaches 0.57 at 2500 m and about 0 at 3000 m. Quintiles of z, Rajasthan: lowest quintile (z < 1.6, n = 100) has both Rajasthan-wrong answers (2 wrong of 97 answers) and the lowest score 41.6; the other four quintiles have 0 wrong and 47-55. Gain holds in every quintile (+7 to +13), so PV-X rescues weak-coarse photos but they are where wrong answers live.
* Earlier edge-ZNCC only coarse (`coarse_confidence`, working 332): truth in the top-5 of at least one query: 1.00 (< 600 m), 0.94 (600-1000), 0.93 (1000-1300), 0.80 (1300-1600), 0.50 (1600-2100). The final pipeline answers 98 % of the 1600-2100 photos, so DINO re-rank and the six queries recover most of this, but it shows that the candidate list is where recall is lost; PV-X does not change the coarse stage and cannot help when the truth is not in the M = 5 places per query.
* LoFTR look-alike pool (`coarse_evidence`, canvas path, wrong candidates beyond 4 sigma; 28 779 candidates): mean inliers at a wrong candidate: NUDT 0.19, official 0.03, Rajasthan 0.00; share at 8 or more: NUDT 0.58 %, official 0.02 %, Rajasthan 0 %. NUDT (urban/village, repeated structure) has 6-60x more spurious matches, and PV-X's wrong rate follows that order: NUDT 6 / 282 answered (2.1 %), Rajasthan 2 / 495 (0.4 %; one-sided 95 % bound about 1.5 %), official 0 / 500. NUDT wrong answers by flight: 5 of 6 in the village flights, 1 in city; city7 night is the lowest set (48.2), 4 flights in total.

### 2.5 How independent are the 1300 photos?

All 500 Rajasthan photos lie in a 2.8 x 13.6 km strip; the locked photos are a median 172 m (min 11 m) from the nearest working photo (cells differ but are neighbours). Photo footprints: about 340 m at 100-350 m, 1.1 km at 350-1000 m, 2.2 km at 1000-1600 m, 3.3 km at 1600-2100 m. Greedy count of non-overlapping scenes (centres at least one footprint apart): 6, 8, 5 and 1 for the four bands (2 at half-footprint spacing above 1600 m). So the 62 photos above 1600 m with 0 wrong are essentially one 3 km patch of desert seen from a handful of positions: the effective sample for "0 wrong at 1600-2100 m" is 1-3 scenes, and the 95 % bound on the scene-level wrong rate is far above 3/n of photos. The official 500 photos cover about 1.5 x 1.5 km. NUDT is 4 flights. Any constant that was tuned on this (priors x0.4, bar 12, gate 0, XoFTR from 0) has been checked against, at best, a few dozen independent scenes.

### 2.6 Extrapolation to 2000-3000 m

Measured trend (Rajasthan, pipeline winner inliers, all heights above 600 m): ln(inliers) falls 1.0 to 1.4 per km (fit from 0 m: 1.01; from 600 m: 1.41; from 1000 m: 1.13); the photo-view diagnostic (XoFTR true place, 130 photos): 1.19 per km. Median inliers at the true place: 322 (< 300 m), 371 (600-1000), 137 (1000-1300), 109 (1300-1600), 53 (1800-2100); extrapolated median 47-60 at 2000 m, 23-36 at 2500 m, 11-22 at 3000 m, i.e. at the bar of 12. Sigma grows 8.5 -> 30.8 m over the dataset (fit 5.33 + 0.0122 h: 42 m at 3000 m) but the median error grows faster (7 -> 42 m), so full-point answers fall to 22-30 % by 2000 m. Linear score trend from 1000 m: 38 at 2000 m, 32 at 2500, 26 at 3000 (95 % bands 33-43 at 2000 m, 11-43 at 3000 m: the fit cannot say).
False-place level (the 520 null draws of the photo-view diagnostic, XoFTR photo view, four random wrong places 3 km away, `photo_view_full.csv`): the null mean rises 2.8 (1000-1300 m) -> 4.1 -> 5.7 -> 4.6 (1800-2100 m), the share of null draws at 5 or more inliers 0.32 -> 0.49 -> 0.56, the share at 12 or more 3.0-3.5 % in every band (16 of 520), max 20, p99 13. So at the bar 12 each wrong candidate has about a 3 % chance of passing, and with up to 30 candidate places per photo (more with the second pass) some wrong place passes in a large share of photos; today it never wins because the true place has 50-100+ inliers. The margin (true p10 22-30 vs null p99 12-15 at 1000-2100 m) is already thin above 1800 m.

Stress model (mc.py; my construction, not a measurement): true inliers = empirical 1600-2100 m XoFTR true pool scaled by e^(-s (h - 1950) / 1000), null counts = empirical null pool scaled by 1 + g (h - 1950) / 1000, N wrong candidates per photo, winner = most inliers, answer if at least the bar. It reproduces the observed 0 wrong in 58 at 1800+ m (predicts 1.0 % at N = 10). Predicted wrong answers per photo, N = 10:

| height | central (s 1.19, g 0.5) bar 12 | bar 20 | optimistic (s 0.6, g 0.2) bar 12 | pessimistic (s 1.8, g 1.0) bar 12 |
|---|---|---|---|---|
| 1950 m | 1.1 % | 0.1 % | 1.0 % | 1.0 % |
| 2250 m | 6.2 % | 1.1 % | 2.8 % | 14 % |
| 2500 m | 17 % | 3.9 % | 5.0 % | 45 % |
| 2750 m | 36 % | 10 % | 8.9 % | 82 % |
| 3000 m | 60 % | 21 % | 14 % | 97 % |

N = 30 candidates roughly doubles these at 2250 m (central 11 % at bar 12, 3 % at bar 20). Expected value per photo (+8 right, -20 wrong, ignoring precision) crosses zero around 2500-2750 m in the central model at bar 12 and around 2750-3000 at bar 20. So: yes, a cliff is plausible, not a slope, and it is in the wrong-answer rate (the dangerous direction), starting between about 2200 and 2700 m unless the bar is raised with height; the model's weak points are the assumed null growth with height and iid random candidates (look-alikes are worse than random places, so the null is optimistic; the Poisson resampling of an already-integer pool is pessimistic). It is an argument for an experiment, not a result.

### 2.7 Seeds and run-to-run noise

* Determinism: runs with the same seed are bit-identical. pv3_official_x0 vs pvv_official_v0, pv3_nudt_x0 vs pvv_nudt_v0 and v40s0, pvv_dev_x2s0 vs x3s0, pv3_dev_b12a vs pv2_dev_b12na (the V2 and V3 code copies): 0 status differences, 0 point differences, 0 differing inlier counts, error identical to 1e-6 m over 1132 photos. So RANSAC (`cv2.setRNGSeed(7)` per query) and the GPU are deterministic here; there is no measurable run-to-run noise. "Noise" is only the hint-jitter seed and the photo sample.
* Seed 0 vs 1 (hint jitter): official 0 photos differ (60.00 / 60.00); working 1 status and 2 point differences (50.69 -> 50.26, one more wrong answer: 2 vs 1); NUDT 4 status and 6 point differences (52.08 -> 51.68, 8 vs 6 wrong, 1 photo errs 3.5 km). Paired bootstrap of the difference: working +0.43 [0.00, 1.23], NUDT +0.40 [-0.96, 2.16]. Two seeds only; the spread is the photos where a different disc center changes which look-alike gets in.
* Resolution of the score: one wrong answer costs 0.36 (working), 0.71 (locked), 0.40 (NUDT), 0.24 (official) points when it replaces a skip, 0.54 / 1.07 / 0.60 / 0.36 when it replaces a perfect answer. Standard error of one set score: 0.98 (working), 1.48 (locked), 1.65 (NUDT). Therefore a difference below 0.5 points is one or two photos flipping and is noise on every set, and a difference of 1 point is not significant on NUDT or locked unless paired.
* Paired comparisons with 95 % CI (positive = first config better; all in-sample): PV-X2 vs PV-X, NUDT s0 +1.48 [+0.16, +3.20] (4 photos up, 1 down), NUDT s1 +2.28 [+0.56, +4.40], official -0.12 [-0.36, 0.00], working -0.05 [-0.16, 0.00] (the "second opinion" rule helps only where wrong answers exist, i.e. one set with 4-8 wrong answers); limit 40 on official -0.96 [-1.68, -0.36]; PV-X (bar 12) vs bar 20 on working +1.99 [+0.67, +3.22], on NUDT -0.20 [-2.24, +1.64]; XoFTR everywhere vs LoFTR below 800 m: working +1.79 [+0.89, +2.75], official +2.76 [+1.81, +3.82], NUDT -0.20 [-1.80, +1.28]. The README's NUDT "+0.2 from agreement rule off" type differences (52.28 vs 52.48) are noise. XoFTR-everywhere and bar 12 are supported on two sets and neutral on the third; PV-X2 is supported by one set only.

---------------------------------------------------------------------------------------------------

## 3. Ranked decisive held-out experiments (GPU server `workstation@100.105.160.107`, work dir `/mnt/e/IdeaForge/work`, copy `Main_Pipeline_V3` as in `pvx_chain.sh`, run through `tools/run_capped.sh` with the 6 GB / 8 core cap; nothing run here)

Timing basis: PV-X took 2.1 s per photo (Rajasthan), 3.1 (official), 3.8 (NUDT) on the server while shared; photo counts below use 3 s as the planning figure.

### Experiment 1 (highest value): false-answer null test (wrong-map-area / look-alike null)
Question: what is the false-answer rate when the true place is not in the search disc, by height band and by set? Every `match` is a wrong answer by construction, so this measures the verification (bar 12 + agreement + fit) directly, which is the only guard against -20, and it is the quantity the extrapolation says rises with height.
Design: reuse the images and store of each set, rewrite `metadata.csv` so that the hint (lat_sc, lon_sc) is a point whose 5.5 km disc excludes the truth: (a) same-region hard null: Rajasthan photos with the hint moved 8-10 km along the strip (needs map there; check the store extent first) so the disc contains only desert look-alikes at the same terrain type; (b) cross-region null: Rajasthan photos against the official store and the NUDT store, official and NUDT photos against the Rajasthan store, hints at random points of the foreign map with a full disc; (c) height-matched: report by the true photo height bands used above. Run with the frozen PV-X switches, `--mode full`, log winner inliers for all photos so the whole bar curve (8, 12, 16, 20, 25) comes from one pass.
Size: Rajasthan 500 photos x 2 nulls, official 500, NUDT 300 against Rajasthan = about 1800 runs, about 1.5 GPU-hours (3 s each).
Confirm the claim "PV-X does not make false answers": at bar 12, false answers at most 1 % overall and at most 2 % in the 1000+ m bands (a 0 of 190 outcome gives a 95 % upper bound of 1.6 %), and false-winner inliers separated from the dev true-place distribution (AUC 0.98 or more) with the median false winner under 8.
Refute: false answers above 3 % in any band of 100 or more photos, any false winner with 20 or more inliers, or a rising false-answer rate with height; then the bar must scale with height (candidate: bar = max(12, 0.4 x median inliers at wrong candidates ...) set from the null, not from the true place) before any freeze. Decision rule written before running: the bar for the freeze is the smallest value giving 0 false answers in this null in every height band.

### Experiment 2: altitude extrapolation above 2017 m (the cliff test)
Question: do true inliers, precision and the false level follow the extrapolation up to 3000 m, and does the code path (levels 12-13, canvas crop at level 11) work?
Design, three arms on the working 332 + locked 168 photos: (A) emulated height: take each real photo at 1000-2017 m (n = 191), downscale the image by k = 1.25, 1.5, 2.0 about its principal point (zero padding to 320 x 256, which also keeps the camera model), multiply metadata altitude by k; this reproduces the ground-resolution loss and the lower contrast of a real 2500-4000 m photo from real thermal data, but not the 2-4x larger footprint; (B) synthetic render from the map at 2000, 2500, 3000 m with the `Synthetic_Geometry_Audit` renderer (exact and noisy metadata, 100 photos per height), which tests levels 12-13, the 4.75 km footprint and the canvas crop but has no thermal domain gap (upper bound); (C) build LEVELS [3..13] in the store (small: levels 12-13 are 19 and 27 m per pixel) and run arm A and B both with LEVELS up to 11 (current dev) and up to 13. Use the frozen PV-X, plus the diagnostic columns (inliers at the winner, null from four wrong places as in `Map_In_Photo_View`).
Size: arm A 191 x 3 k = 570 runs, arm B 300, two level settings, about 2.5 GPU-hours.
Confirm: emulated 2500 m: median true inliers at least 25 (extrapolation 23-36) and null p99 at most 15, wrong at most 2 % at bar 12; emulated 3000 m: median at least 14, 80 % of photos at or above 12, wrong at most 3 %; levels 12-13 do not lower arm B precision. Refute: wrong answers above 5 % at emulated 2500 m, or median true inliers below 2x the null p99 at emulated 3000 m (the cliff is real; then the product limits its answers by evidence ("inliers below bar" is allowed, altitude-only rejection is gaming) and a higher bar/level scheme above 2000 m is required), or arm A with levels 11 vs 13 shows the canvas crop costs more than 5 points (then LEVELS must be extended in the preset).
Caveat to state with the result: arm A understates footprint, arm B overstates ease; only agreement of both directions justifies a claim up to 3000 m. Real evidence above 2017 m cannot be created from existing data.

### Experiment 3: metadata, terrain and image perturbation sweep (D2, D5, D6, D7, D8)
Question: how far can the metadata be wrong and the image degraded before PV-X loses its gain or starts to give wrong answers, and are the x0.4 priors and the "scale" constant a cause?
Design (copy datasets with symlinked images and edited `metadata.csv` / filtered images; PV-X vs base XoFTR-800 vs PV-X with priors x1.0 and x2.5 and with the fit off): all 500 Rajasthan photos plus 150 official and 100 NUDT. Arms: metadata noise at 1x, 2x, 3x of the audit (yaw 2 deg, pitch/roll 1 deg, alt 2 %; the sibling synthetic result already predicts a cliff at 2-3x at 1600 m); constant yaw bias +-5 deg; altitude scale bias -20 %, -10 %, +10 %, +20 % (terrain relief or AGL vs above-take-off) and a height-dependent one (+/-300 m); tilt bias 2 deg; image arms: Gaussian blur 1.0 and 2.0 px, contrast stretch gamma 0.5 / 2, additive noise sigma 5 % of range, polarity inversion, 10 % crop, 12-bit quantisation. Seeds 0 only.
Size: about 17 arms; PV-X and base on every arm, the two prior variants and fit-off only on the noise arms 1x/2x/3x and the +-10/20 % scale arms (7 arms). On a stratified 350 photo subset (heights near 100/500/1000/1500/2000 m, 70 each) that is (17 x 2 + 7 x 3) x 350 = about 19 000 photo-runs, about 16 GPU-hours; halve it by running the image arms on 150 photos.
Confirm: at 2x noise PV-X keeps at least half its nominal gain and wrong answers at most 1 %; scale bias of 10 % (hill of 200 m at 2000 m) costs at most 3 points; image arms except polarity cost at most 3 points; priors x0.4 within 1 point of x1.0 at 2x noise. Refute: any arm where PV-X is worse than the base (gain negative), wrong above 3 %, or priors x0.4 losing at least 2 points to x1.0 at 2x noise (then the prior must scale with an estimated noise, or be widened) , or scale bias 10 % producing more than 3 % wrong (then add scale queries 0.8-1.2 above 1000 m, which the old note in MAIN_PIPELINE item 8 already found necessary for +20 % height noise).

### Experiment 4: fresh held-out real photos and multi-seed protocol (variance and PV-X2 decision)
Question: what is PV-X's wrong-answer rate and score on real photos that were never used, and how much of any difference is seed noise?
Design: (a) the NUDT test and train flights (the manifest `tests/Review_20261113_codex/results/nudt_photo_manifest_codex.csv` lists 2350 photos, 3 more flights than the 300 dev photos use: city, village, day, night, with prior-seen/unseen labels) run unchanged with the NUDT preset and PV-X and PV-X2, seeds 0-2; (b) Rajasthan and official: seeds 2, 3, 4 added to the existing 0 and 1, PV-X and PV-X2; (c) one more scene group if any: the official sat_imagery has 4 scenes, check for map area with no photos (an area to cut synthetic photos is not needed).
Size: (a) 2350 x 3 seeds x 2 configs x 3.8 s = 15 GPU-hours, reduce to 1 seed x 2 configs = 5 GPU-hours plus seeds on a 600-photo subset; (b) 800 x 3 x 2 x 3 s = 4 GPU-hours. About 9 GPU-hours.
Confirm: held-out NUDT wrong rate at most 2.5 % per answer at bar 12 and a PV-X gain over base of at least +2 with paired CI above 0; five-seed range at most 0.6 points on Rajasthan; PV-X2 gain of at least +0.8 on held-out NUDT with paired CI above 0 and no loss on Rajasthan (otherwise PV-X2 is dropped). Refute: PV-X wrong rate above 4 % on held-out NUDT, seed range above 1 point on any set, or PV-X2 gain below its CI.

### Experiment 5: time-budget and speed-scaled replay (D10)
Question: does PV-X still score once the per-query time grows to Jetson levels? The score depends on hardware speed through the 22 s guard.
Design: server, PV-X with `TIME_BUDGET_S = 22 / r` for r = 1, 2.4, 5, 10 (equivalent to a board r times slower), on 332 Rajasthan, 300 NUDT and 500 official; record no_match counts caused by the guard and the score. Alternatively run on the Jetson if access exists (not available here).
Size: 4 x 1132 x 3 s = about 4 GPU-hours.
Confirm: at r = 5 the score stays within 2 points of r = 1 and the guard causes at most 5 % no_match; at r = 10 (the crude Jetson ratio) the score stays above the base's. Refute: at r = 5 score loss above 5 points or any wrong answer created by an early stop; then PV-X needs a speed budget (fewer queries, XoFTR only for the top candidates) before it can replace the submitted pipeline.

Order of execution if time is short: 1 (cheapest and most dangerous), then 2, then the +-10-20 % scale arm and the 2x/3x noise arm of 3, then 4(a), then 5. Experiments 1 and 3 settle the wrong-answer claim, 2 the 3000 m claim, 4 the "not tuned on" claim, 5 deployability.

---------------------------------------------------------------------------------------------------

## 4. Findings that need no experiment (act or at least note)

1. `presets/pv_x.py` omits LEVELS and GEOMETRY; the dev runs passed them on the command line (LEVELS [3..11]; NUDT geometry). Anyone applying the preset without them gets levels up to 7. The "freeze" is therefore incomplete as a definition of what was tested.
2. The 5.5 km disc and `M_FINE 5` x 6 queries (second pass 18) mean up to 30-90 wrong places are verified per photo; at a per-candidate false-pass rate of 3 % (bar 12, XoFTR photo view null) the bar is the sole defence and it is a constant.
3. The two Rajasthan wrong answers are at 12 and 13 inliers; the NUDT wrong answers at 14-24 inliers. Raising the bar to 16 removes both Rajasthan ones and half of NUDT's with net score change +0.26 (working), +0.61 (locked), +0.68 (NUDT) and -0.84 (official, a ceiling set). This is in-sample; do not tune it before Experiment 1.
4. Real headline claim to use externally: gain +10 on Rajasthan (CI 8.5-11.9), +3.6 on NUDT (CI 0.4-7.1), 2 and 6 wrong answers; none of it covers relief, other terrain at 1500+ m, heights above 2017 m or different camera constants.
