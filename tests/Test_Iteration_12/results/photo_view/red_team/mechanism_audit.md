# Red-team audit of the photo view + pose fit mechanism (PV-X), 2026-11-19

Scope: Main_Pipeline_Headless `geoloc/photo_view.py`, `geoloc/pipeline.py`, `settings.py`, `presets/pv_x.py`, `tests/test_photo_view.py`; run files in `runs/*.csv`; problem statement PDF. No source file was changed, no server job started.
Scripts (this folder): `sensitivity_synth.py` (+ `summarize_synth.py`, results `sensitivity_synth_results.csv`, tables `sensitivity_synth_summary.txt`) and `component_checks.py` (+ `component_checks_out.txt`). Other files in this folder (`p1..p6`, `common.py`) belong to another reviewer.

## Ranked findings

### 1. (High, reproducibility) The PV-X preset is not the configuration that produced the high-altitude numbers
- `pvx_chain.sh` ran the working-332 and locked-168 sets with `--set LEVELS=[3,...,11]` and a store with levels 3-11. `presets/pv_x.py` does not set LEVELS and `settings.LEVELS = [3,4,5,6,7]`; the Jetson store holds levels 3-7 only. Nothing calls `pv_x.apply()` (grep: no caller), so `run serve` is the old behaviour unless the freeze copies the values in.
- What changes with levels 3-7 (`component_checks_out.txt`, fx 202): at 1000 / 1400 / 2000 / 3000 m the photo view draws at level 7 (3.39 m) while the photo pixel is 5.1 / 7.2 / 10.3 / 15.4 m (map 1.5 / 2.1 / 3.0 / 4.5 x finer; with levels to 11 the ratio stays 1.07-1.14). The coarse canvas is 760 m wide against a photo footprint of 1581 / 2213 / 3162 / 4743 m. So the 50.69 (working) and 49.32 (locked) are not what the shipped store + preset would give above about 600 m. The problem statement says the evaluation goes up to 3000 m; dev maximum is 2017 m.
- The NUDT "PV-X" run used `GEOMETRY = {attitude False, yaw_offset 0, scale 0.8}` and `LEVELS=[3,4,5]`; the official run used defaults. So the three PV-X scores are three geometry configs, the preset carries none of them. Not leakage, but "PV-X = one frozen thing" is not true as written.

### 2. (High, hidden coupling to the metadata noise size) The 3-sigma drop rule, not the 0.4x prior
- Real dev (`pvx_dev_s1`): 10 of 330 answered photos had the fit dropped (all above 876 m). 9 of those 10 have a median-vote error of 2.3-5.4 sigma (57-166 m), one is a 3447 m wrong answer. The moves of the fits that were kept reach 0.5: 0.79, 0.9: 2.06, 0.95: 2.36, 0.99: 2.70, max 2.86 sigma: the legitimate tail runs into the cut. The tilt-induced bias of the median vote is about height x tan(error), while sigma shrinks relative to it as the photo gets higher.
- Synthetic (exact matches + 1.2 px noise, Rajasthan intrinsics with distortion, query heading/scale chosen nearest to truth, 400 trials per cell), mean points per photo, PV-X prior, with the rule / raw fit / median only. "x1" = the audit's assumed noise 1/1/2 deg, 2 %:

| height | metadata error | fit dropped | PV-X with rule | raw fit | median only |
|---|---|---|---|---|---|
| 800 m | x1 | 3 % | 9.74 | 9.97 | 6.46 |
| 800 m | x2 | 42 % | 6.27 | 9.93 | 3.27 |
| 800 m | x3 | 63 % | 2.32 | 9.93 | 0.44 |
| 1600 m | x1 | 8 % | 9.34 | 9.94 | 5.64 |
| 1600 m | x1 + tilt bias 1/1 deg | 34 % | 7.15 | 9.94 | 3.97 |
| 1600 m | x2 | 49 % | 5.15 | 9.82 | 2.61 |
| 1600 m | x1 + tilt bias 2/2 deg | 74 % | 3.20 | 9.90 | 1.55 |

  The rule never makes it worse than the median vote and adds no wrong answers beyond what the median already has, so it is a safe default, but the gain of the fit quietly shrinks at twice the dev noise (table). Raw fit being near perfect is the idealised case (exact flat-ground geometry); see finding 3 for when it is not. Where the real noise sits: the PV-X fit's tilt corrections have std 0.55-0.74 deg on all three organiser sets, the synthetic x1 gives 0.9 and x0.5 gives 0.46, so the real tilt noise is about 0.6 of the audit's assumption (yaw / height corrections suggest about 1.0-1.3 of it).
- The 0.4x prior itself is not noise-sensitive in the synthetic: prior 0.4x / 1x / 3x differ by at most 0.3 point for tilt noise x0 to x3 and bias up to 2 deg (e.g. 800 m x2: 6.27 / 6.00 / 5.92; 1600 m x3 with rule: 0.57 / 0.42 / 0.38). Where it differs, tighter is worse: matches clustered in one part of the picture, raw fit, 800 m: 0.4x 7.83, 1x 8.53; at x3 clustered 3.70 / 6.16 / 7.30. The real-data cross-validation prefers 0.4x (other reviewer's `out_p3.txt`: 8.54 vs 7.94 for 1.0x), so the real gain of 0.4x is a regulariser against error sources the synthetic does not have, not metadata noise size. It was chosen on one region and one camera; it is safe against larger pose noise, but its value is an empirical constant, not a derived one.

### 3. (Medium-high) The fit assumes flat ground and is applied at every height
Synthetic, ground plane sloping 3 deg in a random direction (PS says flat ground but no DEM and a non-orthorectified map), noise x1, mean points with PV-X prior and rule / raw / median:
- 200 m: 6.34 / 6.34 / 9.34 (rule fires 0 %, sigma floor 8 m). Median-only is better by 3 points; at exact metadata (x0) the same: fit 6.3, median 10.0.
- 800 m: 4.96 / 3.00 / 6.40 (69 % dropped). 1600 m: 4.95 / 2.97 / 5.70 (75 % dropped).
- 1 deg slope: 200 m fit 10.0 (fine); 800 m fit 7.80 vs median 6.26; 1600 m 6.75 vs 5.76.
So below about 300 m, where the fit gains only +0.5 (9.5 -> 10.0 at x1), `PHOTO_VIEW_FROM_HEIGHT_M=0` takes all the slope risk for no gain; the fit is the part of PV-X with the highest exposure to a blind region that is hillier than Rajasthan / Maharashtra. Camera-model mismatch is milder: focal 3 % off costs under 0.5 point; k1 off by 0.03 costs 1.2-2 points at 800-1600 m (intrinsics are supplied, so low risk). Height bias of +4 % or -8 % (altitude is "above take-off", not AGL) costs nothing in the fit (9.8 / 9.76 at 800 m).

### 4. (Medium) The low-inlier answers are calibrated by one set
Bar 12 plus the agreement floor `int(round(3 * 1.5))` = 4 (Python rounds 4.5 down to 4, not 5): two queries with 4 or more inliers each near the same place are enough. Real answered photos by inlier count (PV-X runs, `pvx_*.csv`):
- official 500: 19 answers below 20 inliers, 0 wrong, +190 points.
- working 332, NUDT 300, locked 168 together: 29 answers below 20 inliers, 9 wrong (31 %), net -35 points; NUDT's 2 answers below 12 inliers are both wrong (-40). Break-even is 33 % wrong.
So the bar 12 is right for the one set where weak answers are right (cleaner low-altitude camera); on the other three it is marginally negative. The two queries of the agreement rule are the same photo and place at different headings, so they are not independent evidence. This is data-set coupling, not leakage, and was already partly seen in the README (NUDT wrong answers).

### 5. (Low, the question asked) No ground truth leakage found; metadata pose is not derived from truth
- `lat_gtruth/lon_gtruth` appear only in `cli.py` (scoring, and `move_hint` which makes the hint worse, uniform in a 5 km disc as in the PS). `lat_sc/lon_sc` become `prior_m`, which is used only in `coarse_search.score_map_for_query` (disc centre). The photo view never draws around the metadata position: it draws around each coarse candidate `peaks[k]`, and the position metadata cannot make the drawn view better than reality. PV-X runs used `--hint-jitter-m 5000` (chain script), so dev hint closeness does not apply to them (the Jetson run with hint = truth is not the PV-X run).
- The pose (yaw, pitch, roll, alt) is the organiser's own logged data (attitudes are quantised to 1e-3 rad, e.g. 1.718873 deg = 0.030 rad); the PS states it carries realistic noise, so dev and evaluation poses come from the same process. The fit corrections have the same spread on three sets (std tilt 0.57 / 0.58 / 0.74 deg, Maharashtra, Rajasthan dev, Rajasthan locked) and no trend with altitude (0.4-0.6 in every band), which supports a common noise process. The sets differ in mean (pitch correction +0.16 vs -0.45 deg, roll +0.20 vs +0.28): a per-camera boresight bias the prior (centred at zero) only partly absorbs; the synthetic shows a 2 deg bias is tolerated (800 m: raw fit 9.97) but the drop rule then fires 61-74 % of the time.
- Not leakage but in-sample: prior 0.4x, bar 12, gate 0, XoFTR for all, GEOMETRY constants (yaw 180, scale 1.04, signs -1/-1) were all chosen on the dev sets (README says so); the locked split is a second look at the same region and camera. `offline_fit.py` uses the truth only to score (legitimate offline tuning), with candidates 30-60 m from the truth, tighter than the real coarse offsets; the later pipeline runs are the real test.

### 6. (Low) Signs, north/east, rounding: no cancelling error found
- Derivation: truth = c + G(q) - G(p), north negated for y-south: matches `check_places` (`shift = [east, -north]`) and `fit_pose` (`mx = cx + G_e(pb)`, `my = cy - G_n(pb)`, residual `p[1] - n - my`). Half-pixel conventions cancel because photo and drawn view are the same size and the same convention. The per-query prior is centred on the query's heading and scale, not on the metadata (rounds=1 unaffected): in real runs `fit_dyaw` is about minus half of the winner's heading (heading -5: +2.6, +5: -2.75 deg; scale 1.05: dheight -3 % against -1 %), i.e. the total yaw and height corrections are partly set by which query won; the synthetic with the nearest query shows no loss from it (a fixed 0 / 1.0 query: 9.55 vs 9.74).
- My synthetic recovers the position with random yaw 0-360, pitch and roll +-6 deg and all offset quadrants (exact metadata: median 0.03-0.09 sigma, fit 0.10-0.30 sigma, 10.0 points). Caveat: it uses the code's own conventions for both the world and the model, so it checks consistency, not the convention itself. The convention (-1, -1, 180 deg, scale 1.04) is validated only by real scores, and the same values work on two different organiser cameras and regions (official 60.00 with 0 wrong; Rajasthan), which is strong evidence.
- Lens: `undistorted_rays` round-trips through `photo.distort` to 1e-10 px for both cameras; intrinsics are read per photo from metadata. `read_camera` hard-codes 320 x 256 (PS says 320 x 256; a different size would be wrong silently, no assert).

### 7. (Low) Camera / region constants
- No Rajasthan-specific number in the photo view code. Constants that are dev-fitted: `PHOTO_VIEW_LOCAL_NORM_PX = 15`, vignette degree 3 (hard-coded in `own_photo`), RANSAC 3 px, MIN_MAP_SHARE 0.5, the prior values. Vignette removal changes the matcher input noticeably on the Rajasthan camera (correlation with / without it: median 0.93, min 0.86; amplitude 4.5 x frame std) and little on the Maharashtra camera (0.99); it is fitted per frame from the frame itself, so it adapts, but its benefit was measured on one camera type.
- Map level is chosen from the photo's ground pixel (closest ratio) with no region dependency, but depends on the store levels (finding 1).

## Verdict
Nothing in the mechanism uses the truth or a truth-derived pose, and the hint closeness in the dev data does not enter the photo view. The ways the dev scores can overstate the evaluation are: the configuration is not the shipped one (levels, finding 1), the safety rule and the low-inlier bar are tied to the dev noise and to one set (2, 4), the fit assumes flat ground (3), and every tuned constant is in-sample (5).
