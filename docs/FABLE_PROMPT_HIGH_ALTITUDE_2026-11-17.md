# Prompt for Fable: height-dependent pipelines for the high-altitude photos (2026-11-17)

Context first, then what we need. Please read all of it before proposing anything.

## 1. What the pipeline does today (submission, Main_Pipeline_Headless)

- One photo: the thermal frame is drawn north-up on a 224 px canvas at the map level whose pixel size matches the photo (level L has a pixel of 0.3 * 2^(L/2) m: level 3 = 0.85 m, 7 = 3.4 m, 9 = 6.8 m, 10 = 9.6 m, 11 = 13.6 m; level = ceil(2 * log2(footprint / (224 * 0.3))), footprint = photo width * height / fx, clipped to the levels in the store).
- Coarse: edge-orientation ZNCC by FFT over the 5.5 km disc around the (jittered) hint, K = 50 peaks; DINOv2 ViT-S/14 (layer 8) re-orders them; LoFTR with MINIMA weights on M = 5 places per query; RANSAC (3 canvas px, scale within +-15 %, rotation within 12 deg); the answer is the place with the most inliers if it has at least 8 (the bar).
- 6 standard queries (headings -5/0/+5 x scales 1.0/1.05); if not answered, an agreement rule (two queries within 1 sigma, each with at least 3 inliers), then 12 more queries (18 in all). Optional: altered copies (polarity-flipped and sharpened) of skipped photos, answered by the bar only.
- Scoring: error within 1 sigma +10, 2 sigma +6, 4 sigma +3, 8 sigma 0, beyond 8 sigma -20, no answer 0; sigma depends on height and focal length. Answers later than 30 s count as nothing (the guard is 22 s). Jetson Orin: about 3 times slower than our laptop GPU (13 s per photo on the Jetson against 4-5 s). Only pretrained weights (MINIMA LoFTR, DINOv2 ViT-S/14); other MINIMA checkpoints (eLoFTR, LightGlue, RoMa, XoFTR) exist on our disk but are not in the submission.

## 2. The new high-altitude data

500 photos in a new region (Rajasthan, lat 27.05-27.17, lon 71.46-71.49), 320 x 256 px, fx = 202.4, `alt` 100 to 2017 m (median 709 m; 168 photos above 1000 m). Ground sampling of a photo pixel is height / fx: 0.5 m at 100 m, 4.9 m at 1000 m, 9.9 m at 2000 m. Satellite tiles 0.28-0.29 m per pixel, about 20 x 21 km each. We built a store with levels 3-11 (the old cap was 7 in the submission and 10 in the settings check) and ran the unchanged headless pipeline, hint jitter 5 km. Above about 1950 m the formula wants level 12, so those photos are clipped to level 11 and the 224 px canvas covers less than the photo footprint.

Result (score of 60 for the whole set, 0 wrong answers): 36.35. By flight height:

| height (m) | photos | mean points / 10 | answered | median level | median inliers of the answer |
|---|---|---|---|---|---|
| 0-200 | 61 | 7.3 | 98 % | 4 | 58 |
| 200-400 | 83 | 8.7 | 100 % | 6 | 63 |
| 400-600 | 49 | 7.7 | 100 % | 7 | 60 |
| 600-800 | 93 | 8.2 | 97 % | 9 | 68 |
| 800-1000 | 23 | 5.0 | 78 % | 9 | 46 |
| 1000-1200 | 41 | 4.3 | 66 % | 10 | 24 |
| 1200-1400 | 48 | 5.1 | 65 % | 10 | 14 |
| 1400-1600 | 40 | 3.2 | 48 % | 11 | 5 |
| 1600-2100 | 62 | 0.9 | 16 % | 11 | 0 |

All losses above 1000 m are skips, never wrong answers. Of the 113 skipped photos, 43 have a best candidate within 2 sigma of the truth (with 0 to 2 inliers), so for the other 70 even the candidate list is wrong. Wrong best candidates at levels 10 and 11 never had more than 3 inliers; right ones at level 11 have a median of 8 inliers and a quarter have 0.

## 3. What we tried, and what happened

- Bar 4 at map levels 9-11 (new setting): 36.35 -> 38.54, 29 photos better, none worse, 0 wrong. Plus altered copies 39.48. Found and checked on the same photos, one wrong answer appeared in a re-run, and on NUDT (another region, other camera) lowering the bar to 4 loses 4-6 points with 3 wrong answers in 60: wrong places there get 4-7 inliers. So an inlier bar is data dependent.
- Wider second pass above 1000 m (headings +-10, scales 0.7-1.3): worse (3.83 -> 3.50 points per photo, 2 wrong). The answered photos use scale 1.00 or 1.05, so it is not a scale or heading error.
- On the official dev photos (143 queries per photo, 80 photos): 6 standard queries are as good as 143; more peaks (K 100, 200), more fine candidates (M 10, 20) and a bigger second pass change nothing. The heading x scale map peaks at 0 deg and scale 1.00-1.05.
- Blur (2 px) on the official photos costs 23 points: the true place is still found (about 0.8 of photos) but the matcher gives too few inliers. The same symptom as high altitude: a loss in the fine stage, not in the search.
- The agreement rule adds 2-5 points on the official photos; on NUDT its answers are wrong 11 % of the time (3 of 27) against 0.8 % for bar answers.

## 4. What we need from you

Because the pipeline can depend on the flight height, please design a height-conditioned pipeline, and rank the ideas below (or better ones) by expected gain and by cost on the Jetson:

1. Why does LoFTR give almost no inliers at levels 9-11, and what fixes it? Candidate causes: the canvas is only 224 px (a 28 x 28 coarse grid for LoFTR), the thermal photo at 5-10 m per pixel has almost no texture, a modality gap at coarse scale, the satellite map downsampled to 13.6 m per pixel loses the structure the thermal sees. Candidate fixes: a larger canvas for high levels (448 or 640 px, same footprint, more tokens), matching at a finer map level than the footprint asks for on the central window, matching the edge-orientation maps themselves, tiling the photo into overlapping windows and voting, another MINIMA matcher.
2. A decision that does not depend on one data set's inlier level: a score normalised per map level or per photo (peak-to-background ratio of the coarse peak, DINO z-score, agreement of several queries, a calibrated score per height band). We want one rule that we can fit on one region and trust on another.
3. A table for the height bands (below 400 m, 400-800, 800-1200, above 1200): level, canvas size, query grid, K and M, matcher input scale, how the answer is decided, and the time guard. Say which parts are cheap on the Jetson.
4. How to use the large footprint at high altitude: above 1200 m a photo covers 2-3 km, close to the whole search disc. Is the 5.5 km disc and FFT search the right tool there, or should the map window be matched at once (a global registration)?
5. Where the photo clip at level 11 (above about 1950 m) hurts, and whether levels 12-13 or a different canvas would fix it.

## 5. How we will test (please keep within it)

- Every idea needs a falsification statement written before the test and a test on a set that did not pick the setting: leave-flight-out or a second region. We only have this one high-altitude set (500 photos, one region) plus NUDT (heights 197-318 m) and the official dev photos (96-347 m); say how to make a second high-altitude test set, for example from the dev or NUDT map by down-sampling photos and map together, and what it can and cannot tell us.
- A wrong answer costs 20, so any change must report wrong answers next to the score; gains below one wrong answer's cost per few hundred photos are not worth a new risk.
- Report time per photo on the Jetson-scaled clock (about 3 times our laptop), not only on a desktop GPU.
- We have a GPU server (RTX 4000 Ada, 20 GB) for dumps of every query's raw matches (replayable offline) and the headless pipeline takes `--set NAME=VALUE`; tell us which settings or code changes you need.

Numbers and files: Notion page "Second-list ideas checked in the submission pipeline" (iterations 6-11), tests/Test_Iteration_12/ (notebooks, figures), tests/Test_Iteration_7/OURS_CHECK.md. Nothing is committed.
