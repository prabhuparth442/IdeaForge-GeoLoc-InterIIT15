# src: how the pipeline is cut into sections

## The idea of the whole project in one paragraph

A drone flies at some height and takes a **thermal photo** pointing roughly down. We are told only that it is within 5 km of a given point. We must say **where on a satellite map** the photo was taken, to within a few tens of metres. The method: (1) convert the photo into a north-up picture and describe both the picture and the map by **edge directions**, which survive the change from heat to sunlight; (2) slide the photo's description over the map everywhere in the 5 km circle (done quickly with the FFT) and keep the 25 best places; (3) check each place with a proper keypoint matcher and count how many landmarks agree; (4) answer with the best-supported place, or **skip** the photo if the support is too weak, because a wrong answer costs −20 while a skip costs 0.

`python run.py` (one level up) calls the six packages below in this order. Each folder has its own `README.md` (what to run, inputs, outputs, how it works) and a `concepts/` folder of teaching notes that explain the ideas step by step with worked numbers.

| # | folder | one-line job | concept notes |
|---|---|---|---|
| 1 | [`header/`](header/README.md) | constants, scoring rule, photo and map readers, test-set sampling | 3 |
| 2 | [`image_processing/`](image_processing/README.md) | thermal photo → north-up canvas; map → zoom levels → edge-direction features | 7 |
| 3 | [`coarse_search/`](coarse_search/README.md) | FFT correlation over the whole 5 km disc → 25 candidates | 4 |
| 3b | [`semantic_rerank/`](semantic_rerank/README.md) | optional: DINO re-orders the best 50 coarse places before the fine check (not yet in the pipeline) | 0 |
| 4 | [`fine_matching/`](fine_matching/README.md) | SuperPoint + LightGlue + RANSAC check of each candidate → inliers, shifts | 3 |
| 5 | [`result_thresholding/`](result_thresholding/README.md) | pick the answer, answer-or-skip, single-photo entry point | 3 |
| 6 | [`evaluation/`](evaluation/README.md) | simulate priors, score with the contest rule, experiments | 6 |

## Whole data flow

```
photo (thermal, h, yaw, pitch, roll) ─► [2] canvas 224², valid mask, 8×56×56 edge features
map GeoTIFF ─► [2] 5 zoom levels ─► edge features per level          (built once per run)
prior (x,y) ─► [3] window of map features ─► FFT correlation (cosine, ZNCC) ─► 25 peaks
peaks ─► [4] per peak: map tile ─► keypoints ─► pairs ─► RANSAC ─► inliers, shift
[5] re-rank by inliers, apply shift, answer iff inliers ≥ 12  ─► (x, y, answer?)
[6] errors / σ ─► points (+10…−20) ─► score out of 60
```

## One photo, with numbers (a mental walk-through)

Take a photo from 200 m height, camera focal length 200 pixels:

1. **Ground sample distance** `GSD = 200/200 = 1.0` m per pixel; `σ = √(8² + 3²) = 8.54` m. The photo gets map **level 5** (1.7 m per pixel). *(header, image_processing)*
2. The canvas is a **224 × 224** north-up picture of that photo; its middle is the point directly below the drone. Edge directions are summed over 4 × 4-pixel cells: the fingerprint is **8 × 56 × 56** numbers. *(image_processing)*
3. The prior says "within 5 km": at level 5 that is about **1.7 million** candidate positions. The FFT scores all of them in one go and we keep the 25 best places, at least 380 m apart. *(coarse_search)*
4. For each of the 25, a 224-pixel map tile is cut, keypoints are matched, and RANSAC counts agreeing pairs. Say candidate 7 has 31 inliers and a shift of (+4.1 m, −2.2 m). *(fine_matching)*
5. 31 ≥ 12, so we answer with candidate 7's position plus its shift. *(result_thresholding)*
6. The truth is 11 m away, i.e. `11/8.54 = 1.29σ`: **+6 points**. A wrong answer 80 m away (`9.4σ`) would have been **−20**. *(evaluation, header)*

## Glossary

| term | meaning |
|---|---|
| **prior** | the rough position we are given: the drone is within 5 km of it |
| **nadir point** | the ground point directly below the drone; the canvas centre |
| **GSD** | ground sample distance: ground metres covered by one photo pixel, `height / focal length` |
| **σ (sigma)** | the allowed error for a photo, `√(8² + (3·GSD)²)` metres; all errors are judged in σ units |
| **canvas** | the photo re-drawn north-up, 224 × 224 pixels, ready to compare with the map |
| **level** | one of five map zoom levels (3 to 7), pixel size `0.3·2^(level/2)` m |
| **cell** | a 4 × 4-pixel block; features are stored per cell |
| **edge-direction features** | for each cell, how much edge there is in each of 8 directions |
| **cosine / ZNCC** | two ways of scoring "how alike are two patterns"; ZNCC subtracts the average first |
| **heat map** | the grid of scores for every position of the photo on the map |
| **peak** | a distinct high spot of the heat map; a candidate position |
| **keypoint** | a landmark pixel (corner, junction) found by SuperPoint |
| **inlier** | a keypoint pair that agrees with the best-fitting transform |
| **RANSAC** | fit by repeated random guesses, counting how many pairs agree |
| **answer / skip** | report a position, or give 0 points by saying nothing |
| **AUC** | the chance that a right photo gets a higher confidence number than a wrong one |

## Suggested reading order for a newcomer

1. `header/concepts/scoring_rule.md` (what we are optimising) → `geographic_coordinates_and_map_reading.md` → `stratified_sampling_and_seeds.md`.
2. `image_processing/concepts/` in the README's order: camera → distortion → rotations → thermal normalisation → resampling → zoom levels → edge features.
3. `coarse_search/concepts/`: cosine versus ZNCC → FFT → peaks → (optional) DINO.
4. `fine_matching/concepts/`: keypoint matcher → RANSAC → candidate refinement.
5. `result_thresholding/concepts/`: choosing → answer or skip → one photo at a time.
6. `evaluation/concepts/` to see how each claim was tested.

Each note ends with common mistakes and a few questions to check yourself.

## Running anything

Package imports assume `src` is on the path: `PYTHONPATH=src python -c "..."` from the repository root, or use `python run.py`, which sets this itself. Only `header/`, parts of `image_processing/`, `coarse_search/correlation.py`, `fine_matching/ransac.py`, `result_thresholding/decision.py` and some of `evaluation/` run without the dataset; each README marks which and gives a command you can paste.

## Input checks

Every function starts with `assert` lines that state what it accepts: array shapes and dtypes, value ranges, units, matching lengths, known keys. A wrong input stops at the function that received it, with a message naming the expected form (for example `valid must be a bool array shaped like canvas`), instead of producing wrong numbers several blocks later. Read the asserts at the top of a function as its input contract; the folder READMEs list the same constraints in prose. Asserts are skipped if Python runs with `-O`, so do not run the pipeline that way.

## Status of the code (honest list)

* Verified on synthetic data in a test environment: scoring, FFT correlation versus the slow loop, RANSAC, decision logic, prior sampling, the AUC and honest-threshold statistics, and an end-to-end smoke test of the non-neural path.
* **Not yet run on the real dataset or with the neural parts:** MINIMA stage, DINO hint, real-map store build. Treat reported numbers from the original notebook as the reference until `python run.py` has been reproduced.
* Notebook plots were not ported; the map store is rebuilt on every run.
* `auc()` in `evaluation/metrics.py` now gives tied values the average rank (before, ties could inflate the AUC).
* `SatelliteMap` has no metres-to-latitude/longitude function yet, so answers are in map metres.
