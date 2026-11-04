# Report: seven coarse methods, the fine-stage notebooks, and pseudo-thermal training of the fusion

Date of the runs: 2026-11-04. Project: ideaForge UAV visual geo-localisation (InterIIT Tech 15).
Everything referred to here lives in `tests/Test_Iteration_1/` (runners, stages, notebooks, results) and `train/` (the building blocks); a map of all project files is in `index.md` in the base folder. The experiments were first run in `Code_Testing/test_src/` and then moved here; file names below use the new places.

## 1. Summary

**What we did**
1. Fixed the evaluation crash of the overnight run (a CPU/GPU mix-up and a bug that made result lists unequal in length).
2. Built notebooks that push **seven different coarse-search methods** through the **same fine stage** (MINIMA/LightGlue + RANSAC) and the same decision rule, and score them: `Seven_Methods_Fine.ipynb` (compact), `Seven_Methods_Flow.ipynb` (NUDT) and `Seven_Methods_Flow_D1.ipynb` (official set), the last two in the six-part layout of the teaching notebook with many pictures.
3. Wrote an index of the whole project folder (`index.md`) so later sessions do not have to rescan.
4. Built a **pseudo-thermal training pipeline** (`run_synthetic.py`): fake thermal photos cut from the Changsha satellite map are used to train the fusion model, which is then tested on real thermal photos, including the other-city official set.
5. Read patch v0.5.5 for what it offers the fine-matching part.

**Headline results**
- On the full 5 km maps, the fused model is the best coarse method on both data sets. Top-1 within 4 sigma: NUDT test 0.30 against 0.17 for the edge-ZNCC baseline; official set 0.90 against 0.80 (Section 4).
- After the fine stage the seven methods are close to each other. In the 60-photo official run the best final score is `learned_zncc` with 43.8 of 60; `fused` has the highest accuracy within 4 sigma (0.95) but a final score of 41.2. Only `learned_zncc` and the bad `edge_mag` differ from the baseline beyond sampling noise (Section 5).
- Training the fusion on pseudo-thermal photos **did not clearly help**. Fake-only is 0.027 below real-only on NUDT and 0.005 below on the official set; fake + real is 0.013 and 0.011 above, all inside the noise (Section 6).
- **Main weakness of the synthetic data: the fake photos are far too easy** (Section 6.4). The difficulty calibration hit its upper limit and was still not close to the real photos. This probably limits what the fake data can teach, and it means the synthetic result says little about the idea in general.

## 2. Background

**Pipeline** (`src/`, see `index.md` for the file map): photo -> north-up 224 x 224 canvas -> edge-direction cells (8 directions per 4 x 4 pixel cell) -> FFT correlation over the 5 km disc around a prior -> 25 peaks -> per peak, keypoint matching with SuperPoint + LightGlue (MINIMA weights) and RANSAC -> the candidate with most inliers wins (plus its shift) -> answer only if it has at least 12 inliers.

**Scoring** (`src/header/scoring.py`): error in sigma units, with `sigma = sqrt(8^2 + (3 * GSD)^2)` metres and `GSD = height / focal length`. Within 1 sigma +10, 2 sigma +6, 4 sigma +3, 8 sigma 0, worse -20; a skipped photo 0. Final score = 60 x (total points / (10 x photos)).

**Data**
- NUDT Thermal-UAV (Changsha): thermal photos from 160-330 m, 8115 train / 1425 valid / 2350 test photos, 0.3 m map of 12.3 x 8.3 km (102 km2). Camera focal length 1562 px. Used for training and validation.
- Official development set: 500 thermal photos (320 x 256, focal length about 200 px) and the `small_sat_area` map of another city. Used as the "does it generalise" check.

**Zoom levels.** Map levels 3 to 7 have 0.85 to 3.4 m pixels. NUDT photos only use levels 3 and 4; the official photos use 3 to 7.

## 3. The seven methods

All seven share the same search window and grid, so their score maps can be compared cell by cell. `pick_peaks` (from `src`) turns any score map into 25 peaks.

| # | method | what it is |
|---|---|---|
| 1 | `edge_zncc` | baseline: edge-direction cells, masked ZNCC |
| 2 | `edge_cosine` | same cells, cosine score |
| 3 | `edge_mag` | total edge strength only, direction ignored |
| 4 | `edge_zncc_w` | edge ZNCC after pre-whitening (flattens repeating city grids) |
| 5 | `mind_zncc` | MIND self-similarity cells: never compares brightness between thermal and satellite |
| 6 | `learned_zncc` | two small CNNs trained so that the correlation of their cells peaks at the truth (F4) |
| 7 | `fused` | best learned fusion of the six maps (robust z-score per map, then a monotone model with 90 parameters), picked on NUDT valid |

Not part of any method: **DINO**. `src/coarse_search/dino_hint.py` exists (tile similarity as an optional hint, off by default) but is not used by the overnight run, the notebooks or the fusion. The fusion has exactly the six maps above. The patch v0.5.5 mentions dense DINO as an idea; it was not implemented.

## 4. Overnight run on real photos (earlier work, results re-read from `results/overnight/`)

**Geometry found by trial** (`geometry.json`): yaw sign +1, offset 0, scale 0.8; 75 % of 24 check photos land within 4 sigma with it.
**F4 learned features:** hit rate 0.93 against 0.91 for plain edges on held-out pairs.
**Training data for the fusion:** 787 real train photos kept (the candidate dump was cut by its time budget) and 250 real valid photos that pick the winner.

### 4.1 Fusion ablation (valid photos, truth ranked against a hard candidate list; top-1)

| maps | linear | monotone | gated |
|---|---|---|---|
| edge_zncc | | 0.124 | 0.124 |
| + edge_cosine + edge_mag | 0.204 | 0.200 | 0.200 |
| + edge_zncc_w | 0.208 | 0.176 | 0.204 |
| edge_zncc + cosine + mag + mind_zncc | 0.200 | 0.176 | 0.192 |
| + whitened + mind | 0.200 | 0.172 | 0.188 |
| edge_zncc + learned_zncc | 0.192 | 0.192 | 0.200 |
| **all six maps** | 0.248 | **0.252** | 0.236 |

All six maps together win. Each added family helps; the gain of the six-map model over edge_zncc alone is +0.128 (95 % interval +0.084 to +0.176).

### 4.2 Ranker (R1) on the hard candidates
Fused score alone 0.252, small listwise ranker 0.536 (+0.284, interval +0.208 to +0.356). This is on coarse candidate features only. The ranker is not used in `stage_evaluate` or in the notebooks.

### 4.3 Full-map check, NUDT test photos (150 photos, never used for choosing)

| method | within 1 sigma | 2 sigma | 4 sigma | best of 25 within 4 sigma | vs edge_zncc (4 sigma) |
|---|---|---|---|---|---|
| edge_zncc | 0.053 | 0.100 | 0.173 | 0.493 | 0 |
| edge_cosine | 0.033 | 0.080 | 0.153 | 0.480 | -0.020 |
| edge_mag | 0.000 | 0.000 | 0.007 | 0.100 | -0.167 |
| edge_zncc_w | 0.020 | 0.033 | 0.087 | 0.293 | -0.087 |
| mind_zncc | 0.040 | 0.060 | 0.120 | 0.373 | -0.053 |
| learned_zncc | 0.073 | 0.193 | 0.260 | 0.553 | +0.087 [+0.020, +0.153] |
| fused (monotone) | 0.073 | 0.160 | 0.300 | 0.633 | +0.127 [+0.067, +0.193] |
| fused (linear) | 0.087 | 0.193 | 0.307 | 0.627 | +0.133 [+0.080, +0.200] |

### 4.4 Full-map check, official set (379 photos, another city)

| method | within 1 sigma | 2 sigma | 4 sigma | best of 25 within 4 sigma | vs edge_zncc (4 sigma) |
|---|---|---|---|---|---|
| edge_zncc | 0.554 | 0.789 | 0.805 | 0.900 | 0 |
| edge_cosine | 0.525 | 0.763 | 0.778 | 0.889 | -0.026 |
| edge_mag | 0.201 | 0.301 | 0.306 | 0.562 | -0.499 |
| edge_zncc_w | 0.536 | 0.755 | 0.768 | 0.855 | -0.037 |
| mind_zncc | 0.557 | 0.792 | 0.815 | 0.942 | +0.011 [-0.024, +0.045] |
| learned_zncc | 0.528 | 0.839 | 0.863 | 0.953 | +0.058 [+0.029, +0.087] |
| fused (monotone) | 0.628 | 0.889 | 0.900 | 0.963 | +0.095 [+0.066, +0.127] |
| fused (linear) | 0.615 | 0.879 | 0.894 | 0.955 | +0.090 [+0.061, +0.121] |

**Reading.** The gain of the fused model over the baseline holds on a different city, camera and map (interval entirely above 0), so on the coarse score maps it is not just fitted to Changsha. The absolute NUDT numbers are low because the check uses the full 5 km map with a simulated random prior.

## 5. The seven-method notebooks: coarse + fine stage

### 5.1 What the notebooks do
Per photo: build the canvas, compute all maps once, make seven score maps, take 25 peaks each, run `second_stage` (matcher on all 25 candidates), `final_answer` and `decide`, then compare with the truth. Tables: coarse top-1 and the ceiling "right place anywhere in the 25", fine accuracy at 1/2/4 sigma, median error, share answered, contest score out of 60, a paired bootstrap of points against `edge_zncc`, a "lost / rescued / broke" count, error curves, a gated-score curve and a few good/bad example pictures. Per-photo rows are saved as `seven_methods_*_per_photo.csv`.

The layout follows the teaching notebook (Header, Image Processing, Coarse Searching, Fine Matching, Result Thresholding, Evaluation), but all functions come from `src` instead of being defined inline, and the code follows `HUMAN_CODE_STYLE.md`.

### 5.2 Checks that passed on 6-photo test runs
- `src.coarse_search.search.correlation_maps` and method 1 give the same ZNCC map (difference 0.0).
- `src.result_thresholding.pipeline.localise_photo` (the whole original chain for one photo) gives the same first coarse peak, error and inliers as the notebook for method 1.
- Both flow notebooks ran without errors on 6 photos. The numbers of those runs mean nothing statistically.

### 5.3 Results of the 60-photo official run (provided by the user)

| method | final score (/60) | fine within 4 sigma | coarse best of 25 within 4 sigma | answered | points vs edge_zncc (per photo) | 95 % interval |
|---|---|---|---|---|---|---|
| learned_zncc | 43.8 | 0.933 | 0.967 | 0.783 | +0.700 | +0.200 .. +1.367 |
| mind_zncc | 41.8 | 0.917 | 0.950 | 0.783 | +0.367 | -0.067 .. +0.933 |
| edge_cosine | 41.6 | 0.883 | 0.933 | 0.767 | +0.333 | 0.000 .. +0.833 |
| fused | 41.2 | 0.950 | 0.950 | 0.767 | +0.267 | -0.367 .. +0.967 |
| edge_zncc | 39.6 | 0.850 | 0.917 | 0.733 | 0 | |
| edge_zncc_w | 38.6 | 0.800 | 0.833 | 0.717 | -0.167 | -0.833 .. +0.333 |
| edge_mag | 24.0 | 0.500 | 0.500 | 0.467 | -2.600 | -3.867 .. -1.300 |

**Interpretation.**
- `fused` has the highest accuracy after fine matching but only the fourth final score. The contest score rewards exactness (+10 within 1 sigma, +6 within 2 sigma) and the answered share, not just "within 4 sigma", and these differ by one or two photos out of 60.
- At 60 photos almost nothing is separable: the standard error of an accuracy near 0.9 is about 0.04. Only `learned_zncc` (interval above 0) and `edge_mag` (clearly worse) differ from the baseline.
- The fusion was trained to put the truth first in a candidate list. The fine stage checks all 25 candidates, so rank 1 matters less than having the truth somewhere in the 25 (the ceiling column), where fused (0.950) and learned (0.967) are tied. This mismatch between the training objective and the pipeline is the most likely reason that the coarse advantage of fused largely disappears after fine matching. It is a hypothesis; the coarse top-1 table of the same run was not available to confirm it.
- I do not know which fusion model (older real-only or the newer mixed) that run used.

### 5.4 Problems found and fixed along the way
- **Evaluation crash of the overnight run.** Fusion models were loaded on the CPU while the maps were on the GPU, and the error handler appended a second entry to result lists that were already complete for that photo, which later broke the paired bootstrap (`assert len(a) == len(b)`). Fixed in `overnight/full_eval.py`: models are moved to the device, results are collected per photo and appended once, the first traceback is logged.
- **MINIMA weights path.** `settings.MINIMA_FILE` points to a non-existing `test_src/dataset/...`; the weights are in `Code_Testing/dataset/NUDT/checkpoint/MINIMA/`. The notebooks set the path before `load_matcher()`.
- **Two copies of `settings`.** Writing `from src.header import settings` loads a second module that does not see `nudt_data.configure()` or the other changes made to `header.settings`, so the zoom-level table crashed. The notebook now uses `from header import settings`; functions can keep the `src.` prefix. This is documented in `index.md`.
- **Misleading demo of brightness invariance.** The old cell compared a query with its own negative (1.000 by construction) and with the true tile (0.464 against 0.446 for a far tile: a per-cell cosine has a floor near 0.45). It is replaced in the D1 notebook by the real correlation: the true place ranks first of all positions in the circle (ZNCC 0.142 against a median of 0.0), the negative query gives the same score, and a photo of another place does not stand out (0.023 at this place). That was one example photo. The same replacement was not applied to the NUDT flow notebook.
- **Fused score scale.** The fused chart in section 3.3 shows `sigmoid(S / 10)`, which saturates near 1, so peaks 2 to 5 look close to peak 1 although the underlying gap may be large. It is not comparable with the raw ZNCC chart. Ratios of the fused score are not a usable confidence; the pipeline decides with inlier counts, so results are not affected.

## 6. Training the fusion on pseudo-thermal photos

### 6.1 Idea and why
The project has no thermal training photos of the mission area. A thermal photo shares its edges with a satellite picture but not its brightness. So we cut crops from the satellite map, keep their structure and change what the grey values mean, then use them like real photos to train the fusion on many places.

### 6.2 What was built (`run_synthetic.py`, `overnight/pseudo_thermal.py`, `overnight/synthetic_stage.py`, `overnight/synthetic_report.py`)
- **Generator.** Read a crop straight from the 0.3 m map at the level's pixel size. Blur it, give every region (grey band x flat/textured) a random temperature, mix in part of the original brightness (sometimes inverted), add hot spots, gain drift, small "changed" patches, noise and stripes; apply a small random rotation and scale error; build a rotated 5:4 footprint mask that fills 71-100 % of the canvas; normalise like the real pipeline (percentile stretch + CLAHE). Output looks like a `render_query` canvas. A `strength` parameter scales how different it is from the map.
- **Where the crops come from.** Only places with map data, never within 500 m of a NUDT valid or test photo, spread evenly over three scene types by texture (flat/mixed/dense). 74 km2 allowed, 16 km2 kept free. 1 km blocks serve as the "flight" groups, so a validation split keeps whole blocks apart.
- **Pipeline.** Fake canvas -> same score maps (`compute_maps`) -> same candidate sets (`build_candidate_set`), saved in the same shard format as the real data; then the existing `run_ablation` for the fusion.
- **Levels.** Levels 3-7 with weights 0.25/0.25/0.20/0.15/0.15 because the official set uses all five.
- **Three fusion models compared on real photos:** `real_only` (earlier run), `synth` (fake only), `mixed` (fake + real train photos, each real photo counted twice).
- **Difficulty calibration.** Try strength 0.5, 1.0, 1.5, 2.0 on 30 fake photos each and keep the one whose per-map "truth beats every other candidate" rates are closest to the real training photos.

### 6.3 Run statistics
- 2400 fake photos in 27 minutes (0.64 s per photo; stopped by the 25-minute dump budget), from 88 blocks: level 3: 610, level 4: 598, level 5: 474, level 6: 379, level 7: 339.
- Training sets: synth 2400 photos; mixed 3974 (2400 + 787 x 2 real). Model chosen on the 250 real valid photos.
- Whole run 43.7 minutes (setup 1.6, dump 26.9, fusion 9.7, NUDT evaluate 2.6, official 2.8).
- Real photos used for the check: 150 NUDT test photos and 188 official photos (not 200: the height-band sampling returned fewer).

### 6.4 The fake photos are too easy (important)

Share of photos where the truth beats every other candidate (per map: edge_zncc, edge_cosine, edge_mag, edge_zncc_w, mind_zncc, learned_zncc):

| | edge_zncc | edge_cosine | edge_mag | edge_zncc_w | mind_zncc | learned_zncc |
|---|---|---|---|---|---|---|
| real train photos | 0.18 | 0.15 | 0.01 | 0.09 | 0.13 | 0.27 |
| fake, strength 0.5 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 |
| fake, strength 2.0 (chosen) | 0.97 | 0.90 | 0.13 | 0.97 | 0.90 | 0.93 |
| all 2400 fake photos | 0.94 | 0.87 | 0.24 | 0.91 | 0.90 | 0.90 |

The calibration picked the largest strength it tried (2.0) and the gap to the real photos was still 0.66, so the search ran into the edge of its range. As expected, a fake query made from the same map it is matched against is much easier than a real thermal photo. The generator output looks plausible (building blocks, roads, region contrast; see `pseudo_samples.png`) but it keeps the geometry of the map exactly, so the correlation finds it almost every time. The calibration used only 30 photos per strength, so the strength numbers themselves are noisy. We did not try strengths above 2.0 or other ways of making the task harder (for example adding more unrelated structure, or using a different map date).

### 6.5 Results on real photos (top-1 on the full 5 km maps, within 4 sigma)

NUDT test, 150 photos:

| model | 1 sigma | 2 sigma | 4 sigma | best of 25 | vs edge_zncc (95 % interval) |
|---|---|---|---|---|---|
| edge_zncc | 0.760 | 0.793 | 0.813 | 0.900 | 0 |
| learned_zncc | 0.760 | 0.807 | 0.820 | 0.940 | +0.007 [-0.040, +0.053] |
| fused: real_only | 0.813 | 0.853 | 0.860 | 0.940 | +0.047 [+0.013, +0.087] |
| fused: synth | 0.780 | 0.820 | 0.833 | 0.953 | +0.020 [-0.013, +0.053] |
| fused: mixed | 0.820 | 0.860 | 0.873 | 0.960 | +0.060 [+0.020, +0.107] |

Official set (other city), 188 photos:

| model | 1 sigma | 2 sigma | 4 sigma | best of 25 | vs edge_zncc (95 % interval) |
|---|---|---|---|---|---|
| edge_zncc | 0.559 | 0.777 | 0.803 | 0.888 | 0 |
| learned_zncc | 0.511 | 0.830 | 0.862 | 0.952 | +0.059 [+0.021, +0.101] |
| fused: real_only | 0.617 | 0.867 | 0.888 | 0.952 | +0.085 [+0.048, +0.128] |
| fused: synth | 0.622 | 0.851 | 0.883 | 0.936 | +0.080 [+0.043, +0.122] |
| fused: mixed | 0.617 | 0.867 | 0.899 | 0.952 | +0.096 [+0.059, +0.144] |

(The other single maps are in `results/synthetic/report.md`.) These numbers differ from Section 4 because the photo samples differ (150 and 188 against 150 and 379) and the priors are drawn per sample.

### 6.6 What this does and does not show
- The two new models were chosen on the real valid photos (`synth`: monotone, valid top-1 0.196; `mixed`: gated, 0.252, equal to the real-only model's 0.252).
- The mixed model is nominally best on both sets (+0.013 and +0.011 over real-only), but those gaps are far smaller than the intervals and were measured on 150 and 188 photos only. **We cannot say the pseudo-thermal data helped.**
- The fake-only model is only slightly behind (-0.027, -0.005) despite being trained on queries that are much easier than real ones. This suggests that the fusion mostly learns a general way to weight the six maps, which is stable across data sources. It does not show that the fake data is good.
- The result does not test the idea properly because of the difficulty gap in 6.4. A fairer test needs harder fake photos.

## 7. What patch v0.5.5 says about fine matching

(From `Documents/Our_Architecture/Patch_v0.5.5_Breakthroughs.tex`, section on `fine_matching`. All gains are the authors' guesses, to be replaced by measurements.)

The patch splits the photos of the last official 60-photo run into: answered within 1 sigma (0.60), right place but between 1 and 4 sigma (0.28, the **precision gap**, worth about 9 points) and place never found (0.12, the **recall gap**, about 7 points). It says precision is the bigger and cheaper gap and needs geometry, which is the fine stage.

| ID | idea | guess |
|---|---|---|
| P1 | Register at the photo's own pixel size (second canvas up to 448-640 px) for the top 1-3 candidates | +2 to +4 |
| P2 | Solve the camera pose from the matches (planar PnP) with the IMU as a prior; removes the IMU tilt error, which matters above 300 m | +2 to +5 |
| P3 | Re-render with the solved pose and match again, 2 rounds | +1 to +2 with P4 |
| P4 | Dense matcher (MINIMA RoMa/LoFTR) or ECC/Lucas-Kanade when there are few inliers | see P3 |
| P5 | Fine-tune LightGlue on NUDT pairs | 0 to +2, risky |
| P6 | Mutual information as a modality-free re-ranking feature | feeds R1 |
| R1 / R2 | Learn to rank candidates using fine-stage features; answer by expected points per photo | +1 to +3 |

None of P1 to P6, R1 or R2 is in the code. The README of `test_src` lists P1-P4 as not covered because MINIMA had not run; `lightglue` is installed and the notebooks now run it. This is consistent with Section 5: the seven coarse methods end up close after fine matching and the shift refinement is the same for all, so the remaining loss is mainly precision.

## 8. Limitations and open points
1. **Sample sizes are small.** 60 photos for the fine-stage table, 150 and 188 for the synthetic comparison. Most differences between fused, learned, mind and cosine are inside the noise.
2. **The fusion objective does not match the pipeline.** It optimises top-1 in a candidate list; the fine stage uses all 25. A recall-at-25 objective or a ranker with fine-stage features is the natural fix.
3. **Fake photos are too easy** (Section 6.4).
4. **The geometry** (heading, scale 0.8 for NUDT) was found on a few training photos, not known. The official set uses the original pipeline's settings (yaw offset 180, scale 1.04).
5. **Prior and map.** The 5 km prior is simulated from the truth. The real evaluation uses other imagery, so absolute numbers will move.
6. **Not verified by me:** the 60-photo run was done by the user; I did not see its coarse table or which fusion model it used. The synthetic run (43.7 minutes) was also run by the user; I only read its outputs. The notebooks were only executed by me on 6 photos.
7. **DINO** is in neither the notebooks nor the fusion.

## 9. Suggested next steps (in order)
1. Run the 500 official photos through the seven-method notebook (about an hour at about 8.5 s per photo for all seven methods) and add a paired comparison of `fused` against `learned_zncc`.
2. Look at the coarse top-1 table in the notebook next to the fine table to confirm or reject the objective-mismatch explanation.
3. Make the fake photos harder and rerun (larger strength range, or queries made from a map copy with different imagery or added structure), then check that the "truth wins" rates come close to the real ones before training.
4. Train the fusion for recall in the top 25.
5. Fine stage: error-anatomy report (X0), then P1 (register at the photo's resolution for the top 3 candidates) and P2 (pose solve), measured on the official photos.
6. DINO as an extra map or dense family, if retraining the fusion is acceptable.

## 10. Files and how to reproduce

| what | where (inside the repo) |
|---|---|
| Notebook, NUDT | `tests/Test_Iteration_1/notebooks/Seven_Methods_Flow.ipynb` (writes `results/overnight/seven_methods_per_photo.csv`, `seven_methods_summary.csv`) |
| Notebook, official set | `tests/Test_Iteration_1/notebooks/Seven_Methods_Flow_D1.ipynb` (writes `results/overnight/seven_methods_official_*.csv`) |
| Overnight run as a notebook | `tests/Test_Iteration_1/notebooks/Overnight_Run.ipynb` |
| Runners | `tests/Test_Iteration_1/run_overnight.py`, `run_synthetic.py` |
| Stages (one file per stage) | `tests/Test_Iteration_1/stages/` |
| Building blocks | `train/features/`, `train/maps/`, `train/datasets/`, `train/models/`, `train/checks/`, `train/config.py` |
| Results of the real-photo run | `tests/Test_Iteration_1/results/overnight/` (`report.md`, `state.json`, `models/`) |
| Results of the synthetic run | `tests/Test_Iteration_1/results/synthetic/` (`report.md`, `state.json`, `pseudo_samples.png`, `models/{synth,mixed}/`) |
| Project index | `index.md` in the base folder |

Commands (from `tests/Test_Iteration_1`):
```
python run_synthetic.py                    # about 45 minutes; --dump-minutes sets the fake-photo budget
python run_synthetic.py --stages fusion evaluate official report     # only these stages
python run_overnight.py --smoke            # quick check of the overnight stages on a tiny synthetic world
```
The D1 notebook uses the synthetic **mixed** fusion model (section 3.1); change the three lines there that read `ablation_mixed` to go back to the real-only model (`results/overnight/state.json`, key `fusion_best`).
