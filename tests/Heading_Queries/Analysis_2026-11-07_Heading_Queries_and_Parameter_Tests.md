# Heading queries, parameter sweeps and the 5.5 km disc: what was done on 6-7 Nov 2026, what worked and what did not

7 Nov 2026. Covers every notebook and run of the last two days: `tests/Parameter_Test` (the old Stage_Evaluation, run on the new Cartosat map), the recreated `Stage_Evaluation.ipynb`, `tests/Heading_Queries`, and the changes to `Single_Image_Run`. Every number below was read from the saved files listed in Section 9 (paths are relative to the repo folder `IdeaForge-GeoLoc-InterIIT15/`).

Tags: **[measured]** computed from a saved file; **[derived]** arithmetic from measured numbers; **[in-sample]** a setting chosen on the same photos it is scored on; **[not tested]** an idea nobody has run. Scores are out of 60 (`final_score`: 60 x points / (10 x photos)); one photo of 100 at +10 is 0.6 of a point.

---

## 1. The short version

1. **Heading queries work.** Drawing every photo several times with the yaw moved by a few degrees, and letting the best candidate across all drawings answer, lifted the score from **41.3 to 49.7 / 60** (9 headings, 273 photos, 3 seeds, **[measured]**). The gain has the same sign in all 3 seeds (+7.3, +9.3, +9.5). The true place is among the candidates for **100 %** of the photos instead of 87 %.
2. **Scale queries add to it.** The cheapest set that matches the 9 headings is **3 headings x scales {1.0, 1.05}** (6 queries): **49.5 / 60** against 49.7 for 9 headings, 4.2 s per photo against 6.2 s. The paired difference between the two is -0.24 [-1.76, +1.30] (indistinguishable).
3. **The gain is at low altitude.** Flights at 100-148 m go from 29.6 to 44.8 (9 headings); flights at 280-351 m gain only 2.4.
4. **The price is a higher false-flag risk.** The share of photos where a *wrong* candidate reaches the answer bar goes from 0 % (plain) to 1.8 % (9 headings) and 2.6 % (6 queries). One wrong answer (-20) occurred in 273 photos in both: photo 347, a 100 m flight, exactly 8 inliers (the bar), 189 sigma away.
5. **What did not work:** the heading penalty (lost 0.1 to 0.8 points at every setting), confidence 0.1 as a default (false flags, see 4.5), shifting the matcher bar below 4, and every "free" hyperparameter of the old sweeps (K, M, DINO weight, gamma, border, margin were flat to within 1 point).
6. **What is NOT yet measured:** the heading queries inside `Stage_Evaluation.ipynb` (only a 4-photo smoke test ran), anything on the real evaluation map or photos, the Jetson speed, and the fixed-offset idea (Section 6.1).

---

## 2. What was built

| Piece | File | State |
|---|---|---|
| One query = render, maps, coarse, DINO, fine (reusable) | `single_photo_pipeline.py`: `candidates_for_query`, `decide_answer`; new settings `dino_weight`, `geometry`, `loftr_border_px`, `loftr_margin` | done, `run()` uses it |
| Several queries per photo | `heading_query_pipeline.py`: `HeadingQueryPipeline` (`run_queries`, `answer_for`, `run`) and the new `from_pipeline(base, headings, scale_factors, k_peaks, m_fine, coarse)` that reuses an already loaded pipeline | done, tested |
| Query list, geometry per query, pooling, heading penalty | `src/coarse_search/heading_queries.py` | done, 4 tests in `tests/pipeline/test_heading_queries.py` |
| Result tables per photo/variant | `train/checks/heading_results.py` | done |
| LoFTR border and margin | `src/fine_matching/loftr_matcher.py` (`border_px` erodes the valid mask at fit time, `margin` pads the canvas and the map piece, multiple of 4) | done, 6 tests in `tests/pipeline/test_loftr_matcher.py` |
| Search disc 5 km to **5.5 km**, map margin 6000 m | `src/header/settings.py` (DISC_RADIUS 5500), `src/header/map_source.py` (MARGIN_M 6000); the simulated hint `random_prior` stays within 5000 m | done, READMEs updated |
| Data-setup guide | `dataset/README.md`, link from the top `README.md` | done |
| Old Stage_Evaluation moved | `tests/Parameter_Test/Parameter_Test.ipynb` + its data (analysis md, CSVs, `full_runs/`) | done; has the Part 12 save-everything cell |
| Original Stage_Evaluation recreated | `Stage_Evaluation.ipynb` (root), writes `tests/Stage_Evaluation/` | done; now also has **Part 6.7 heading queries** and a points-lost-per-stage figure |
| Heading test notebook | `tests/Heading_Queries/Heading_Queries.ipynb` (3 seeds side by side, one photo at a time, OOM retry, stops if more than 5 % of photos fail) | done |
| Single_Image_Run knobs | DINO_WEIGHT, SCALE, YAW_OFFSET, HEIGHT_OFFSET, LOFTR_BORDER_PX, LOFTR_MARGIN | done; **heading queries are not in Single_Image_Run yet** |

Setting changes worth remembering: Stage_Evaluation and Heading_Queries now use LoFTR confidence **0.21** and coarse threshold 0.21 (never 0.1 as a default).

---

## 3. The heading idea and why

`Documents/Our_Architecture/Patch_v0.5.6_Recall_Geometry_and_Bias.tex` found [as read from the document]: 12 of 100 photos are lost at the coarse stage because the yaw metadata is off by up to +-10 degrees (and the scale by about 5 % at 350 m); the decision stage is already near its ceiling (oracle 805 / 1000); sweeps of decision parameters cannot move the score; a lateral pointing/roll bias of about 0.98 degrees exists; and any answer bar must be re-derived for each matcher setting. The coarse search tolerates roughly 3 degrees of heading error, so the photo is drawn at yaw + heading for headings -10 ... +10 in steps of 2.5, each drawing is its own query (coarse, DINO, LoFTR), the candidates are pooled, and the winner is the one with the most `inliers + gamma x z` (no heading prior). Nothing is re-run to score a variant: 3, 5 and 9 headings, scales and penalties are all scored from the same stored per-query candidates.

---

## 4. Results

### 4.1 Which runs are valid

| Run (folder in `tests/Heading_Queries/results/`) | Photos | Valid? | Why |
|---|---|---|---|
| `..._n30_heads9_scales1_K50_M5_conf0.20_bar8` | 81 distinct (3 seeds x 30 draws) | **valid, small** | no failures; plain 43.9, 3 / 5 / 9 headings 51.4 / 53.6 / 55.1, 0 wrong answers |
| `..._n150_heads9_scales1_K50_M5_conf0.20_bar8` | 138 of 273 | **INVALID** | 135 photos died with CUDA out of memory (another program held 3.8 of 6 GB); the 138 survivors are mostly high flights, so plain scores 50.7 there instead of 41.3 |
| `..._n150_heads9_scales1_K50_M10_conf0.21_bar8` | 273 | **valid (main result)** | `failures.csv` empty |
| `..._n150_heads3_scales3_K50_M5_conf0.20_bar8` (in `Heading_Queries_results_v2/`) | 273 | valid | scale test with scales 0.95, 1.0, 1.05 |
| `..._n150_heads3_scales2_K50_M10_conf0.21_bar8` | 273 | **valid (controlled)** | scales 1.0 and 1.05, same K/M/confidence as the main result. The variant names inside still say "3 scales"; it has 2 |

`Heading_Queries_results_v2.zip` is the same data as `Heading_Queries_results.zip` (byte-identical). The 273 photos are the distinct photos of 3 seeds x 150 (the seeds overlap); the seeds pick the same number of photos from each height group.

### 4.2 Main result: 9 headings, M 10, confidence 0.21 [measured]

| Variant | Queries | Score | Answered | Wrong answers | Target in candidates | Wrong candidate reaches bar | s / photo |
|---|---|---|---|---|---|---|---|
| plain | 1 | 41.3 | 70.7 % | 0 | 87.2 % | 0.0 % | 0.69 |
| 3 headings (-5, 0, +5) | 3 | 46.5 | 81.0 % | 1 | 98.9 % | 0.7 % | 2.07 |
| 5 headings (step 5) | 5 | 47.8 | 83.2 % | 1 | 99.6 % | 1.5 % | 3.45 |
| **9 headings** | 9 | **49.7** | 86.4 % | 1 | 100 % | 1.8 % | 6.22 |
| 9 headings, penalty 0.25 / 0.5 / 1 per degree | 9 | 49.6 / 49.3 / 49.0 | 86.1 / 85.7 / 84.2 % | 1 / 1 / 0 | 100 % | 1.8 % | 6.22 |

Per seed (plain -> 9 headings): seed 11 43.6 -> 50.9 (+7.3), seed 22 40.9 -> 50.2 (+9.3), seed 33 41.8 -> 51.3 (+9.5). The 3-heading score is identical to the digit (46.527) in two separate runs, so the pipeline is deterministic for fixed settings.

On the same photos, 9 headings vs plain: 46 photos better, 7 worse, 220 unchanged. The worse ones: 5 at -4 points (a +10 answer became a +6), 1 at -3, and the one wrong answer at -20 (photo 347).

### 4.3 Scale and heading together [measured]

Controlled run (M 10, confidence 0.21):

| Variant | Queries | Score | Wrong answers | Target in candidates | Wrong candidate reaches bar | s / photo |
|---|---|---|---|---|---|---|
| plain | 1 | 41.3 | 0 | 87.2 % | 0.0 % | 0.69 |
| 3 headings | 3 | 46.5 | 1 | 98.9 % | 0.7 % | 2.09 |
| scale 1.05 only (heading 0) | 2 | 44.0 | 0 | 90.8 % | 0.4 % | 1.39 |
| **3 headings x scales {1.0, 1.05}** | 6 | **49.5** | 1 | 99.6 % | 2.6 % | 4.17 |

Per seed gain of the 6-query set over plain: +7.1, +8.6, +8.9. On the same photos: 44 better, 5 worse (4 at -4, 1 at -20), 224 unchanged.

Scale-only test (M 5, confidence 0.20, scales 0.95 / 1.0 / 1.05): plain 41.7, 3 headings 47.8, 3 scales alone 44.2, **3 headings x 3 scales 50.2** (9 queries, 4.8 s / photo, wrong candidate reaches bar 2.9 %).

What each single query finds (share of photos where that query alone has a candidate within 4 sigma): heading 0, scale 1.0: 87.2 %; heading 0, scale 1.05: 86.1 %; heading 0, scale 0.95: 72.9 %. So the useful direction of scale is **up**; 0.95 alone is clearly worse and adds little. The M 5 / 0.20 run (50.2) scores slightly above the M 10 / 0.21 run (49.5); the settings differ and the gap is inside the noise.

### 4.4 Where the gain is (flight height) [measured, M 10 run]

Height groups of 91 photos each: low 100-148 m, mid 148-280 m, high 280-351 m. Score out of 60:

| Variant | low | mid | high |
|---|---|---|---|
| plain | 29.6 | 47.6 | 46.7 |
| 3 headings | 38.2 | 54.2 | 47.1 |
| 5 headings | 40.9 | 55.5 | 47.1 |
| 9 headings | 44.8 | 55.3 | 49.1 |
| scale 1.05 only | 33.8 | 48.9 | 49.2 |
| 3 headings x 2 scales | 41.5 | 55.5 | 51.4 |

The gain is largest at the low flights (+15.2 for 9 headings), where a few degrees of heading error moves the footprint the most relative to the picture. Real evaluation altitudes are unknown, and nothing above 351 m was tested.

### 4.5 Which heading finds the place [measured, all seeds]

Share of photos where a single query has a candidate within 4 sigma of the truth, 9 headings, scale 1.0:

| Heading | -10 | -7.5 | -5 | -2.5 | 0 | +2.5 | +5 | +7.5 | +10 |
|---|---|---|---|---|---|---|---|---|---|
| target found | 35.5 % | 59.0 % | 78.4 % | **90.5 %** | 87.2 % | 78.0 % | 69.2 % | 50.2 % | 40.3 % |
| right candidate reaches the bar | 26.4 | 45.8 | 59.7 | 70.7 | 70.7 | 64.8 | 61.2 | 41.0 | 25.3 |

Two readings: (a) the search tolerates about 2.5 degrees (a 12-point drop per step), so a step of 2.5 is about right and a step of 5 loses some photos; (b) the best single heading is **-2.5**, not 0, which fits the ~1 degree roll/pointing bias of the patch. A fixed shift of about -1 to -2 degrees might therefore be worth testing (Section 6.1).

### 4.6 Heading penalty [measured]

Each candidate pays `penalty x |heading|` inliers when the winner is picked. At M 10: 49.74 (none), 49.60 (0.25), 49.30 (0.5), 48.99 (1.0). It never helps, because a correct match found at a large heading is exactly what the extra queries are for. A sinusoidal shape would not change that. The code stays in `choose_with_heading_penalty`; the default should be 0.

---

## 5. The parameter test on the Cartosat map (`tests/Parameter_Test`)

Run: 100 photos from `FIRST_PHOTO = 0`, `COARSE = fused`, LoFTR + MINIMA, K 100, M 25, bar 8, **confidence 0.10**, RANSAC 3 px. Baseline **46.56 / 60** (first half 49.68, second half 43.44), 80 % answered, 0 wrong. Data: `tests/Parameter_Test/full_runs/official_fused_loftr_minima_first0_n100_conf0.10_thr0.1_border0_margin0_K100_M25_bar8/` (`panels/` one CSV per sweep, `tables/`, `raw_matches_margin*.npz`, `candidates.csv.gz`, `fine_variants.csv.gz`, `canvases.npz`, `heat_maps/`, and `Full_Variety_Run_Results.zip` which duplicates it). One-factor sweeps from that baseline, all **[measured, in-sample, 100 photos]**:

| Parameter | Values -> score | Verdict |
|---|---|---|
| LoFTR confidence | 0.10: 46.6, 0.15: 45.0, 0.20: 42.2, 0.30: 37.1, 0.40: 29.2, 0.5: 24.4 ... 0.8: 9.0 | The score falls steadily as the cut rises, because inlier counts fall and the bar of 8 is then missed (right-but-skipped goes 4 -> 69). **This is why 0.1 looks best, and why it cannot be compared across settings without moving the bar.** No wrong answers at any value |
| Answer bar | 1-3: 40.9 (**6 wrong**), 4: 43.3 (4 wrong), 7: 46.3 (1 wrong), **8-10: 46.6 (0 wrong)**, 13: 44.8, 16: 42.6 | At confidence 0.1 a low bar buys false flags. 8 is the lowest bar with no wrong answer |
| K (places from coarse) | 5: 44.2, 10: 44.2, 25: 46.0, 50-100: 46.6; target in K 0.81 -> 0.88 | recall still rising slowly with K |
| M (sent to fine) | 1: 45.4, 3: 45.4, 5-50: 46.6 | 5 candidates are as good as 50; the fine stage does not need more |
| DINO weight | 0: 46.0, 0.3-100: 46.6; top-1 0.77 -> 0.83-0.87 | DINO helps the order, the score barely moves |
| DINO layer off | 45.96 vs 46.56 | +0.6 |
| gamma (DINO in the choice) | 0 ... 4: 46.56 everywhere | no effect |
| RANSAC distance | 1.0: 44.4, 1.5: 47.3, 2: 47.0, 3: 46.6, 4: 46.1 (1 wrong), 6: 47.2 (1 wrong), 8: 46.9 (1 wrong) | flat within about +-1; 4 px and above introduce a wrong answer |
| Border (px of the canvas ignored in the fit) | 0-8: 46.6-46.8, 12-16: 46.8, 24: 45.8 (1 wrong) | up to 16 px does nothing useful |
| Margin (grey context around the canvas) | 0: 46.6, 8: 46.1, 16: 46.4, 32: 46.1 (1 wrong each) | no help, slight harm |
| Rescue rule (inliers, z) | none 46.6; (4, 2.0): 46.3 (1 wrong); (6, 2.0): 46.3 (1 wrong); (8, 2.5): 46.6; **(0, 2.5): 47.5, 0 wrong**; (0, 3.0): 46.9; (0, 3.5): 46.6 | only the 0-inlier rescue helps, by about +1 [in-sample] |

Reading: **the old hyperparameters are exhausted.** Everything except the bar and the confidence moves the score by less than the noise (the standard error is about 2.7 / 60 for 100 photos). That is the same message as the patch: the lever is recall and geometry (what the heading queries address), not decision parameters. The earlier analysis `tests/Parameter_Test/Analysis_2026-11-06_Stage_Evaluation_and_Strategies.md` has the funnel (11 photos lost at the coarse stage, the biggest bucket) and the strategies S1-S8 for the same data at confidence 0.3.

`tests/Parameter_Test/runs_summary.csv` holds four rows: n 66 / conf 0.3 (43.5), n 100 / conf 0.3 (39.5), **n 14 / conf 0.1 (49.7: this is one of my 14-photo smoke runs, not a real result; delete that row)**, and n 100 / conf 0.1 (46.6, the real run). The "49" seen early on came from that n 14 row.

---

## 6. What to try next

1. **Fixed heading offset [not tested].** -2.5 is the best single heading and the patch estimates a ~1 degree roll bias. Test headings {-5, -2.5, 0} and {-3.75, -1.25, +1.25} against {-5, 0, +5} with scales {1.0, 1.05}, 3 new seeds. If a shifted 3-heading set with scale 1.05 matches the 6-query score, the Jetson cost drops from 4.2 s to about 2 s. (Edit `HEADINGS` and `VARIANTS` in the settings cell; the asserts need 0.0 in `HEADINGS`.)
2. **Agreement vote across headings [not tested].** The one wrong answer was a single weak match at the bar (8 inliers, 100 m flight). A rule like "a bar-level answer must be supported by a second query within a few sigma" could remove it. The per-query candidates are in the saved run, so it can first be tested offline.
3. **Put it into `Single_Image_Run`** with a `HEADINGS` and `SCALE_FACTORS` setting (default 3 headings x {1.0, 1.05}), using `HeadingQueryPipeline`.
4. **Confirm on different seeds** (for example 44, 55, 66) with the chosen set only; do not tune on 11, 22, 33 any more.
5. **Run `Stage_Evaluation.ipynb` Part 6.7 on the 100 photos** (see 7.2). It gives the per-stage view (points lost at coarse+DINO, fine, decision) for plain against the heading variants.
6. Re-derive the answer bar for the final heading set (the patch warns that every bar is tied to the matcher settings), looking at the wrong-candidate-reaches-bar rate (2.6 % at 6 queries).

---

## 7. State of the notebooks

### 7.1 `tests/Heading_Queries/Heading_Queries.ipynb`
Settings cell: `DATA official`, `MAP new`, `POOL_PHOTOS 500`, `SEEDS [11, 22, 33]`, `N_PER_SEED 150`, `FOUND_SIGMA 4.0`, `HEADINGS`, `SCALE_FACTORS`, `COARSE edge_zncc`, `K_PER_QUERY`, `M_PER_QUERY`, `GAMMA 1.0`, `MATCHER loftr_minima`, confidence 0.21, `MIN_INLIERS 8`, `VARIANTS`, `HEADING_PENALTIES`. One photo at a time goes through all its queries; each variant decides from its own subset. Failures: OOM is retried 3 times and the run stops if more than 5 % of photos fail. A paired bootstrap of every variant against plain (2000 resamples) sits after the cell that defines `plain_name` / `change`. The Colab runs of this notebook produced the valid results; the executed local copy `Heading_Queries_executed_n150.ipynb` is the invalid OOM run (kept only as a record; safe to delete).

### 7.2 `Stage_Evaluation.ipynb` (root)
The recreated original, now with: confidence 0.21; settings `RUN_HEADINGS`, `HEADINGS`, `SCALE_FACTORS`, `HEADING_COARSE` (edge_zncc), `HEADING_K 50`, `HEADING_M 10`, `HEADING_VARIANTS`; **Part 6.7** (runs the queries on the same photos as the stage tables, then a variant table, a funnel with ceilings, the per-heading table, a paired bootstrap, and a points-lost-per-stage table and figure comparing the baseline with every variant); a save cell for the `headings_*` CSVs. Status: **only a 4-photo smoke test ran** (0 failed photos, all tables and figures produced, results written to a scratch folder). No real run has been saved. Notes: the baseline row uses the fused map, K 100, M 25, the heading rows use `edge_zncc`, K 50, M 10, so compare the heading variants with "plain (heading 0)", not with the baseline row. The heading step merges "target in K" and "target in M" into one (only the final candidates of each query are kept). The IDE overwrites open notebooks with stale buffers: close the tab without saving before running.

### 7.3 `Stage_Evaluation_Colab.ipynb`
Not created by me. It is the previous `Stage_Evaluation.ipynb` plus three cells (mount Drive and build the Python 3.10 environment, copy results to Drive). It has **none** of the changes above. It should be regenerated from the main notebook by script, not edited by hand.

### 7.4 `tests/Parameter_Test/Parameter_Test.ipynb`
Executed, 100 photos, confidence 0.1 (Section 5). Its "save everything" cell wrote `full_runs/...`. `tests/Stage_Evaluation/` no longer exists on disk (moved), and the recreated `Stage_Evaluation.ipynb` will create it again on its first run.

---

## 8. Memory, disk and problems met

- **RAM** [measured]: edge_zncc pipeline after setup 2.6 GB (peak 3.5 GB): map pictures 0.5 GB, map features 1.0 GB (8 channels, float32). With `COARSE = fused` setup is **4.7 GB** (MIND +1.0 GB, learned +1.0 GB). Stage_Evaluation was reported as the RAM problem on the offline machine (about 11 of 15 GB were already used by other programs). The simplest fix is `COARSE = "edge_zncc"` there (-2.1 GB); the heading part already uses edge_zncc. Options not applied: float16 map features (-0.5 GB, changes scores slightly, needs a re-run), `MAP = "new_931"` (halves disk and map time, but cuts the search disc for the southern photos).
- **Disk**: `dataset/` 29 GB (Dataset_Official_2 24 GB of it), results small (Parameter_Test 54 MB, Heading_Queries 5 MB).
- **GPU**: 6 GB shared with the user's kernels; another process held 3.8 GB and caused the invalid run. Close other GPU notebooks before running.
- **Timing** on the development GPU: 0.7 s per query. 3 headings 2.1 s, 5 headings 3.5 s, 9 headings 6.2 s, 6 queries 4.2 s. The Jetson time is **not measured**.
- Editor problem: open notebooks get overwritten by a stale buffer (happened to Stage_Evaluation and Heading_Queries several times).
- Pre-existing failing test (unrelated): `test_report_runs_on_synthetic_results`.

---

## 9. Where everything is (relative to the repo folder)

| What | Path |
|---|---|
| This report | `tests/Heading_Queries/Analysis_2026-11-07_Heading_Queries_and_Parameter_Tests.md` |
| Earlier analysis (confidence 0.3, strategies S1-S8) | `tests/Parameter_Test/Analysis_2026-11-06_Stage_Evaluation_and_Strategies.md` |
| Parameter test notebook and data | `tests/Parameter_Test/Parameter_Test.ipynb`, `runs_summary.csv`, `per_photo_*.csv`, `sweep_*.csv`, `maps_*.csv` |
| Parameter test full run (conf 0.1) | `tests/Parameter_Test/full_runs/official_fused_loftr_minima_first0_n100_conf0.10_thr0.1_border0_margin0_K100_M25_bar8/` (+ `Full_Variety_Run_Results.zip`) |
| Heading test notebook | `tests/Heading_Queries/Heading_Queries.ipynb` |
| Heading results, 30 photos (valid) | `tests/Heading_Queries/results/official_edge_zncc_loftr_minima_seeds11-22-33_n30_heads9_scales1_K50_M5_conf0.20_bar8/` |
| Heading results, 150 photos, OOM (invalid) | `tests/Heading_Queries/results/official_edge_zncc_loftr_minima_seeds11-22-33_n150_heads9_scales1_K50_M5_conf0.20_bar8/` and `tests/Heading_Queries/Heading_Queries_executed_n150.ipynb` |
| **Heading results, 150 photos, 9 headings (main)** | `tests/Heading_Queries/results/official_edge_zncc_loftr_minima_seeds11-22-33_n150_heads9_scales1_K50_M10_conf0.21_bar8/` |
| Scale test (0.95 / 1.0 / 1.05) | `tests/Heading_Queries/results/Heading_Queries_results_v2/official_edge_zncc_loftr_minima_seeds11-22-33_n150_heads3_scales3_K50_M5_conf0.20_bar8/` |
| **6-query controlled run (3 headings x {1.0, 1.05})** | `tests/Heading_Queries/results/official_edge_zncc_loftr_minima_seeds11-22-33_n150_heads3_scales2_K50_M10_conf0.21_bar8/` (and the `.zip` next to it) |
| Duplicate uploads | `tests/Heading_Queries/results/Heading_Queries_results.zip`, `Heading_Queries_results/`, `Heading_Queries_results_v2.zip` (same data as the main and invalid runs) |
| Files in each result folder | `summary_by_seed.csv` (score per variant per seed and all), `per_photo_per_variant.csv`, `per_photo_per_query.csv`, `by_heading.csv`, `changed_photos.csv`, `photos_per_seed.json`, `settings.json`, `pipeline.txt`, `failures.csv` |
| Stage evaluation (recreated) | `Stage_Evaluation.ipynb`; will write `tests/Stage_Evaluation/headings_*.csv` and `headings_points_lost_per_stage_*.csv` |
| Code | `single_photo_pipeline.py`, `heading_query_pipeline.py`, `src/coarse_search/heading_queries.py`, `train/checks/heading_results.py`, `src/fine_matching/loftr_matcher.py`, `src/header/settings.py`, `src/header/map_source.py` |
| Tests | `tests/pipeline/test_heading_queries.py`, `tests/pipeline/test_loftr_matcher.py`, `tests/pipeline/run_all.py` |
| Source patch document | `Documents/Our_Architecture/Patch_v0.5.6_Recall_Geometry_and_Bias.tex` (outside the repo, next to it) |
| Data-setup guide | `dataset/README.md` |

---

## 10. Caveats

- All headings, scales and thresholds were looked at on the same 273 development photos (3 seeds, 100-351 m). The set was not used to *fit* a model, but the choice of "3 headings x {1.0, 1.05}" was made after seeing these results; confirm on new seeds.
- The standard error of a score is about 2.7 / 60 for 100 photos and about 1.6 for 273; the paired bootstrap of the gains is far above it, but the differences between the top variants (9 headings, 6 queries, 9 queries with 3 scales) are inside it.
- One map (Cartosat-3, Mumbai), one matcher (LoFTR + MINIMA), one camera/geometry setting. The real evaluation map and photos are different, and the mix of flight heights there is unknown.
- The yaw error that the headings fix is estimated from the development photos. If the evaluation yaw metadata is better, the gain will be smaller; if worse, the headings matter more.
- The ratio of false flags rises with every extra query. 1 wrong answer in 273 is a small count; do not read "1.8 %" or "2.6 %" as precise.
