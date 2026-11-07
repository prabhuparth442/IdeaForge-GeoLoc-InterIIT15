# Test_Iteration_3_Thresholding

Goal: score more points by answering more of the photos that are answerable, without adding wrong (-20) answers.
Why: in the saved single-image runs (`tests/Single_Image_Run/*.csv`, ~100 photos each) many skipped photos had an error within 1 to 2 sigma but too few inliers.
Lowering the inlier bar alone is not safe (the same low count is mostly right for one matcher and mostly wrong for another), so the work first finds out
WHY the counts are low, then improves the evidence before and after the fine match. Rules: `../../DESIGN.md` (notebook is the face, atomic blocks do the work).

**Nothing here edits `src/`.** Blocks only read from it (DESIGN rule 5), so there is no `changed_src/`.

## Step 1: the oracle-tile test

`notebooks/Step1_Oracle_Tile.ipynb` re-runs the unchanged pipeline on the skipped photos (plus a few answered ones as controls) and runs the same
fine matcher on extra map pieces: the tile at the true position, tiles at several offsets around it, the chosen candidate and coarse peak 0.
For each tile it records the number of pairs, the raw inliers (before the gates), why a fit was rejected (too few pairs, RANSAC failed, scale/angle gate,
shift gate), and the fitted scale, angle and signed shift. Each photo gets a cause label and a picture with the match lines.
The notebook explains every block in plain words and starts with how to choose the Python kernel.

| path | what |
|---|---|
| `notebooks/Step1_Oracle_Tile.ipynb` | the face: settings (one `QUICK_TRY` switch), checks, load the pipeline, choose photos, probe, tables, graph, pictures |
| `thresholding_blocks/probe_blocks.py` | numpy/OpenCV only: `fit_with_reasons`, `classify_cause`, `offset_positions` |
| `thresholding_blocks/probe_tables.py` | pandas only: per-photo causes, offset curve, signed shift at the truth, probe-vs-pipeline check |
| `thresholding_blocks/photo_selection.py` | which photos to probe; check that the loaded photos are the photos of the saved CSV |
| `thresholding_blocks/environment_check.py` | which Python, which packages, which files |
| `thresholding_blocks/probe_matcher.py` | needs torch: runs the real matcher (LoFTR-MINIMA or SuperPoint + LightGlue) on any tile |
| `thresholding_blocks/probe_run.py` | needs the pipeline: one photo through the unchanged pipeline, then the extra tiles |
| `tests/test_blocks.py` | tests on made-up data, no GPU or dataset: `python tests/test_blocks.py` |
| `results/` | written by the notebook: `step1_*.csv`, `step1_offset_curve.png`, `pictures/*.png` (not in git) |

## Status (6 Nov 2026)

* Tested: all blocks that need no GPU (11 tests pass), and the notebook's table, graph, picture, photo-choice and environment cells on made-up data.
* NOT tested: `probe_matcher.py`, `probe_run.py` and the cells that load the pipeline and run the probe loop. They need the datasets, the weights and a GPU.
  On the first real run, read the "safety check" block (probe vs pipeline inliers): if it warns, the later results cannot be trusted.
* The true position is used here only to diagnose. It must never enter the real pipeline.

## Next (after Step 1 is read)

* Step 2: improve the evidence before the fine match (bigger tile, yaw/scale tries, photo-preparation variants, second pass), each tested first on the oracle tile.
* Step 3: judge the result after the fine match (fitted scale/angle against the pose prior, margin over the other candidates, inlier ratio and spread).
* Step 4: one calibrated answer-or-skip rule on expected points, fitted on one data set and tested on the other, with a paired bootstrap.
