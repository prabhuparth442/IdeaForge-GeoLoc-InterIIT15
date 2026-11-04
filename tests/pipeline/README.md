# tests/pipeline: small scripts that test the pipeline in src/, and swap parts of it

One script per pipeline section. None needs the real dataset: they build a small **synthetic world** (a 1.8 km random-texture map and three photos cut from it, with known truth) in a temporary folder. Each prints `PASS`/`FAIL` per test.

| script | what it checks |
|---|---|
| `test_header.py` | scoring rule (statement example 28.8), σ, tiers, sampling, bad input rejected |
| `test_image_processing.py` | levels, normalisation, edge-direction bins, canvas and feature shapes, map store, `map_piece` |
| `test_coarse_search.py` | FFT = slow loop, hidden piece found, ZNCC floor, `fast_length`, peak separation, sentinel rows, true place within the top 25 |
| `test_fine_matching.py` | RANSAC with outliers, the scale/angle gate, `second_stage` with a stand-in matcher |
| `test_result_thresholding.py` | most inliers wins, ties, fallback, `decide`, `localise_photo` (correlation only) and order independence |
| `test_evaluation.py` | AUC (including ties), uniform prior, honest gate, gated curve |
| `test_end_to_end.py` | the whole chain with the stand-in matcher; the report code on synthetic results |
| `run_all.py` | runs all of the above |

```bash
python tests/pipeline/run_all.py                     # everything (a few minutes on a CPU)
python tests/pipeline/test_coarse_search.py          # one section
python tests/pipeline/test_fine_matching.py --only ransac
```

## Swapping a part with your own script

`--swap target=spec` replaces one function of the repo **for that run only**.

* `target`: `package.module.function`, for example `fine_matching.ransac.fit_transform`
* `spec`: `path/to/your_file.py:function` or `some.module:function`

Your function must keep the original signature and return type. Then the same tests judge your version:

```bash
# stricter answer rule: decide() needing 20 inliers (test_decide_threshold should FAIL: proof the swap is active)
python tests/pipeline/test_result_thresholding.py --swap result_thresholding.decision.decide=tests/pipeline/examples/my_rules.py:decide

# a different RANSAC (the gate test should FAIL: your version has no scale/angle gate)
python tests/pipeline/test_fine_matching.py --swap fine_matching.ransac.fit_transform=tests/pipeline/examples/my_fit.py:fit_transform

# a different matcher for the whole chain (same two signatures as fine_matching.matcher)
python tests/pipeline/test_end_to_end.py --swap fine_matching.second_stage.find_keypoints=my_matcher.py:find_keypoints \
                                --swap fine_matching.second_stage.match_keypoints=my_matcher.py:match_keypoints
```

Other good swap targets: `header.scoring.points_for_error`, `coarse_search.correlation.correlate_everywhere`, `coarse_search.search.pick_peaks`, `image_processing.edge_features.edge_features`, `image_processing.thermal.normalise_frame`.

## The stand-in matcher

`stand_in_matcher.py` implements `find_keypoints` and `match_keypoints` with OpenCV SIFT, so `second_stage` and `localise_photo` run **without lightglue or the MINIMA weights**. It works on the synthetic world (same kind of image on both sides). It will not bridge thermal and visible images, so it says nothing about how the real matcher performs.

## What a pass does and does not mean

* Pass: the code does what the notes describe, on clean synthetic data (e.g. photo error 0.1 to 0.2 m with 400 to 560 inliers).
* Not covered: real thermal photos, the real 2.6 GB map, MINIMA, DINO. The synthetic search radius is cut to 600 m (`search.DISC_RADIUS` in the tests) because the map is only 1.8 km wide.
