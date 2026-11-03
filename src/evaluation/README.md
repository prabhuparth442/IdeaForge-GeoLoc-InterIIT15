# evaluation: how good is it, and how do we know we are not fooling ourselves?

Nothing here is used when the system answers a real photo. This folder simulates the contest on the 500 development photos, scores the result with the official rule, and runs the experiments that justified the settings.

## Concept files

| file | teaches | code |
|---|---|---|
| [`simulating_the_prior.md`](concepts/simulating_the_prior.md) | uniform point in a disc, `r = R√u` | `testset.py` |
| [`measuring_in_sigma_units.md`](concepts/measuring_in_sigma_units.md) | errors in σ, the tier table, ceiling, what each report table means | `run_sample.py`, `report.py` |
| [`auc_and_confidence_signals.md`](concepts/auc_and_confidence_signals.md) | AUC as a probability, the rank formula | `metrics.py` |
| [`honest_threshold_selection.md`](concepts/honest_threshold_selection.md) | tuning bias, split-half selection | `metrics.py` |
| [`geometry_hypothesis_tests.md`](concepts/geometry_hypothesis_tests.md) | letting data choose yaw, tilt signs, scale | `geometry_sweep.py` |
| [`sensitivity_to_errors.md`](concepts/sensitivity_to_errors.md) | `h·tanδ`, `d·δ`, `(s−1)d`; heading sweep | `sensitivity.py` |

`independence_check.py` is explained in `result_thresholding/concepts/one_photo_at_a_time.md`.

## Flow

```
metadata ─► TestSet: truths (m), priors (truth + R√u·dir), σ per photo, levels
sample_ids ─run_correlation─► disc[i] = peaks per photo    ─run_minima─► stage2[i], final[i]
                                                                   │
report.collect_answers ─► errors/σ and points for 3 answer types ─► print_results / print_gated / print_final_scores
```

## How to run

Everything through the launcher (needs the dataset; `--no-minima` avoids the neural matcher):
```bash
python run.py --n-photos 60 --no-minima            # correlation only, quick check
python run.py                                      # full: 500 photos + MINIMA
python run.py --sweep --sensitivity --independence # the experiments
```
Parts that run on synthetic input alone:
```bash
PYTHONPATH=src python -c "
from evaluation.metrics import auc
print(auc([1,2,3,4],[False,True,False,True]))"     # 0.75
PYTHONPATH=src python -c "
import numpy as np
from evaluation.testset import random_prior
rng = np.random.default_rng(0)
d = [np.hypot(*random_prior(np.zeros(2), rng)) for _ in range(20000)]
print(np.median(d))"                               # ≈ 3536 (= 5000·√0.5)
```

## `testset.py`

* `truth_in_metres(item, satellite_map)` → numpy `(2,)`: `(col·gsd, row·gsd)` of the photo's GPS.
* `random_prior(truth, rng, radius=5000)` → truth + `R√u` at a random angle.
* `TestSet(items, camera, satellite_map, flight_of_frame, sample_ids)`: arrays `truths (N,2)`, `priors (N,2)` (seed 0), `sigmas (N,)`, `query_levels (N,)`, flags `city` (height ≤ 200), `sample_*`; method `error_m(i, x, y)`.

## `run_sample.py`

* `run_correlation(testset, store, variants, dino=None)` → `(disc, seconds)`; `disc[i] = {"peaks": {variant: (25,6)}, "cutoff": float}`, `i` the photo index.
* `run_minima(testset, store, matcher, disc)` → `(stage2, final, seconds_per_photo)`; `final[i] = (x, y, inliers, chosen)`.
Separate so that the slow second stage can be re-run on stored peaks.

## `report.py`

* `collect_answers(testset, disc, stage2, final, use_minima)` → `(results, closest, ratios, inlier_counts)`; `results[name] = {"errors": metres, "points": tier points}` for `correlation`, `+MINIMA re-rank`, `+MINIMA re-rank + refine`; `closest` = distance from the truth to the nearest of the 25 candidates.
* `print_results`, `print_gated`, `print_final_scores`: the tables described in `measuring_in_sigma_units.md` (share within 1/4σ, ceiling at 8σ, AUCs, best gated score, honest gated score, tier counts, effect of K, fixed-threshold table). `print_table(header, rows)` is plain column printing.

## `metrics.py`

* `auc(values, is_right)`: rank formula; `nan` if one class empty.
* `gated_curve(signal, points)` → `(shares, scores)` at 40 quantile thresholds from 0 to 95 %.
* `best_gate(signal, points)` → `(best score, threshold, share answered)`.
* `honest_gate(signal, points, repeats=300)` → `(mean score, std, mean share)` over random half/half splits.

## `geometry_sweep.py`, `sensitivity.py`

`share_within_4_sigma(testset, store, geometry, ids)` runs the coarse search with a given geometry and returns the share of photos within 4σ. `sweep_geometry(testset, store)` prints the chosen geometry and whether it equals `settings.GEOMETRY`. `run_sensitivity(testset, store)` prints the share for heading errors −12 … +12° on 18 photos (heading only). Both need the map store; roughly 100 to 300 coarse searches each.

## `independence_check.py`

* `check_independence(testset, store, matcher, ids)`: runs `localise_photo` for `ids` forward, then reversed, then the last one alone, and **asserts** identical answers, peaks and inlier counts. Proves no hidden state leaks between photos.
* `check_replay(testset, store, matcher, final)`: re-runs 4 random sample photos singly and asserts they equal the batch results.
Requires `matcher` (fine stage) because it compares inlier counts.
