# evaluation: how good is it, and how do we know we are not fooling ourselves?

## What this section does, in plain words

Nothing here runs when the system answers a real photo. This folder **simulates the contest** on the 500 development photos:

1. invents a rough starting position (prior) for each photo,
2. runs the pipeline,
3. scores the result with the official point rule,
4. runs the experiments that justified the settings (geometry conventions, error tolerance, one-photo-at-a-time independence),

and does all of it in a way that tries not to flatter us (honest split-half threshold, ceilings, sample-size warnings).

## Learn the ideas first: read the notes in this order

| step | note | one-line takeaway |
|---|---|---|
| 1 | [`simulating_the_prior.md`](concepts/simulating_the_prior.md) | A uniform point in a disc is `R·√u`, not `R·u`; the median distance is 3536 m, not 2500 m |
| 2 | [`measuring_in_sigma_units.md`](concepts/measuring_in_sigma_units.md) | Judge every error in σ, then read the tables: ceiling, tiers, K-table, fixed-threshold table |
| 3 | [`auc_and_confidence_signals.md`](concepts/auc_and_confidence_signals.md) | AUC is the chance that a right photo outranks a wrong one; ties count half |
| 4 | [`honest_threshold_selection.md`](concepts/honest_threshold_selection.md) | A threshold tuned on the same photos looks better than it is; on pure noise 5.9 versus an honest 1.9 |
| 5 | [`geometry_hypothesis_tests.md`](concepts/geometry_hypothesis_tests.md) | Let the data choose yaw, tilt signs, height, scale; the share within 4σ tells right from wrong |
| 6 | [`sensitivity_to_errors.md`](concepts/sensitivity_to_errors.md) | `h·tan δ`, `d·δ`, `(s−1)·d`: what an input error costs on the ground, and the measured tolerance |

`independence_check.py` is explained in `result_thresholding/concepts/one_photo_at_a_time.md`.

## The flow

```
metadata ─► TestSet: truths (m), priors (truth + R√u in a random direction), σ per photo, levels
sample_ids ─ run_correlation ─► disc[i] = 25 peaks per photo ─ run_minima ─► stage2[i], final[i]
                                                                  │
report.collect_answers ─► errors/σ and points for 3 answer types ─► print_results / print_gated / print_final_scores
```

## How to run

Everything goes through the launcher (needs the dataset; `--no-minima` avoids the neural matcher):
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
from evaluation.metrics import auc
print(auc([0,0,0,5],[False,False,True,True]))"     # 0.75 (ties count half)
PYTHONPATH=src python -c "
import numpy as np
from evaluation.testset import random_prior
rng = np.random.default_rng(0)
d = [np.hypot(*random_prior(np.zeros(2), rng)) for _ in range(20000)]
print(np.median(d))"                               # about 3531 (theory 5000·√0.5 = 3536)
```

## File by file

### `testset.py`: the simulated contest

* `truth_in_metres(item, satellite_map)` returns a numpy `(2,)`: `(col·gsd, row·gsd)` of the photo's GPS in map metres.
* `random_prior(truth, rng, radius=5000)` returns `truth + R√u` in a random direction.
* `TestSet(items, camera, satellite_map, flight_of_frame, sample_ids)` holds arrays `truths (N,2)`, `priors (N,2)` (seed 0), `sigmas (N,)`, `query_levels (N,)`, flags `city` (height ≤ 200 m) and `sample_*`; the method `error_m(i, x, y)` returns the straight-line error in metres.

### `run_sample.py`: run the stages on the sample

* `run_correlation(testset, store, variants, dino=None)` returns `(disc, seconds)`; `disc[i] = {"peaks": {variant: (25,6)}, "cutoff": float}`.
* `run_minima(testset, store, matcher, disc)` returns `(stage2, final, seconds_per_photo)`; `final[i] = (x, y, inliers, chosen)`.
They are separate so the slow second stage can be re-run on stored peaks.

### `report.py`: the tables

* `collect_answers(testset, disc, stage2, final, use_minima)` returns `(results, closest, ratios, inlier_counts)`. `results[name] = {"errors": metres, "points": tier points}` for `correlation`, `+MINIMA re-rank` and `+MINIMA re-rank + refine`; `closest` is the distance from the truth to the nearest candidate.
* `print_results`, `print_gated`, `print_final_scores` print the tables described in `measuring_in_sigma_units.md`: shares within 1σ and 4σ, the ceiling at 8σ, the AUCs, the best (optimistic) and honest gated scores, the tier counts, the effect of K, and the fixed-threshold table. `print_table(header, rows)` is plain column printing.

### `metrics.py`: the statistics

* `average_ranks(values)` returns ranks with ties given the average rank.
* `auc(values, is_right)`: rank formula; `nan` if one class is empty.
* `gated_curve(signal, points)` returns `(shares, scores)` at 40 quantile thresholds from 0 to 95 %.
* `best_gate(signal, points)` returns `(best score, threshold, share answered)`.
* `honest_gate(signal, points, repeats=300)` returns `(mean score, std, mean share)` over random half-and-half splits.

### `geometry_sweep.py` and `sensitivity.py`: the experiments

* `share_within_4_sigma(testset, store, geometry, ids)` runs the coarse search with a given geometry and returns the share within 4σ plus the errors in σ.
* `sweep_geometry(testset, store)` tests yaw offset, tilt signs, height offset and scale, prints the chosen geometry and whether it equals `settings.GEOMETRY`.
* `run_sensitivity(testset, store)` prints the share for heading errors of −12 … +12° and for scale factors 0.85 … 1.15 on 18 photos.
Both need the map store and about 100 to 600 coarse searches.

### `independence_check.py`

* `check_independence(testset, store, matcher, ids)` runs `localise_photo` for `ids` forward, then in reverse order, then the last one alone, and **asserts** identical answers, peaks and inlier counts. It proves that no hidden state leaks between photos.
* `check_replay(testset, store, matcher, final)` re-runs 4 random sample photos one by one and asserts they equal the batch results.
It needs `matcher` because it compares inlier counts.

## Input checks

`TestSet` asserts that there are photos, one flight label per photo and valid sample indices; the metrics assert equal lengths and non-empty input; `honest_gate` needs at least 4 photos.

## What goes wrong most often

* Reading an optimistic gated score as the expected score (use the honest row).
* Over-reading tables built from ~20 photos.
* Changing the prior seed between two runs you want to compare.
* Carrying the geometry from one dataset to another without re-running `--sweep`.

## Honest status

The statistics (`auc`, `gated_curve`, `honest_gate`), the prior sampling and the scoring are verified on synthetic data. The full report has not been produced on the real map and photos in this repo, so no real score is quoted here.
