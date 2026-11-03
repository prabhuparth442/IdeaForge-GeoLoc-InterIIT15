# result_thresholding: one answer, and whether to give it

**Question answered:** from 25 candidates with inlier counts and shifts, what position do we report, and is it safe to report it at all (a wrong answer costs −20, a skip costs 0)? This folder also contains `localise_photo`, the single entry point that runs the whole chain for one photo.

## Concept files

| file | teaches |
|---|---|
| [`choosing_the_answer.md`](concepts/choosing_the_answer.md) | `np.lexsort` re-ranking, the fallback, how `MIN_INLIERS` and `ANSWER_MIN_INLIERS` differ |
| [`answer_or_skip_expected_value.md`](concepts/answer_or_skip_expected_value.md) | expected value, break-even `p > 20/(20+r)`, why a fixed per-photo threshold, honest tuning |
| [`one_photo_at_a_time.md`](concepts/one_photo_at_a_time.md) | statelessness, the per-photo RNG seed, CUDA timing, the result dictionary |

## Decision logic

```
inputs : peaks[k] = (corr_k, x_k, y_k, ...),  inliers[k],  shifts[k]
best   = argmax over k of  (inliers_k,  then corr_k)
if inliers[best] ≥ MIN_INLIERS : position = (x_best + dx_best, y_best + dy_best)
else                           : position = (x_0, y_0)             # top correlation peak, unverified
answer = inliers[best] ≥ ANSWER_MIN_INLIERS
```

## `decision.py`

* `final_answer(peaks, stage2)` → `(x, y, inliers, chosen)`. `peaks` is the numpy `(25, ≥3)` array; `stage2` the dict from `second_stage`. Ties in inliers (e.g. all zero) are resolved by correlation, so with no fine evidence the result is exactly the coarse winner.
* `decide(inliers)` → `bool(inliers >= ANSWER_MIN_INLIERS)`.

Both are pure functions: test them with no data.
```bash
PYTHONPATH=src python -c "
import numpy as np
from result_thresholding.decision import final_answer, decide
peaks = np.zeros((25,6)); peaks[:,0] = np.linspace(.9,.5,25); peaks[:,1] = np.arange(25)
s = dict(inliers=np.zeros(25), shifts=np.zeros((25,2))); s['inliers'][3] = 20; s['shifts'][3] = (1.5,-2)
print(final_answer(peaks, s), decide(20))"
# (4.5, -2.0, 20.0, 3.0) True   (chosen index 3)
```

## `pipeline.py`

`localise_photo(item, prior_m, camera, store, matcher=None)` → dictionary:

| key | meaning |
|---|---|
| `x`, `y` | answer, map metres |
| `answer` | report it or skip |
| `inliers`, `chosen` | evidence and which candidate won |
| `ratio` | `corr₀/corr₁` of the coarse stage |
| `peaks`, `stage2` | full candidate data for diagnostics |
| `seconds` | wall-clock time |

Inputs: `item` (from `make_items`), `prior_m` numpy `(2,)` metres, `camera` (`Camera`), `store` (`build_map_store`), optional `matcher` (`load_matcher`; `None` = correlation only, then `answer=False`).
Order of work: `level_for_height` → `render_query` → `search_frame` (ZNCC, no DINO) → `second_stage` → `final_answer` → `decide`. It reseeds OpenCV's RNG with `SEED` at the start so repeated calls give identical results, and calls `torch.cuda.synchronize()` before stopping the clock.
Convert the answer back to latitude/longitude with the inverse of the map transform (see `header/concepts/geographic_coordinates_and_map_reading.md`); `SatelliteMap` currently provides only the forward direction.

Run it for real: `python run.py --n-photos 5 --independence` calls it per photo, in forward and reverse order, and asserts the same answers.
