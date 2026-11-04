# result_thresholding: one answer, and whether to give it

## What this section does, in plain words

By now we have 25 candidate places for the photo, each with an inlier count and a refinement shift. This section does two final things:

1. **Choose one position**: re-rank the candidates by inlier count (the strongest evidence), apply the winner's shift.
2. **Decide whether to answer at all**: a wrong answer costs −20, a skipped photo costs 0, so a photo with weak evidence is better skipped.

The folder also holds `localise_photo`, the **single entry point** that runs the whole chain (levels, canvas, coarse search, fine check, decision) for one photo.

## Learn the ideas first: read the notes in this order

| step | note | one-line takeaway |
|---|---|---|
| 1 | [`choosing_the_answer.md`](concepts/choosing_the_answer.md) | Most inliers wins, ties go to the higher correlation (`np.lexsort`); under 12 inliers the answer is the unverified coarse winner |
| 2 | [`answer_or_skip_expected_value.md`](concepts/answer_or_skip_expected_value.md) | Answer only when `p > 20/(20+r)`, at least about 70 %; the threshold is measured on data, not computed |
| 3 | [`one_photo_at_a_time.md`](concepts/one_photo_at_a_time.md) | One photo in, one dictionary out; reseed per photo; synchronise the GPU before timing |

## The decision logic

```
inputs : peaks[k] = (corr_k, x_k, y_k, ...)   inliers[k]   shifts[k]
best   = candidate with the most inliers (ties: higher corr_k)
if inliers[best] ≥ MIN_INLIERS (12):  position = (x_best + dx_best, y_best + dy_best)
else:                                 position = (x_0, y_0)        # top correlation peak, unverified
answer = inliers[best] ≥ ANSWER_MIN_INLIERS (12)
```

## File by file

### `decision.py`: two small pure functions

* `final_answer(peaks, stage2)` returns `(x, y, inliers, chosen)`. `peaks` is the numpy `(25, ≥3)` array, `stage2` the dictionary from `second_stage`. If nothing is verified the result is exactly the coarse winner.
* `decide(inliers)` returns `bool(inliers >= ANSWER_MIN_INLIERS)`.

They have no randomness and no state, so you can test them with no data:
```bash
PYTHONPATH=src python -c "
import numpy as np
from result_thresholding.decision import final_answer, decide
peaks = np.zeros((25,6)); peaks[:,0] = np.linspace(.9,.5,25); peaks[:,1] = np.arange(25)
s = dict(inliers=np.zeros(25), shifts=np.zeros((25,2))); s['inliers'][3] = 20; s['shifts'][3] = (1.5,-2)
print(final_answer(peaks, s), decide(20))"
# (4.5, -2.0, 20.0, 3) True      (chosen index 3)
```

### `pipeline.py`: the whole chain for one photo

`localise_photo(item, prior_m, camera, store, matcher=None)` returns a dictionary:

| key | meaning |
|---|---|
| `x`, `y` | the answer, map metres |
| `answer` | report it, or skip |
| `inliers`, `chosen` | the evidence and which candidate won |
| `ratio` | `corr₀/corr₁` of the coarse stage |
| `peaks`, `stage2` | full candidate data for diagnostics |
| `seconds` | wall-clock time |

Inputs: `item` (from `make_items`, with `rel_height > 0`), `prior_m` numpy `(2,)` metres (x east, y south), `camera` (`Camera`), `store` (`build_map_store`), and optionally `matcher` (`load_matcher`; `None` means correlation only, and then `answer` is False).

Order of work: `level_for_height` → `render_query` → `search_frame` (ZNCC, no DINO) → `second_stage` → `final_answer` → `decide`. It reseeds OpenCV's random generator with `SEED` at the start so repeated calls give identical results, and calls `torch.cuda.synchronize()` before stopping the clock.

To convert the answer back to latitude and longitude you need the inverse of the map transform (`header/concepts/geographic_coordinates_and_map_reading.md`); `SatelliteMap` currently provides only the forward direction.

Run it for real: `python run.py --n-photos 5 --independence` calls it per photo in forward and reverse order and asserts identical answers.

## Input checks

`final_answer` asserts matching lengths and non-negative counts; `decide` asserts a non-negative count; `localise_photo` asserts a finite prior, a positive height and a proper store.

## What goes wrong most often

* Tuning the answer threshold on the same photos you report (optimistic; use the honest split in `evaluation/`).
* A prior in the wrong frame: the coarse stage finds nothing, row 0 is the `1e9` sentinel, and the position returned is meaningless (the `answer` flag is False).
* Using other photos' information inside `localise_photo`; the contest forbids it.

## Honest status

`final_answer` and `decide` are verified by examples. `localise_photo` is verified end to end only with a synthetic stand-in for the matcher; the real matcher path has not run in this repo.
