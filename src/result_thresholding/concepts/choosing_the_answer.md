# Choosing the answer among the candidates

**Code:** [`../decision.py`](../decision.py) (`final_answer`, `decide`).
**You need first:** `fine_matching/concepts/candidate_refinement_and_context.md` (where `inliers` and `shifts` come from).

---

## 1. The situation

After the two search stages we hold, for each of the (up to) 25 candidates `k`:

* `peaks[k] = (corr, x, y, …)`: the coarse correlation score and the position in map metres. The rows are already ordered by correlation, so row 0 is the coarse winner.
* `inliers[k]`: how many keypoint pairs agreed on one transform at that candidate (0 if the check failed or was rejected).
* `shifts[k] = (dx, dy)`: the sub-cell correction in metres.

We must output **one position** and **one yes/no**: "do we answer this photo at all?" This note covers the first; [`answer_or_skip_expected_value.md`](answer_or_skip_expected_value.md) covers the second.

## 2. Why the inlier count goes first

Correlation measures how well the overall *layout* matches. It is fooled by look-alikes. The inlier count measures how many individual *landmarks* line up, from a different method with different weaknesses. Two methods making independent mistakes are much stronger together than apart: a look-alike that fools the correlation rarely also produces a dozen agreeing keypoint pairs. In the team's tests, re-ranking by inliers was the main gain of the project.

So the rule is: **most inliers wins; if tied, higher correlation wins.**

## 3. The rule as one line: `np.lexsort`

```python
order = np.lexsort((-peaks[:, 0], -stage2["inliers"]))
best = int(order[0])
```

`np.lexsort(keys)` sorts by several keys. Two details people trip over:

* the **last** key in the tuple is the **primary** one; the earlier keys only break ties;
* it sorts **ascending**, so we put a minus sign on both to get "largest first".

So here: primary = `-inliers` (most inliers first), secondary = `-corr` (highest correlation first among equals). `order` is the list of candidate indices from best to worst; `order[0]` is the winner.

### A worked example

Five candidates. Correlations `[0.9, 0.8, 0.7, 0.6, 0.5]` (so index 0 is the coarse winner), inliers `[3, 15, 15, 40, 0]`.

| index | corr | inliers | sort key (−inliers, −corr) |
|---|---|---|---|
| 0 | 0.9 | 3 | (−3, −0.9) |
| 1 | 0.8 | 15 | (−15, −0.8) |
| 2 | 0.7 | 15 | (−15, −0.7) |
| 3 | 0.6 | 40 | (−40, −0.6) |
| 4 | 0.5 | 0 | (0, −0.5) |

Sorted ascending by the first entry, ties by the second: `[3, 1, 2, 0, 4]`. Candidate 3 wins with 40 inliers even though its correlation was only the fourth best. Candidates 1 and 2 tie on 15 inliers and the higher correlation (1) goes first. (This is run with the real code; the printed order is `[3 1 2 0 4]`.)

**Ties at zero are important.** If the matcher found nothing anywhere (all inliers 0, or no matcher at all), everybody ties and the correlation order is kept. So with no fine evidence, the answer degrades gracefully to the coarse winner.

## 4. The fallback: when no candidate is verified

```python
if stage2["inliers"][best] < MIN_INLIERS:
    return peaks[0, 1], peaks[0, 2], stage2["inliers"][best], 0
return peaks[best, 1] + stage2["shifts"][best, 0], peaks[best, 2] + stage2["shifts"][best, 1], stage2["inliers"][best], best
```

* **Enough inliers** (at least `MIN_INLIERS = 12`): the answer is the winner's coarse position **plus its shift**.
* **Not enough:** no candidate is verified. The code falls back to the top correlation peak (index 0) with no shift. The inlier number it returns is still the best one found (below 12), so the next decision will say "skip".

Continuing the example with inliers `[3, 5, 5, 8, 0]`: the best has only 8 inliers, below 12, so the answer is candidate 0's position `(100, 10)` unshifted, the reported inliers are 8 and the chosen index is 0. With the original `[3, 15, 15, 40, 0]` the winner is candidate 3: `x = 400 + (−2) = 398`, `y = 40 + 1.5 = 41.5`, inliers 40, chosen 3 (the code prints `(398.0, 41.5, 40.0, 3)`).

The returned tuple is `(x, y, inliers, chosen)`. `x` and `y` are map metres (x east, y south); converting to latitude and longitude needs the inverse of the map's transform (see `header/concepts/geographic_coordinates_and_map_reading.md`; `SatelliteMap` does not provide that direction yet).

**Edge case to know about.** If the coarse stage found *no* usable position at all (prior far off the map), row 0 is the empty-slot sentinel `x = 1e9` and the fallback would return it. The pipeline's `answer` flag is then False, but the returned `x, y` are meaningless; do not plot them.

## 5. `MIN_INLIERS` versus `ANSWER_MIN_INLIERS`

Both are 12 today, but they answer different questions:

```python
def decide(inliers):
    return bool(inliers >= ANSWER_MIN_INLIERS)
```

* `MIN_INLIERS` decides whether the **position** uses the verified result (winner plus shift) or the unverified fallback.
* `ANSWER_MIN_INLIERS` decides whether we **answer** the photo.

Keep `ANSWER_MIN_INLIERS ≥ MIN_INLIERS`. If it were lower, a photo could be "answered" with the fallback position (the unverified coarse winner), exactly the case the fallback marks as not trustworthy. Raising `ANSWER_MIN_INLIERS` above `MIN_INLIERS` makes the system answer fewer photos but only those with stronger evidence.

## 6. Inputs and outputs

`final_answer(peaks, stage2)`: `peaks` a numpy array `(n, ≥3)` with columns score, x, y; `stage2` the dictionary from `second_stage` (`inliers (n,)`, `shifts (n, 2)`), all asserted to have matching lengths and non-negative counts. Returns `(x, y, inliers, chosen)`. `decide(inliers)`: number to bool. Both are **pure functions** (no randomness, no stored state), so they are trivial to test with no data:

```bash
PYTHONPATH=src python -c "
import numpy as np
from result_thresholding.decision import final_answer, decide
peaks = np.zeros((25,6)); peaks[:,0] = np.linspace(.9,.5,25); peaks[:,1] = np.arange(25)
s = dict(inliers=np.zeros(25), shifts=np.zeros((25,2))); s['inliers'][3] = 20; s['shifts'][3] = (1.5,-2)
print(final_answer(peaks, s), decide(20))"
# (4.5, -2.0, 20.0, 3) True      (x = 3 + 1.5, y = 0 − 2)
```

## 7. Common mistakes

* Reading `np.lexsort` as "first key primary" (it is the last).
* Forgetting the minus signs and getting the *fewest* inliers first.
* Applying the shift of a candidate that did not meet `MIN_INLIERS`.
* Treating the returned `inliers` as proof when the position came from the fallback.

## 8. Check yourself

1. Inliers `[12, 12, 30]`, correlations `[0.9, 0.7, 0.5]`: who wins and why?
2. Why is it a bad idea to let `ANSWER_MIN_INLIERS` be 8 while `MIN_INLIERS` is 12?
3. What does the answer look like if the matcher is switched off?
