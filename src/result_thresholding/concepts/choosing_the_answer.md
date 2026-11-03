# Choosing the answer among the candidates

Code: `src/result_thresholding/decision.py` (`final_answer`, `decide`).

---

## 1. The situation

After `second_stage` we have, for each of the 25 correlation peaks `k`:

* `peaks[k] = (corr, x, y, ...)`, sorted by correlation (peak 0 is the best by correlation),
* `inliers[k]`: how many keypoint pairs agreed on one transform,
* `shifts[k] = (dx, dy)` metres: the sub-cell correction.

We must output one position and one yes/no ("answer this photo?").

## 2. Why re-rank by inliers

Correlation is a *global layout* score and has many look-alikes. The inlier count is *local landmark* evidence from a different method, so the two errors are largely independent. Using the inliers as the primary key gives a stronger system than either alone (the team's tests: this is the main gain of the project).

## 3. Lexicographic sort

```python
order = np.lexsort((-peaks[:, 0], -stage2["inliers"]))
best = int(order[0])
```

`np.lexsort(keys)` sorts by the **last** key first, ties broken by the earlier ones. Here:

1. primary: `-inliers` (ascending of the negative = most inliers first);
2. secondary: `-corr` (among equal inlier counts the higher correlation first).

The ties matter: many candidates have 0 inliers, and then the correlation order is kept, so with no fine evidence the answer degrades gracefully to the coarse winner. `order[0]` is the index of the winner.

## 4. The fallback

```python
if stage2["inliers"][best] < MIN_INLIERS:
    return peaks[0, 1], peaks[0, 2], stage2["inliers"][best], 0
return peaks[best, 1] + stage2["shifts"][best, 0], peaks[best, 2] + stage2["shifts"][best, 1], stage2["inliers"][best], best
```

* Enough inliers (`>= MIN_INLIERS = 12`): answer = the winner's coarse position **plus its shift**.
* Not enough: nobody is verified; use the top correlation peak (index 0) with no shift. The inlier number returned is the best one found (below 12) so `decide` will say "skip", but the position is still a sensible guess. Whether the position is used depends on the second decision.

The returned tuple is `(x, y, inliers, chosen_index)`. `x, y` are map metres (x east, y south); convert back to latitude and longitude with the map's affine transform inverse (see `header/concepts/geographic_coordinates_and_map_reading.md`).

## 5. `MIN_INLIERS` vs `ANSWER_MIN_INLIERS`

Both are 12 at present, and they are coupled:

```python
def decide(inliers):
    return bool(inliers >= ANSWER_MIN_INLIERS)
```

* `MIN_INLIERS` controls whether the **position** uses the verified result (shift applied, re-ranked winner).
* `ANSWER_MIN_INLIERS` controls whether we **answer** at all.

Keep `ANSWER_MIN_INLIERS >= MIN_INLIERS`. If it were lower, a photo could be "answered" using the unverified fallback position (the top correlation peak with no shift), which is exactly the case the fallback marks as not trustworthy. Raising `ANSWER_MIN_INLIERS` above `MIN_INLIERS` answers fewer photos and only those with stronger evidence.

## 6. Inputs, outputs

`final_answer(peaks, stage2)`: `peaks (25,6)`, `stage2` dict from `second_stage`. Returns `(x, y, inliers, chosen)`. `decide(inliers)`: number → bool. Both are pure functions: no randomness, no state, trivial to unit-test:

```python
PYTHONPATH=src python -c "
import numpy as np
from result_thresholding.decision import final_answer, decide
peaks = np.zeros((25,6)); peaks[:,0]=np.linspace(.9,.5,25); peaks[:,1]=np.arange(25); peaks[:,2]=0
s = dict(inliers=np.zeros(25), shifts=np.zeros((25,2))); s['inliers'][3]=20; s['shifts'][3]=(1.5,-2)
print(final_answer(peaks, s), decide(20))"
```
Expected: x = 3 + 1.5 = 4.5, y = −2, 20 inliers, chosen 3, and `True`.
