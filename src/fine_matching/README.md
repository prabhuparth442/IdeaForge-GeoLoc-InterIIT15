# fine_matching: checking each candidate with a real image matcher

**Question answered:** the coarse search returned 25 places. For each, would the photo's pixels actually line up with the map there? We find landmark pairs, fit a shift, count how many pairs agree (inliers), and measure the leftover shift. The counts are what the decision stage (`result_thresholding`) uses.

## Concept files, in order of the data flow

1. [`keypoint_matching_superpoint_lightglue_minima.md`](concepts/keypoint_matching_superpoint_lightglue_minima.md): detect/describe/match, SuperPoint, LightGlue attention, why MINIMA weights, the layer-name patch. Explains `matcher.py`.
2. [`similarity_transform_and_ransac.md`](concepts/similarity_transform_and_ransac.md): complex-number form `z' = a·z + b`, least squares, RANSAC iteration count, chance of accidental inliers, the scale/angle gate. Explains `ransac.py`.
3. [`candidate_refinement_and_context.md`](concepts/candidate_refinement_and_context.md): tile cutting, the shift formula, the believable-shift limit, the two optional switches. Explains `second_stage.py`.

## Flow

```
canvas (224²) ──prepare_canvas──► SuperPoint keypoints A           (once per photo)
for each of 25 peaks k:
    map_piece(level k, x_k, y_k) ──► SuperPoint keypoints B_k
    LightGlue(A, B_k)      ──► point pairs (a_i ↔ b_i)
    RANSAC similarity      ──► matrix T, inlier flags
    shift_k = (T·centre − centre) · gsd_L      (metres)
    if |shift_k| > 3 cells·gsd: inliers_k = 0
result: inliers[25], shifts[25,2]
```

## Prerequisites to run anything here

* Python packages: `torch`, `lightglue` (`pip install git+https://github.com/cvg/LightGlue.git`), `opencv`. The weights file `dataset/NUDT/checkpoint/MINIMA/minima_lightglue.pth` (or it is downloaded from `MINIMA_URL` on first use).
* A GPU is strongly recommended: 25 candidates × one SuperPoint+LightGlue pass per photo.
* `fit_transform` alone needs only numpy and OpenCV and runs anywhere (below).

## `matcher.py`

| name | input | output |
|---|---|---|
| `Matcher` | (container) | holds `.keypoint_finder` (SuperPoint) and `.pair_matcher` (LightGlue) |
| `load_matcher()` | none | a `Matcher`, weights loaded; prints how many weight names did not match |
| `find_keypoints(matcher, picture)` | 2-D float array in `[0,1]`, any size | feature dictionary (tensors with batch size 1) |
| `match_keypoints(matcher, fa, fb, picture_size)` | two feature dicts, side length of the pictures in pixels | `(points_a (n,2), points_b (n,2), scores (n,))` float32, in the pictures' pixel coordinates |

Process: resize the picture to `MATCH_SIZE` (448) with cubic interpolation, run SuperPoint (≤ 2048 keypoints, threshold 0.0005, NMS radius 4), LightGlue (9 layers, filter 0.1), take pairs, divide positions by `MATCH_SIZE/picture_size`. Always read the line printed by `load_matcher` about unmatched weights; anything but ~0 missing model weights means the layer renaming failed.

## `ransac.py`

`fit_transform(points_a, points_b)` → `(matrix 2×3 or None, inliers bool (n,))`. Model `z' = a z + b`, `a = s·e^{iθ}`. RANSAC with threshold 3 px, up to 3000 iterations, confidence 0.995 (`cv2.estimateAffinePartial2D`). Returns `None` and all-False if fewer than 4 pairs or `|s − 1| > 0.15` or `|θ| > 12°`.

Runs without any model:
```bash
PYTHONPATH=src python -c "
import numpy as np
from fine_matching.ransac import fit_transform
rng = np.random.default_rng(0)
a = (rng.random((40,2))*200).astype(np.float32)
b = a + np.array([5,-3], dtype=np.float32); b[:10] = (rng.random((10,2))*200)   # 10 outliers
M, inl = fit_transform(a, b); print(M.round(2).tolist(), int(inl.sum()))"
# [[1.0, 0.0, 5.0], [-0.0, 1.0, -3.0]] 30
```
The recovered matrix is the pure shift (5, −3), and the 30 true pairs are the inliers.

## `second_stage.py`

* `prepare_canvas(matcher, canvas, valid)` → features of the canvas placed in a `CTX × CTX` grey picture (fill = mean of real pixels); with `USE_BORDER_MASK` drops keypoints near the rotated photo's border.
* `check_candidate(matcher, canvas_features, level_images, level, x_m, y_m)` → dict with the number of pairs, `n_inliers`, `shift` (metres, east/south), points and tile. Cuts the tile with `map_piece(…, CTX)`, finds keypoints, matches, fits the transform, maps the canvas centre `(CTX−1)/2` through it.
* `second_stage(matcher, canvas, valid, level, peaks, level_images)` → `{"inliers": (25,), "shifts": (25,2)}`. Skips empty slots (`peaks[k,1] > 1e8`). Believable-shift limit: `(MAX_SHIFT_CELLS·CELL + CONTEXT_MARGIN//2)·gsd_L` metres; beyond it the candidate's inliers become 0.

Constraints: `peaks` must be the `(25, ≥3)` array from `search_frame` (columns 1, 2 = x, y in metres); `level_images` is `store["levels"]`; canvas and tile must be the same level or the pixel sizes will not agree.
