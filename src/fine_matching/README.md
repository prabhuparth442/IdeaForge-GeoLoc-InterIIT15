# fine_matching: checking each candidate with a real image matcher

## What this section does, in plain words

The coarse search returned up to 25 places. Its answer is only as exact as one 4-pixel cell, and a look-alike may sit among the candidates. For **each** candidate we now look at actual landmarks:

1. cut the map tile centred on the candidate;
2. find keypoints in the canvas and in the tile (SuperPoint);
3. pair them up (LightGlue with MINIMA weights, which understand thermal-versus-visible);
4. fit one consistent transform to the pairs, ignoring wrong pairs (RANSAC);
5. record **how many pairs agree** (inliers) and **how far the canvas centre moved** (shift).

The inlier count is the main evidence the decision stage uses. The shift refines the final position to a fraction of a pixel.

## Learn the ideas first: read the notes in this order

| step | note | one-line takeaway |
|---|---|---|
| 1 | [`keypoint_matching_superpoint_lightglue_minima.md`](concepts/keypoint_matching_superpoint_lightglue_minima.md) | Detect, describe, match; SuperPoint finds points, LightGlue pairs them, MINIMA weights bridge thermal and visible |
| 2 | [`similarity_transform_and_ransac.md`](concepts/similarity_transform_and_ransac.md) | `z' = a·z + b` in complex numbers; RANSAC guesses from 2 pairs and counts votes; 12 accidental inliers are essentially impossible, except in repetitive scenes |
| 3 | [`candidate_refinement_and_context.md`](concepts/candidate_refinement_and_context.md) | The nadir point's displacement is the refinement; shifts larger than 3 cells are not believed |

## Data flow

```
canvas (224,224) ── prepare_canvas ──► SuperPoint keypoints A              (once per photo)
for each of the 25 peaks k  (skip rows with x > 1e8):
    map_piece(level, x_k, y_k, 224) ──► SuperPoint keypoints B_k
    LightGlue(A, B_k)       ──► pairs (a_i ↔ b_i), in canvas pixels
    fit_transform (RANSAC)  ──► matrix T (2×3) or None, inlier flags
    shift_k = (T·centre − centre) × metres_per_pixel
    if |shift_k| > 3 cells × gsd: inliers_k = 0, shift_k = 0
result: inliers (25,), shifts (25, 2)
```

## What you need installed

* `torch`, `opencv`, and `lightglue` (`pip install git+https://github.com/cvg/LightGlue.git`) for the matcher.
* The weights file `dataset/NUDT/checkpoint/MINIMA/minima_lightglue.pth`, or network access once so that it can be downloaded from `settings.MINIMA_URL`. **At the final run there is no network, so ship the file.**
* A GPU is strongly recommended (25 SuperPoint and 25 LightGlue passes per photo).
* `fit_transform` alone needs only numpy and OpenCV and runs anywhere.

## File by file

### `matcher.py`

| name | input | output |
|---|---|---|
| `Matcher` | (container) | holds `.keypoint_finder` (SuperPoint) and `.glue` (LightGlue) |
| `load_matcher()` | none | a `Matcher` with MINIMA weights loaded; prints how many weight names did not match |
| `find_keypoints(matcher, picture)` | square 2-D float array in [0, 1] | feature dictionary (tensors, batch size 1) |
| `match_keypoints(matcher, fa, fb, picture_size)` | two feature dicts, side of the original pictures in pixels | `(points_a (n,2), points_b (n,2), scores (n,))` float32 in the pictures' pixel coordinates |

Process: enlarge the picture to `MATCH_SIZE = 448` with cubic interpolation, run SuperPoint (at most 2048 keypoints, threshold 0.0005, NMS radius 4), run LightGlue (9 layers, filter threshold 0.1), take the pairs, divide positions by `448 / picture_size`.

**Always read the line `load_matcher` prints**: "model layers not in file" should be 0. Anything else means the layer renaming failed and part of the network is random.

### `ransac.py`

`fit_transform(points_a, points_b)` returns `(matrix 2×3 or None, inliers bool (n,))`. The model is `z' = a·z + b` with `a = s·e^{iθ}`. RANSAC with threshold 3 pixels, up to 3000 iterations, confidence 0.995 (`cv2.estimateAffinePartial2D`). It returns `None` and all-False if there are fewer than 4 pairs, or if `|s − 1| > 0.15`, or if `|θ| > 12°`.

Runs without any model:
```bash
PYTHONPATH=src python -c "
import numpy as np
from fine_matching.ransac import fit_transform
rng = np.random.default_rng(0)
a = (rng.random((40,2))*200).astype(np.float32)
b = a + np.array([5,-3], dtype=np.float32); b[:10] = (rng.random((10,2))*200)   # 10 outliers
M, inl = fit_transform(a, b); print(M.round(2).tolist(), int(inl.sum()))"
# [[1.0, -0.0, 5.0], [0.0, 1.0, -3.0]] 30
```
The recovered matrix is the pure shift (5, −3); the 30 true pairs are the inliers.

### `second_stage.py`

* `prepare_canvas(matcher, canvas, valid)`: puts the canvas in a `CTX × CTX` grey picture (grey = mean of real pixels) and finds its keypoints; with `USE_BORDER_MASK` it drops keypoints near the rotated photo's border.
* `check_candidate(matcher, canvas_features, level_images, level, x_m, y_m)`: cuts the tile with `map_piece`, finds keypoints, matches, fits the transform, maps the canvas centre `(CTX−1)/2` through it. Returns a dictionary: `n_pairs`, `n_inliers`, `shift` (metres, east/south), `points_a`, `points_b`, `inliers`, `tile`.
* `second_stage(matcher, canvas, valid, level, peaks, level_images)`: loops over the 25 peaks (skipping empty slots, `x > 1e8`) and returns `{"inliers": (25,), "shifts": (25, 2)}`. The believable-shift limit is `(MAX_SHIFT_CELLS·CELL + CONTEXT_MARGIN//2) × gsd` metres: 10.2 m at level 3, 40.7 m at level 7. Beyond it the candidate's inliers become 0.

Constraints: `peaks` must be the `(25, ≥3)` array from `search_frame` (columns 1 and 2 are x and y in metres); `level_images` is `store["levels"]`; canvas and tile come from the same level so that their pixel sizes agree.

## Input checks

Every function starts with `assert` lines: picture must be square with values in [0, 1], points must be finite and the same shape, `peaks` must have 25 rows, candidate positions must be finite metres (not the `1e9` sentinel).

## What goes wrong most often

* The MINIMA weights not fully loaded (read the printed line).
* The `lightglue` package missing, or the weights file missing at a run without network.
* Passing a candidate position from another map frame (the tile is then mostly empty).
* Expecting this stage to decide: it only measures. `result_thresholding/` decides.

## Honest status

`fit_transform` is verified (the example above). `matcher.py` and `second_stage.py` are written, asserted and syntax-checked, but have **not been run on real data** in this repo because `lightglue` and the weights were not available where it was tested. The end-to-end synthetic test replaces the matcher with a stand-in to check the plumbing around it.
