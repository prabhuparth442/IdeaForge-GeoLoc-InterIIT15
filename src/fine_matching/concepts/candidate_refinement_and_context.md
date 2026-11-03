# Checking a candidate: tile, shift, and how far we believe it

Code: `src/fine_matching/second_stage.py` (`prepare_canvas`, `check_candidate`, `second_stage`).

---

## 1. The loop

The coarse search produced 25 candidate positions `(x_k, y_k)` (map metres). For each candidate we ask: "if the drone were at `(x_k, y_k)`, would the canvas fit the map there?" Answer by matching keypoints between the canvas and the map tile **centred on the candidate**, fitting a transform (`similarity_transform_and_ransac.md`), and counting inliers.

```python
def second_stage(matcher, canvas, valid, level, peaks, level_images):
    canvas_features = prepare_canvas(matcher, canvas, valid)       # once per photo
    ...
    for k in range(NUM_PEAKS):
        if peaks[k, 1] > 1e8: continue                             # empty slot
        result = check_candidate(matcher, canvas_features, level_images, level, peaks[k,1], peaks[k,2])
        inliers[k] = result["n_inliers"]
        if math.hypot(*result["shift"]) <= limit: shifts[k] = result["shift"]
        else: inliers[k] = 0
    return dict(inliers=inliers, shifts=shifts)
```

The canvas keypoints are found **once** and reused for all 25 candidates, because only the map tile changes. This saves 24 SuperPoint passes.

## 2. Preparing the canvas (`prepare_canvas`)

```python
fill = float(canvas[valid].mean()) if valid.any() else float(canvas.mean())
padded = np.full((CTX, CTX), fill, dtype=np.float32)
padded[M:M+TILE, M:M+TILE] = np.where(valid, canvas, fill)
features = find_keypoints(matcher, padded)
```

* Empty canvas pixels get the mean grey of the real pixels, so the border of the real photo is the only strong feature, not a black/white step.
* `M = CONTEXT_MARGIN`. With `M = 0` (the current setting) `CTX = TILE = 224` and the canvas fills the whole picture.

## 3. Cutting the map tile (`check_candidate`)

```python
tile = map_piece(level_images, level, x_m, y_m, CTX)
tile_features = find_keypoints(matcher, tile)
points_a, points_b, scores = match_keypoints(matcher, canvas_features, tile_features, CTX)
matrix, inliers = fit_transform(points_a, points_b)
```

`map_piece` (see `image_processing/concepts/map_zoom_levels.md`) cuts a `CTX × CTX` square of the same level centred on the candidate, so canvas and tile have identical pixel size and orientation. Returned dictionary: number of pairs, inliers, shift, points, tile.

## 4. The sub-cell shift: where the nadir point went

The canvas middle is the nadir point (the drone). If the matching were perfect and the tile exactly centred on the drone, the transform would map the canvas middle to the tile middle. Any difference is how far the candidate was off:

$$\mathbf m=\Big(\tfrac{CTX-1}{2},\tfrac{CTX-1}{2}\Big),\qquad \mathbf q=T\begin{pmatrix}\mathbf m\\1\end{pmatrix},\qquad\Delta\mathbf x_{\text{m}}=(\mathbf q-\mathbf m)\cdot g_L.$$

```python
middle = (CTX - 1) / 2
mapped = matrix @ np.array([middle, middle, 1.0])     # 2x3 matrix times (x, y, 1)
result["shift"] = (mapped - middle) * level_gsd(level)
```

`(x, y, 1)` is the usual homogeneous trick that lets one 2 × 3 matrix apply rotation/scale and translation together. The shift is in metres, `x` east and `y` south, matching the positions in `peaks` (canvas x right = east, canvas y down = south), so it is simply **added** to the candidate position in `final_answer`.

Because RANSAC's final fit uses all inliers (least squares), the shift is accurate to a fraction of a canvas pixel (about 0.85 m at level 3 up to 3.4 m at level 7), far finer than the 3.4 to 13.6 m cell of the correlation.

## 5. How far do we believe a shift?

```python
limit = (MAX_SHIFT_CELLS * CELL + CONTEXT_MARGIN // 2) * level_gsd(level)      # metres
if math.hypot(*result["shift"]) <= limit: shifts[k] = result["shift"]
else: inliers[k] = 0
```

With `MAX_SHIFT_CELLS = 3`, `CELL = 4`, margin 0: `limit = 12 × gsd_L` metres (10 m at level 3, 41 m at level 7). A correct coarse peak is at most about a cell or two from the truth, so a fit that wants to move the answer by more than three cells is a *different* alignment than the one the correlation found. We distrust it: the shift is dropped and the candidate loses all inliers (it can no longer win the re-ranking).

## 6. The two optional switches (both off)

* **`CONTEXT_MARGIN`** (pixels): cut the map piece larger by this many pixels on every side and place the canvas in the middle of a grey picture of the same size. A candidate that is a few cells off then still has its landmarks inside the piece and can be corrected. Costs speed (a 448 px picture is about 60 % slower). In the team's 379-photo test the effect did not help (score 25.6 with both switches on vs 26.6 off).
* **`USE_BORDER_MASK`**: drop keypoints within 4 px of the photo's own border (erode the validity mask by a 9 × 9 box); those corners are artefacts of the rotation. Implementation: keypoint pixel coordinates (scaled by `CTX/MATCH_SIZE` back to canvas pixels) index the eroded mask, and only reliable ones survive:

```python
keep = torch.from_numpy(reliable[rows, columns]).to(DEVICE)
for key in ("keypoints", "keypoint_scores", "descriptors"):
    features[key] = features[key][:, keep]       # axis 1 holds the keypoints
```

The effects of the two were not tested separately.

## 7. Inputs, outputs, constraints

`second_stage(matcher, canvas, valid, level, peaks, level_images)`

* `canvas (224,224) float`, `valid (224,224) bool`, `level` int, `peaks (25,6)` numpy from `coarse_search`, `level_images` = `store["levels"]`.
* Returns `{"inliers": (25,) float, "shifts": (25, 2) metres}`. Empty slots stay 0.
* Cost: 25 × (one SuperPoint on the tile + one LightGlue). This is the slowest stage of `localise_photo`; reducing `NUM_PEAKS` trades the ceiling (probability that the truth is among the candidates) for speed (see the "How many candidates to check" table in the evaluation report).
* It decides **nothing**: choosing the winner and answering/skipping belongs to `result_thresholding/`.
