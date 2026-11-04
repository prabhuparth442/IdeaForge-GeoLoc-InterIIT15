# Checking each candidate: the tile, the shift, and how far we believe it

**Code:** [`../second_stage.py`](../second_stage.py) (`prepare_canvas`, `check_candidate`, `second_stage`).
**You need first:** the two other notes in this folder (the matcher and RANSAC), and `image_processing/concepts/map_zoom_levels.md` for `map_piece`.

---

## 1. The loop in plain words

The coarse search handed us up to 25 candidate positions `(x_k, y_k)` in map metres. For each one we ask:

> "If the drone really were at `(x_k, y_k)`, would the canvas line up with the map there?"

To answer, we cut the **map tile centred on the candidate**, find keypoints in it, match them against the canvas keypoints, fit a transform with RANSAC, and record (a) how many pairs agree (inliers) and (b) how far the fit says the true position is from the candidate (the shift).

```python
def second_stage(matcher, canvas, valid, level, peaks, level_images):
    canvas_features = prepare_canvas(matcher, canvas, valid)       # once per photo
    ...
    for k in range(NUM_PEAKS):
        if peaks[k, 1] > 1e8: continue                             # empty slot, skip
        result = check_candidate(matcher, canvas_features, level_images, level, peaks[k,1], peaks[k,2])
        inliers[k] = result["n_inliers"]
        if math.hypot(*result["shift"]) <= limit: shifts[k] = result["shift"]
        else: inliers[k] = 0
    return dict(inliers=inliers, shifts=shifts)
```

The canvas keypoints are found **once** and reused for all 25 candidates, because only the map tile changes. That saves 24 SuperPoint passes per photo.

## 2. Preparing the canvas (`prepare_canvas`)

```python
fill = float(canvas[valid].mean()) if valid.any() else float(canvas.mean())
padded = np.full((CTX, CTX), fill, dtype=np.float32)
padded[M:M+TILE, M:M+TILE] = np.where(valid, canvas, fill)
features = find_keypoints(matcher, padded)
```

Read it as three actions:

1. **Choose a neutral grey**: the average of the *real* pixels.
2. **Make a blank picture** of size `CTX × CTX` filled with that grey, and put the canvas in the middle (offset `M = CONTEXT_MARGIN`). Canvas pixels that are not valid (the empty corners left by the tilt) are also set to the grey.
3. **Find keypoints** in that picture.

Why the grey? If empty corners were black or white, the sharp border between "photo" and "nothing" would produce strong fake keypoints along the border. With neutral grey the border is far weaker, so keypoints come from the real content.

With the current settings `M = 0`, so `CTX = TILE = 224` and the canvas fills the whole picture.

## 3. Cutting the map tile (`check_candidate`)

```python
tile = map_piece(level_images, level, x_m, y_m, CTX)
tile_features = find_keypoints(matcher, tile)
points_a, points_b, scores = match_keypoints(matcher, canvas_features, tile_features, CTX)
matrix, inliers = fit_transform(points_a, points_b)
```

`map_piece` cuts a `CTX × CTX` square from the same zoom level, **centred on the candidate**. Since canvas and tile are the same level, they have identical pixel size and both are north-up, so a correct candidate gives nearly the identity transform. The function returns a dictionary with the number of pairs, the number of inliers, the shift, the points, the inlier flags and the tile (so you can draw it).

If the candidate lies near the map border, part of the tile is zeros (no map). That is fine; the matcher simply has fewer landmarks.

## 4. The refined shift: where did the nadir point end up?

The canvas centre is the nadir point, the spot directly below the drone. If the tile was centred exactly on the drone and the matching were perfect, the transform would map the canvas centre onto the tile centre. Any difference is how far off the candidate was.

$$\mathbf m=\Big(\tfrac{CTX-1}{2},\tfrac{CTX-1}{2}\Big),\qquad \mathbf q=T\begin{pmatrix}\mathbf m\\1\end{pmatrix},\qquad \Delta\mathbf x_{\text{m}}=(\mathbf q-\mathbf m)\cdot g_L.$$

```python
middle = (CTX - 1) / 2
mapped = matrix @ np.array([middle, middle, 1.0])     # 2x3 matrix times (x, y, 1)
result["shift"] = (mapped - middle) * level_gsd(level)
```

* `(CTX-1)/2 = 111.5` is the pixel-centre coordinate of a 224-pixel picture (pixels are numbered 0 to 223).
* `(x, y, 1)` is the standard trick to apply rotation, scale and translation with a single 2 × 3 matrix: the 1 multiplies the translation column.
* Multiplying by `g_L` (metres per pixel at this level) converts pixels to metres.

### A worked example (level 5, 1.697 m per pixel)

Suppose RANSAC returns the pure shift matrix `[[1, 0, 2.4], [0, 1, −1.3]]`. Then `q = (111.5 + 2.4, 111.5 − 1.3) = (113.9, 110.2)`. The difference `q − m = (2.4, −1.3)` pixels, which times 1.697 gives **(4.07 m east, −2.21 m, i.e. 2.2 m north)**. The shift length is 4.6 m. Interpretation: the landmark that sits at the middle of the canvas is found 2.4 pixels to the right of and 1.3 pixels above the middle of the tile, so the true drone position is about 4 m east and 2 m north of the candidate.

### Sign and axes

Canvas `x` is right (east), canvas `y` is down (south), exactly like map metres (`x` east, `y` south). So the shift is simply **added** to the candidate position in `final_answer`. The final accuracy is a fraction of a canvas pixel (0.85 m at level 3 to 3.4 m at level 7), much finer than the 3.4 to 13.6 m of the correlation cell, because RANSAC's last step averages over all inliers.

## 5. How far do we believe a shift?

```python
limit = (MAX_SHIFT_CELLS * CELL + CONTEXT_MARGIN // 2) * level_gsd(level)      # metres
if math.hypot(*result["shift"]) <= limit: shifts[k] = result["shift"]
else: inliers[k] = 0
```

`math.hypot(dx, dy)` is the length `√(dx² + dy²)`. With `MAX_SHIFT_CELLS = 3`, `CELL = 4` and `CONTEXT_MARGIN = 0`, the limit is `12 × gsd` metres:

| level | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| limit (m) | 10.2 | 14.4 | 20.4 | 28.8 | 40.7 |

Reasoning: the coarse peak is at most a cell or two from the truth. If the matcher wants to move the answer by more than three cells, it has found a *different* alignment from the one the correlation found. We do not trust it: the shift is dropped (stays 0) **and the candidate loses all its inliers**, so it can no longer win the re-ranking. (The example shift of 4.6 m at level 5 passes: the limit there is 20.4 m.)

## 6. The two optional switches (both off)

* **`CONTEXT_MARGIN`** (pixels). Cut the map piece larger by this many pixels on every side, and put the canvas in the middle of a grey picture of the same size. Then a candidate that is a few cells off still has its landmarks inside the piece and can be corrected by a larger shift. The cost is speed (a 448-pixel tile is about 60 % slower). In the team's 379-photo test it did not help (score 25.6 with both switches on versus 26.6 off).
* **`USE_BORDER_MASK`**. Drop keypoints within 4 pixels of the photo's own border, because the corners created by rotation are artefacts. Implementation: erode the validity mask with a 9 × 9 box, look up each keypoint's rounded pixel coordinates in the eroded mask, and keep only the reliable ones:

```python
keep = torch.from_numpy(reliable[rows, columns]).to(DEVICE)
for key in ("keypoints", "keypoint_scores", "descriptors"):
    features[key] = features[key][:, keep]       # axis 1 holds the keypoints
```

(Keypoint coordinates are in the 448-pixel picture, so they are first multiplied by `CTX / MATCH_SIZE` to land in canvas pixels.) The two switches were not tested separately.

## 7. Inputs, outputs, constraints

`second_stage(matcher, canvas, valid, level, peaks, level_images)`

* `canvas (224,224)` float, `valid (224,224)` bool, `level` an int in `level_images`, `peaks (25, ≥3)` from `search_frame` (columns 1, 2 are x, y in metres), `level_images = store["levels"]`. All asserted.
* Returns `{"inliers": (25,), "shifts": (25, 2)}` in metres. Empty slots, rejected fits and out-of-limit shifts all stay 0.
* Cost: 25 SuperPoint passes plus 25 LightGlue passes per photo, the slowest stage of `localise_photo`. Lowering `NUM_PEAKS` trades the ceiling (the chance that the truth is among the candidates) for speed; see the "How many candidates to check" table in the evaluation report.
* It **decides nothing**: choosing the winner and answering or skipping belongs to `result_thresholding/`.

## 8. Common mistakes

* Using the shift when the fit was rejected (it is 0 then, and so are the inliers).
* Mixing levels: a tile from another level has a different pixel size, the shift in metres would be wrong by that factor.
* Believing a large inlier count at a candidate whose shift was clipped: the clip zeroes it on purpose.

## 9. Check yourself

1. The transform maps the canvas centre to `(110.0, 113.0)` at level 3 (0.849 m/pixel). What is the shift in metres? (`(−1.5, +1.5)` pixels, i.e. about 1.8 m in length: 1.27 m west and 1.27 m south.)
2. Why do we fill the empty canvas corners with a grey, not black?
3. Why is the believable-shift limit given in cells and not in metres?
