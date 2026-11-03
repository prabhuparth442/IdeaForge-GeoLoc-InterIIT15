# Edge-direction cells: what we really compare

Code: `src/image_processing/edge_features.py`.

---

## 1. The modality gap

A thermal photo records heat; a satellite photo records reflected sunlight. A road can be warm and bright on the thermal image and dark on the map, or the reverse after sunset. Comparing brightness fails. What the two pictures share is the **geometry**: where roads, roofs and field boundaries end. More precisely, the *direction* of those boundaries. A north–south road is a vertical edge in both pictures whether it is bright or dark.

## 2. Gradient: how fast brightness changes

For an image `I(x,y)` the gradient `∇I = (∂I/∂x, ∂I/∂y)` points in the direction of steepest increase and its length is the steepness. At an edge, the gradient is **perpendicular to the edge line**.

On a pixel grid we approximate the derivatives by **central differences** (right neighbour minus left neighbour):

$$g_x[i,j]=I[i,j+1]-I[i,j-1],\qquad g_y[i,j]=I[i+1,j]-I[i-1,j].$$

(Dividing by 2 would give the exact derivative; we skip it because later steps normalise away any constant factor.) The code leaves the 1-pixel border at 0:

```python
gx[:, 1:-1] = blurred[:, 2:] - blurred[:, :-2]
gy[1:-1, :] = blurred[2:, :] - blurred[:-2, :]
```

Before differentiating the picture is blurred with a Gaussian of `σ = 1` pixel. Differentiation amplifies noise (it measures *changes*), so smoothing first suppresses single-pixel noise while keeping real edges, which are several pixels wide.

## 3. Strength and direction

$$s=\sqrt{g_x^2+g_y^2},\qquad\alpha=\operatorname{atan2}(g_y,g_x)\bmod\pi.$$

* `s` is the **edge strength**; flat regions have `s ≈ 0`.
* `α` is the gradient angle. The `mod π` identifies `α` with `α + π`, which means "a light-to-dark edge and a dark-to-light edge are the same". This is the whole polarity trick, proved next.

### Proof that brightness changes do not matter

Suppose the other sensor sees `I' = aI + b` (a gain `a ≠ 0` and an offset `b`). Then `g_x' = a g_x`, `g_y' = a g_y` (the offset disappears in the difference). Hence

$$s'=|a|\,s,\qquad\alpha'=\operatorname{atan2}(a g_y,a g_x)=\begin{cases}\alpha&a>0\\\alpha+\pi&a<0\end{cases}\equiv\alpha\pmod\pi.$$

The direction is unchanged even for `a < 0` (contrast reversal, e.g. a thermal inversion). The strength changes by `|a|`, and the later per-cell normalisation removes `|a|`. Only *nonlinear* brightness changes (not captured by `aI + b`) are not covered exactly, but monotonic ones come close.

## 4. Eight direction bins

The angle range `[0, π)` is split into 8 equal bins of 22.5°:

```python
bin_id = np.minimum((angle / np.pi * N_BINS).astype(int), N_BINS - 1)
```

`angle/π · 8` maps `[0,π)` to `[0,8)`; `astype(int)` floors. `np.minimum(…, 7)` guards the single value `α = π` that floating point rounding can produce. Bin 0 contains gradient angles 0–22.5°, i.e. a horizontal gradient: a vertical edge. Bin 4 (90°–112.5°) is a vertical gradient: a horizontal edge.

Limitation: hard bins have no overlap. An edge at 22.4° and one at 22.6° fall in different bins although nearly parallel. The 4 × 4 cell sums in the next step soften this, but it is a known weakness (soft binning or bin smoothing is a possible improvement).

## 5. Cells: 4 × 4 pixel blocks

For each bin `b` we sum the edge strength of the pixels in that bin over every 4 × 4 block:

```python
weight = (strength * (bin_id == b))[:rows*CELL, :cols*CELL]
features[b] = weight.reshape(rows, CELL, cols, CELL).sum(axis=(1, 3))
```

* `strength * (bin_id == b)` keeps strength only where the pixel's direction is `b`.
* `reshape(rows, 4, cols, 4)` cuts the image into blocks without copying; `sum(axis=(1,3))` adds the two within-block axes.

A 224 × 224 canvas becomes `8 × 56 × 56` numbers. Why cells at all?

1. **Tolerance:** 4 pixels of misregistration (geometry never being perfect) move an edge inside a cell but do not change the cell's content.
2. **Cost:** 16× fewer positions to test in the correlation (the cost scales with the number of cells).
3. **Robustness:** summing many pixels averages out noise.

The price: positions are only resolved to one cell (3.4 to 13.6 m); the fine stage (`fine_matching/`) recovers the sub-cell part.

## 6. Normalising each cell

Let `v` be a cell's 8 numbers, `‖v‖ = sqrt(Σ v_b²)`, and `t` a typical strength (the 30th percentile of `‖v‖` over the valid cells). Output

$$\tilde v=\frac{v}{\|v\|+t+10^{-3}}.$$

Behaviour:

* strong cell, `‖v‖ ≫ t`: `‖ṽ‖ = ‖v‖/(‖v‖+t) → 1`: only the *direction mix* matters; saturated;
* weak cell, `‖v‖ ≪ t`: `ṽ ≈ v/t`: small, so weak noise-like cells contribute little;
* empty cell: `ṽ ≈ 0`.

So `‖ṽ‖ = ‖v‖/(‖v‖+t) ∈ [0, 1)`: a smooth saturating "soft threshold". Both the strong gain differences between sensors and the huge dynamic range of edge strength are tamed. `+1e-3` avoids division by zero for completely flat blocks.

```python
def normalise_cells(features, cells_to_use=None):
    length = np.sqrt((features ** 2).sum(axis=0))
    ...
    typical = np.percentile(length[cells_to_use], 30)
    return features / (length + typical + 1e-3)
```

## 7. The query side

```python
def query_edge_features(canvas, valid):
    features = edge_features(canvas)
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8))
    cell_valid = safe.reshape(side, 4, side, 4).mean(axis=(1, 3)) >= 0.8
    features = normalise_cells(features, cell_valid) * cell_valid
    return features, cell_valid
```

* The canvas has empty corners (filled with the mean grey). Their borders create fake edges. `erode` with a 7 × 7 box shrinks the valid area by 3 pixels on every side to discard the contaminated pixels.
* A cell counts as valid only if ≥ 80 % of its pixels are valid.
* Invalid cells are zeroed after normalisation, and `typical` is computed from valid cells only.

Output: `features (8, 56, 56)` float32 and `cell_valid (56, 56)` bool.

## 8. The map side

The map level pictures use the same recipe (blur, gradient, 8 bins, 4 × 4 cells, normalise) so query and map are made of the same stuff. Differences caused by size:

* **Strips:** a level picture is processed 1024 rows at a time with 8 extra rows above and below (the blur and gradient reach 3σ + 1 ≈ 4 pixels, so 8 are enough) and the padding rows are discarded. No seams appear at strip joins.
* **No-data borders:** where the map ends (value 0) next to real pixels there is a huge fake step edge. `no_data_cells` marks every 4 × 4 cell touching a 0 pixel and `cv2.dilate` with 3 × 3 also marks its neighbours; their features are set to 0.
* **One `typical` for the whole level:** the 30th percentile of the nonzero cell lengths, subsampled every 50th cell (`[::50]`) to save time. Windows cut anywhere from the level are then on one scale.

```python
border = no_data_cells(image)[:features.shape[1], :features.shape[2]]
features[:, border] = 0
length = np.sqrt((features ** 2).sum(axis=0))
typical = np.percentile(length[length > 0][::50], 30)
return features / (length + typical + 1e-3)
```

## 9. Where this leads

Both sides now are non-negative `8 × h × w` arrays. Finding where the query fits in the map is a **template matching** problem. Because all entries are non-negative, plain cosine similarity has a high floor and ZNCC is needed; this is the next folder (`coarse_search/concepts/template_matching_cosine_and_zncc.md`).
