# Where to look, which peaks to keep, and how sure are we

Code: `src/coarse_search/search.py` (`cut_map_features`, `correlation_maps`, `pick_peaks`, `search_frame`).

---

## 1. The search window

The prior says: "the drone is within `R = 5000` m of the point `p`". Every possible *query centre* lies inside that disc. The correlation heat map is indexed by where the query's **top-left cell** goes, so we must give the FFT a window big enough that every centre in the disc has a full query underneath.

Work in level pixels (`g` metres per pixel, `side` = query size in cells = 56, `CELL = 4`):

* The query is `side·CELL` pixels wide, so its centre is `side·CELL/2` pixels from its edge.
* The leftmost wanted centre is at `p_x/g − R/g`. The window's left edge must therefore be at `p_x/g − R/g − side·CELL/2`.

Hence the code:

```python
reach = DISC_RADIUS / gsd + side * CELL / 2                # pixels from the prior to the window edge
cells_wide = int(math.ceil(2 * reach / CELL))              # window size in cells
top  = int(round((prior_m[1] / gsd - reach) / CELL))       # window corner, in cells
left = int(round((prior_m[0] / gsd - reach) / CELL))
window = cut_map_features(map_features[level], top, left, cells_wide, cells_wide)
```

`cut_map_features` copies the part of the window that lies inside the map and leaves zeros elsewhere (zero cells = "no map", detected by the guards). Window size at level 3: `reach = 5000/0.849 + 112 ≈ 6005` px, `cells_wide ≈ 3003`. At level 7: `reach ≈ 1585`, `cells_wide ≈ 793`: a lower level means a much smaller window, which is why high photos are cheaper.

### From heat-map index to metres

Heat-map entry `(r, c)` means the query's top-left cell sits at map cell `(top + r, left + c)`. The query centre in pixels is that cell index times `CELL` plus half the query width; multiply by `g` for metres:

$$y_m=\big((\text{top}+r)\cdot\text{CELL}+\tfrac{\text{side}\cdot\text{CELL}}{2}\big)\,g,\qquad x_m=\big((\text{left}+c)\cdot\text{CELL}+\tfrac{\text{side}\cdot\text{CELL}}{2}\big)\,g.$$

```python
y_m = ((top + row_ids) * CELL + side * CELL / 2) * gsd
x_m = ((left + col_ids) * CELL + side * CELL / 2) * gsd
```

(`x` east, `y` south, map metres.) The query centre is the canvas middle = the drone's nadir point, so these are directly the candidate drone positions.

### The disc mask

The window is a square but the prior is a disc. Corner positions are farther than `R`:

```python
dx = x_m[None, :] - float(prior_m[0])
dy = y_m[:, None] - float(prior_m[1])
inside = dx ** 2 + dy ** 2 <= DISC_RADIUS ** 2
```

`x_m[None, :]` is a row vector and `y_m[:, None]` a column vector; subtracting them broadcasts to the full 2-D grid of distances, no loops. A self-test checks that `inside.sum()` matches the area `π(R/(CELL·g))²` of the circle in heat-map entries within 1 %.

## 2. Picking peaks (non-maximum suppression)

The best score alone is risky: a look-alike may win. So we keep the 25 best **distinct** places (`NUM_PEAKS = 25`; the notebook's prose says 10, the code value is 25).

Algorithm (`pick_peaks`):

1. Set `ranking = score_map`; unusable entries (score `-1`, or outside the disc) get `-10⁹`.
2. Repeat 25 times: find the maximum of `ranking`; record its dense score, `x`, `y`, sharpness and ranking value; **blank** a square of half-width `gap` around it with `-10⁹`.

```python
gap = max(TILE // CELL, side // 2)        # = 56 cells = 224 pixels = one tile
ranking[max(row-gap, 0):row+gap+1, max(col-gap, 0):col+gap+1] = -1e9
```

Without blanking, the next best entry would be the pixel next to the best one (the same place, one cell over). The blank square makes candidates at least one tile (190 m at level 3, 760 m at level 7) apart.

`ranking.reshape(-1).max(0)` returns the maximum and its flat index; `row = flat // width`, `col = flat % width` converts back.

If fewer than 25 usable places exist (e.g. mostly off-map), the loop `break`s and the list is padded:

```python
peaks.append((0.0, 1e9, 1e9, 0.0, -1.0))
```

A position of `1e9` m is the **sentinel** "empty slot"; all later code tests `peaks[k, 1] > 1e8` to skip it.

### Output table `peaks` (25 × 6, numpy)

| column | content |
|---|---|
| 0 | dense score (ZNCC or cosine) at the peak |
| 1 | x (metres, east) |
| 2 | y (metres, south) |
| 3 | sharpness |
| 4 | ranking value (dense + β·hint) |
| 5 | DINO similarity under the peak (`−1` if not used) |

## 3. The confidence signals

**Ratio** = peak 1 / peak 2 (dense scores, `ratio = peaks[0,0] / max(peaks[1,0], 1e-6)`). If two places fit equally well (a repetitive scene), the ratio is near 1 and we should doubt the winner. A clearly larger value means one place stands out. Weakness: if peak 2 is near zero or negative (ZNCC), the quotient explodes or flips sign; treat the ratio as a ranking signal, never as a probability.

**Sharpness** = `dense − mean(four neighbours)`. A real match is a narrow spike (neighbours are much lower); a vague blob (a smooth region matching everywhere) has sharpness near 0. The four neighbours are the cells above, below, left and right, with `max/min` clamps at the borders.

How good these are as predictors of a correct answer is measured by the AUC in `evaluation/concepts/auc_and_confidence_signals.md`. The matcher's inlier count (next stage) is a stronger signal, and the one the final decision uses.

## 4. The optional hint

`pick_peaks(score_map, geometry, prior_map, beta)` ranks by `score + β·prior_map` when a DINO prior is supplied (`dino_descriptor_hint.md`) but still *reports* the dense score so that ratio and sharpness are untouched by the hint.

## 5. `search_frame`: the whole stage

```python
def search_frame(canvas, valid, level, prior_m, store, variants=(("cosine", 0.0), ("zncc", 0.0)), dino=None)
```

1. `query_edge_features(canvas, valid)` → cells + mask.
2. `correlation_maps(store["features"], level, ...)` → both heat maps and the geometry dict.
3. Optional: `dino.tile_score_raster` and `dino.prior_on_grid`.
4. For each `(score name, beta)` in `variants`: `pick_peaks` and attach the DINO column.

Returns `({(mode, beta): peaks (25×6)}, cutoff)`; `cutoff` is the 30 % quantile of the DINO scores (0 without DINO), currently not used downstream.

The pipeline uses exactly one variant, `("zncc", 0.0)`; the evaluation computes several at once because they share the (expensive) correlation.

## 6. Inputs, constraints, failure modes

* `canvas`: float `(224, 224)`; `valid`: bool `(224, 224)`; `level`: 3 to 7; `prior_m`: `(x, y)` metres; `store`: dict from `build_map_store`.
* The prior must lie within the map or at least have part of its disc over the map. A prior far outside gives `peaks[:,1] > 1e8` for all entries (the self-test "empty map" verifies this).
* The score maps live on `settings.DEVICE`; `pick_peaks` calls `float()` on GPU scalars 25 times × several, each forcing a synchronisation. Cheap compared with the FFT.
