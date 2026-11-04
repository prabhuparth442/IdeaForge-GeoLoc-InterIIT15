# Where to search, which candidates to keep, and how sure are we

**Code:** [`../search.py`](../search.py) (`cut_map_features`, `correlation_maps`, `pick_peaks`, `search_frame`).
**You need first:** the two notes before this one (what the heat map is, and how it is computed quickly).

---

## 1. What this stage does, in four sentences

The prior tells us the drone is somewhere within 5 km of a point `p`. We cut out the part of the map's edge-fingerprint around `p`, ask [`correlate_everywhere`](fft_cross_correlation.md) for the heat map, convert every heat-map entry into a position in metres, ignore the entries outside the 5 km circle, and finally pick the **25 best distinct places**. Those 25 candidates, with a few confidence numbers each, are what the next stage (keypoint matching) examines.

## 2. Step 1: how big a window do we cut?

Every possible *drone position* lies in the disc of radius `R = 5000 m` around `p`. The heat map is indexed by where the query's **top-left** cell lands, but the drone sits at the query's **centre** (the canvas middle is the nadir point). So for a drone at the far left edge of the disc, the window must start half a query-width further left, otherwise that query would hang off the window.

Work in map pixels of size `g` metres (level-dependent), with query size `side = 56` cells and `CELL = 4` pixels per cell, so the query is 224 pixels wide:

```python
reach = DISC_RADIUS / gsd + side * CELL / 2                # pixels from the prior to the window's edge
cells_wide = int(math.ceil(2 * reach / CELL))              # window size in cells
top  = int(round((prior_m[1] / gsd - reach) / CELL))       # window's corner, in cells
left = int(round((prior_m[0] / gsd - reach) / CELL))
```

Read it as a recipe: "from the prior, walk the disc radius plus half a query (in pixels); that is how far the window must extend in every direction; convert to cells; find the corner."

Verified numbers (from the code):

| level | metres per pixel | `reach` (pixels) | window (cells wide) | positions inside the disc |
|---|---|---|---|---|
| 3 | 0.849 | 6005 | 3003 | 6 817 692 |
| 4 | 1.200 | 4279 | 2140 | 3 408 846 |
| 5 | 1.697 | 3058 | 1530 | 1 704 423 |
| 6 | 2.400 | 2195 | 1098 | 852 212 |
| 7 | 3.394 | 1585 | 793 | 426 106 |

A higher level (a coarser map) means far fewer positions, which is why photos taken from greater height are cheap: each step up in level halves the number of positions (the pixel size grows by √2 in each direction, so the count drops by 2).

### Edge of the map

`cut_map_features` copies the part of the requested window that lies inside the map and **leaves zeros elsewhere**. The zero cells are later recognised by the "no map" guard and scored −1. So a prior near the map border simply produces a window with a blank part; nothing crashes.

## 3. Step 2: from heat-map entry to metres

Heat-map entry `(r, c)` means: the query's top-left cell sits at map cell `(top + r, left + c)`. The query's **centre** in pixels is that cell index times `CELL` plus half the query width. Times `g` for metres:

$$y_m=\big((\text{top}+r)\cdot \text{CELL}+\tfrac{\text{side}\cdot\text{CELL}}2\big)\,g,\qquad x_m=\big((\text{left}+c)\cdot\text{CELL}+\tfrac{\text{side}\cdot\text{CELL}}2\big)\,g.$$

```python
y_m = ((top + row_ids) * CELL + side * CELL / 2) * gsd
x_m = ((left + col_ids) * CELL + side * CELL / 2) * gsd
```

Convention again: `x` is east, `y` is south, both measured from the map's top-left corner. Example at level 7 (`g = 3.394`), window corner at cell (0, 0): entry `(r, c) = (60, 70)` has `x = (70·4 + 112)·3.394 = 1330.4 m` and `y = (60·4 + 112)·3.394 = 1194.7 m` (the toy table in §5 uses g = 3.39 and shows 1328.9 and 1193.3). Since the query centre is the nadir point, these are directly candidate **drone positions**.

## 4. Step 3: cutting the disc out of the square

The window is a square, the prior is a disc, so the corners of the square are further than 5 km from `p` and must be ignored:

```python
dx = x_m[None, :] - float(prior_m[0])       # a row of east-west distances
dy = y_m[:, None] - float(prior_m[1])       # a column of north-south distances
inside = dx ** 2 + dy ** 2 <= DISC_RADIUS ** 2
```

`x_m[None, :]` is a row vector and `y_m[:, None]` is a column vector. Subtracting them makes numpy "broadcast" them into a full 2-D grid: entry `(r, c)` gets `x_m[c] − px` and `y_m[r] − py`. That builds the whole grid of distances without a loop. `inside` is `True` for entries within `R` of the prior. A self-test confirms that `inside.sum()` equals the circle's area `π(R/(CELL·g))²` to within 1 % (e.g. 6 817 692 at level 3).

## 5. Step 4: why not just take the single best entry?

Look at the toy heat map the repo uses to illustrate this: 300 × 300 entries with three smooth bumps, heights 0.95, 0.80 and 0.30 (the first with a smaller satellite bump next to it). The three largest entries are:

```
(row 60, col 70)  0.954
(row 60, col 71)  0.951      ← the very next column
(row 61, col 71)  0.923      ← and the next row
```

They are all the same hill. A correct match has a score that falls off *smoothly* around it, so the next-highest entries are always its neighbours. Without extra care the "top 25" would be 25 pixels of one hill and nothing else. We want 25 **different places**.

### Non-maximum suppression (NMS)

The standard fix: take the best entry; then **blank out a square around it** so its neighbours cannot be chosen; take the best of what is left; blank around it; and so on.

```python
gap = max(TILE // CELL, side // 2)              # = max(56, 28) = 56 cells
...
for _ in range(NUM_PEAKS):                       # NUM_PEAKS = 25
    best_value, best_flat = ranking.reshape(-1).max(0)
    if float(best_value) < -1e8:                 # nothing usable left
        break
    row = int(best_flat) // score_map.shape[1]
    col = int(best_flat) % score_map.shape[1]
    ...
    ranking[max(row - gap, 0):row + gap + 1, max(col - gap, 0):col + gap + 1] = -1e9
```

Line by line:

* `ranking.reshape(-1).max(0)` flattens the 2-D map into one long list and returns the biggest value together with its position in that list (`best_flat`).
* `row = best_flat // width` and `col = best_flat % width` turn the list position back into row and column (integer division and remainder).
* The `-1e8` test says "if what remains is the blanked value (−10⁹), stop".
* The last line writes −10⁹ into a square of half-width `gap` cells around the pick (`max(·, 0)` keeps the slice from going negative).

The gap is one query width (56 cells = 224 pixels): two candidates are at least one canvas apart, i.e. 190 m at level 3 and 760 m at level 7. Candidates closer than that would show overlapping ground and are really the same hypothesis.

### The toy example, through `pick_peaks`

```
rank  score  x (m)    y (m)    sharpness
1     0.954  1328.9   1193.3   0.045
2     0.800  3362.9   3091.7   0.043
3     0.300  2413.7   2006.9   0.016
4     0.000  ... (flat, no more hills)
```

The two smaller bumps near the biggest one have been swallowed by the blank square; the next hill (0.8) and the weakest one (0.3) are found. After that only flat regions remain, so the rest of the list is filled with zero-score entries.

### Empty slots

If fewer than 25 usable entries exist (for instance a prior mostly off the map), the loop stops early and the list is **padded**:

```python
peaks.append((0.0, 1e9, 1e9, 0.0, -1.0))
```

A position of `1e9` metres is the sentinel "this slot is empty". All later code checks `x > 1e8` before using a candidate. Never treat the padding as a real place.

## 6. The confidence signals that come with each candidate

For every kept peak the code records a small row of numbers:

| column | content | why it matters |
|---|---|---|
| 0 | dense score (ZNCC or cosine) at the peak | how well the layout fits |
| 1 | x, metres east | where |
| 2 | y, metres south | where |
| 3 | **sharpness** | is the peak a narrow spike or a broad blob? |
| 4 | ranking value | what the peak was chosen by (dense + β·hint) |
| 5 | DINO similarity (added by `search_frame`; −1 if DINO is off) | optional extra |

### Ratio

`ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)`: how much better is the winner than the runner-up. In the toy: 0.954 / 0.800 = **1.19**. A ratio near 1 means two places fit about equally well, as happens in repetitive scenes (grids of fields, rows of houses). A big ratio means one place stands out. Weakness: if the runner-up is near zero or negative the quotient explodes or flips sign. Treat it as a rough ranking signal, never as a probability.

### Sharpness

`sharpness = dense − mean(the four neighbours: above, below, left, right)`. A true match is a *spike*: its neighbours (shifted by one cell = 3 to 14 m) already fit noticeably worse. A smooth region that fits "kind of OK everywhere" gives a broad plateau with sharpness near 0. In the toy the spike of width 3 gives 0.045; a perfectly flat bump would give 0.

How well these two signals predict a correct answer is measured, not assumed: see `evaluation/concepts/auc_and_confidence_signals.md`. The decision in the end rests on the matcher's inlier count, which is a stronger signal.

## 7. The optional hint

`pick_peaks(score_map, geometry, prior_map, beta)` accepts a second map (from [`dino_descriptor_hint.md`](dino_descriptor_hint.md)) and ranks by `score + β·prior_map`. Crucially the **reported** score stays the plain dense one, so ratio and sharpness are not distorted by the hint. The hint only changes which places make it into the list and in which order.

## 8. `search_frame`: the whole stage in one function

```python
search_frame(canvas, valid, level, prior_m, store, variants=(("cosine", 0.0), ("zncc", 0.0)), dino=None)
```

Steps in the code:

1. `query_edge_features(canvas, valid)` gives the 8 × 56 × 56 fingerprint and the cell mask.
2. `correlation_maps(...)` cuts the window, runs the FFT code, and builds `geometry` (`x_m`, `y_m`, `inside`).
3. If a `DinoHint` is given: compute tile similarities and interpolate them to the heat-map grid.
4. For each `(score name, beta)` variant, call `pick_peaks` and append the DINO column.

It returns `({(mode, beta): array (25, 6)}, cutoff)`. `cutoff` (the 30 % quantile of DINO scores, 0 without DINO) is currently unused downstream. The normal pipeline uses a single variant `("zncc", 0.0)`; evaluation asks for several variants at once because the expensive FFT step is shared.

A minimal use on real data:

```python
canvas, valid = render_query(items[i], camera, level)
found, _ = search_frame(canvas, valid, level, prior_xy, store, variants=[("zncc", 0.0)])
peaks = found[("zncc", 0.0)]
ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)
```

## 9. Inputs and what can go wrong

* `canvas (224, 224)` float, `valid (224, 224)` bool with at least one `True`, `level` in 3 to 7, `prior_m = (x, y)` finite metres in the **map's own frame**, `store` from `build_map_store`. All asserted.
* A prior far outside the map: every window cell is empty, every score −1, every slot is the `1e9` sentinel. (`map_tests` checks exactly this.)
* Note `NUM_PEAKS = 25` in the code (some older prose says 10).
* A wrong prior frame (using lat/lon, or `y` pointing north) shifts the window; the search then looks in the wrong place and finds nothing, silently.

## 10. Check yourself

1. A window 1000 cells wide and a 56-cell kernel: how many heat-map entries per row? (945.)
2. Why is the window larger than the disc by half a query on every side?
3. Without NMS, why would the 25 best entries be nearly useless?
4. What does `x = 1e9` in a peak row mean, and what must code do with it?
