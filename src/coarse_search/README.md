# coarse_search: where in the 5.5 km disc does this photo fit?

## What this section does, in plain words

The photo has been turned into an edge-direction fingerprint (`image_processing/`). The map has been turned into the same kind of fingerprint. This section **slides the photo's fingerprint over the map's fingerprint everywhere inside the 5 km circle around the prior**, scores every position, and returns the **25 best distinct places**.

```
query features (8, 56, 56) + mask ─┐
                                    ├─► cut a window of the map features around the prior
map features per level             ─┘          │
                                                ▼
                         correlate_everywhere  (FFT, about 6.8 million positions at level 3)
                                                │
                              heat map of cosine and ZNCC scores
                                                │
                       keep positions inside the disc, convert to metres
                                                │
                              pick_peaks: 25 candidates, each with
                              score, x, y, sharpness, ranking value
```

No neural network is needed for the main path. The result is a short list of candidates, not an answer: the next section (`fine_matching/`) checks each candidate properly.

## Learn the ideas first: read the notes in this order

| step | note | one-line takeaway |
|---|---|---|
| 1 | [`template_matching_cosine_and_zncc.md`](concepts/template_matching_cosine_and_zncc.md) | Cosine scores unrelated places at 0.94; subtracting the average first (ZNCC) makes unrelated places score about 0 |
| 2 | [`fft_cross_correlation.md`](concepts/fft_cross_correlation.md) | The FFT computes the sum-of-products at every shift in seconds; keep only the part that did not wrap around |
| 3 | [`peaks_ratio_and_the_search_disc.md`](concepts/peaks_ratio_and_the_search_disc.md) | Window size, disc mask, "blank the neighbours" peak picking, ratio and sharpness |
| 4 | [`dino_descriptor_hint.md`](concepts/dino_descriptor_hint.md) | Optional: a pretrained network's "looks similar" score as a soft nudge, not a decision |

## Data flow with shapes

```
query_edge_features(canvas, valid)     canvas (224,224), valid (224,224)
   ▼
query_features (8,56,56), query_cells (56,56)
   │             map_features[level]  (8, H_L/4, W_L/4)
   ▼                    ▼
correlation_maps ── cut_map_features ──► window (8, W, W)   W = 3003 cells at level 3, 793 at level 7
   ▼
correlate_everywhere ──► {"cosine": (W-55, W-55), "zncc": same}      −1 = unusable
   ▼
geometry = {level, side, x_m, y_m, inside}
   ▼
pick_peaks ──► peaks (25, 5): [score, x_m, y_m, sharpness, ranking value]
   ▼
search_frame adds the DINO column ──► (25, 6)
```

## Numbers to keep in mind

| level | metres per pixel | window (cells) | positions inside the disc |
|---|---|---|---|
| 3 | 0.849 | 3003 | 6 817 692 |
| 4 | 1.200 | 2140 | 3 408 846 |
| 5 | 1.697 | 1530 | 1 704 423 |
| 6 | 2.400 | 1098 | 852 212 |
| 7 | 3.394 | 793 | 426 106 |

One cell is 4 pixels. The candidate separation (`gap`) is 56 cells = one canvas width: 190 m at level 3, 760 m at level 7.

## File by file

### `correlation.py`: score every position at once

* `fast_length(n)`: smallest integer ≥ n made only of the factors 2, 3, 5 (the sizes FFT likes). `2999 → 3000`, `1021 → 1024`, `3001 → 3072`.
* `correlate_everywhere(window, kernel, cells, min_energy, min_variance)`

| argument | shape / type | meaning |
|---|---|---|
| `window` | torch float `(8, H, W)` | map features around the prior |
| `kernel` | torch float `(8, h, w)` | query features, zero where invalid |
| `cells` | torch float `(h, w)`, 0/1 | which query cells are real |

Returns `{"cosine": (H−h+1, W−w+1), "zncc": same}` in [−1, 1], with **−1 = unusable** (no map under the query, or, for ZNCC, a flat patch).

How it works, in the order of the code:

1. Pad both to `fast_length` sizes and transform with `rfft2`.
2. **Dot product** over the 8 layers: multiply the window transform by the *conjugate* of the kernel transform, add the layers, invert once.
3. **Energy** (sum of squares under the query): correlate the layer-summed squares with the mask.
4. `cosine = dot / (‖Q‖·√energy)`.
5. **ZNCC**: the kernel minus its per-layer average, correlated the same way, divided by the window's variance, obtained as `energy − Σ (local sum)²/n` without ever forming the window means.
6. Crop to the valid positions `[:H−h+1, :W−w+1]` (everything else wrapped around).
7. Write −1 where `energy < 0.05·n` or `variance < 0.005·n`.

Try it (random window, a piece cut from it at row 20, column 25):
```bash
PYTHONPATH=src python -c "
import torch
from coarse_search.correlation import correlate_everywhere
w = torch.rand(8, 60, 60); k = w[:, 20:30, 25:35].clone(); c = torch.ones(10, 10)
z = correlate_everywhere(w, k, c)['zncc']; print(z.shape, divmod(int(z.argmax()), z.shape[1]), float(z.max()))"
# torch.Size([51, 51]) (20, 25) 1.0
```

### `search.py`: window, metres, peaks

* `cut_map_features(features, top, left, rows, cols)`: a zero-padded crop `(8, rows, cols)`; zeros outside the map.
* `correlation_maps(map_features, level, query_features, query_cells, prior_m)` returns `(maps, geometry)`.
  * `reach = DISC_RADIUS/gsd + side·CELL/2` pixels (window extends this far from the prior in every direction);
  * `geometry["x_m"]`, `["y_m"]`: the **centre** of the query at each heat-map entry, in metres (x east, y south);
  * `geometry["inside"]`: `True` where the distance to the prior is at most `DISC_RADIUS` (5500 m; the hint itself is within 5000 m).
* `pick_peaks(score_map, geometry, prior_map=None, beta=0.0)` returns a numpy array `(25, 5)`: score, x, y, sharpness, ranking value. Missing slots: `(0, 1e9, 1e9, 0, −1)`, so check `x < 1e8` before using a row. `sharpness = score − mean of the four neighbours`. `prior_map` and `beta` add `beta·prior` to the *ranking* only; the stored score stays the correlation.
* `search_frame(canvas, valid, level, prior_m, store, variants, dino=None)` runs the whole stage and returns `({(mode, beta): (25, 6) array}, cutoff)`; the sixth column is the DINO tile similarity (−1 without DINO).

Minimal use on real data (after building `store`, `items`, `camera`):
```python
canvas, valid = render_query(items[i], camera, level)
found, _ = search_frame(canvas, valid, level, prior_xy, store, variants=[("zncc", 0.0)])
peaks = found[("zncc", 0.0)]
ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)
```
The prior must be in the same metre frame as the map: x east, y south from the map's top-left corner.

### `dino_hint.py`: optional (`python run.py --dino`)

`DinoHint(level_images)` loads a DINOv2 ViT-S/14 through `timm`, cuts each level into 224-pixel tiles every 112 pixels (at least half real map), and stores one unit-length 384-number vector per tile. At search time `tile_score_raster` scores the photo against the nearby tiles, `prior_on_grid` interpolates those scores to the heat-map grid, and `pick_peaks(..., beta=0.5)` ranks by `ZNCC + 0.5·similarity`. Needs `timm` and weights (ship them, no network at evaluation). **Not yet run in this repo.**

### `self_tests.py`: is the fast code right?

* `slide_slowly(window, kernel, cells)`: the definition of cosine and ZNCC as two explicit loops (slow, obviously correct).
* `fast_equals_slow()`: builds a random window, hides a perturbed piece at (20, 31), and checks that the FFT result and the loop result differ by less than 1e-4 and both find the hidden spot. Needs no data:
  `PYTHONPATH=src python -c "from coarse_search.self_tests import fast_equals_slow; fast_equals_slow()"` (differences about 1e-6).
* `map_tests(store, centre_m)`: on real features, (1) a piece cut from the map is found again with score above 0.99 and error under 1 m, (2) a prior far outside the map yields only sentinel rows, (3) the number of positions in the disc matches `π(R/(CELL·gsd))²` within 1 %. Run through `python run.py --tests`.

## Input checks

Every function starts with `assert` lines stating what it accepts ("kernel is bigger than the window", "window, kernel and cells must be on the same device", "prior_m must be two finite numbers"). They are the quickest reference for the constraints above.

## What goes wrong most often

* Believing a cosine of 0.95 means a match (see note 1).
* Using the full FFT output instead of the cropped valid part (see note 2).
* Treating the `1e9` padding rows as real places (see note 3).
* A prior in the wrong frame (lat/lon, or y pointing north): the window is cut in the wrong place and every score is −1 or poor.
* Memory: level 3 needs a window of 3003 × 3003 × 8 floats (288 MB) plus spectra several times larger; use the CPU or fewer levels on a small GPU.

## Honest status

The FFT code is checked against the slow loops (difference about 1e-6) and by the end-to-end synthetic test. Its speed on the real 2.6 GB map, and the DINO hint, have not been measured or run in this repo.
