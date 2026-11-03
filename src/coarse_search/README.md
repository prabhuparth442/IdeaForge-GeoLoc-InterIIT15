# coarse_search: where in the 5 km disc does this photo fit?

**Question answered:** given the north-up canvas of one photo and a rough position (prior), compare the photo's edge-direction fingerprint against *every* position of the map inside the 5 km disc, and return the 25 best-matching places. About 7 million positions (level 3) are scored by FFT in one pass (speed on the real map not yet measured here). No learned network is required for the main path (DINO is an optional extra).

## Concept files

| file | teaches | code |
|---|---|---|
| [`template_matching_cosine_and_zncc.md`](concepts/template_matching_cosine_and_zncc.md) | cosine vs zero-mean normalised correlation; why plain cosine saturates at ≈0.75 even for noise | `correlation.py` |
| [`fft_cross_correlation.md`](concepts/fft_cross_correlation.md) | convolution theorem, the 5 FFT products, padding to fast lengths | `correlation.py` |
| [`peaks_ratio_and_the_search_disc.md`](concepts/peaks_ratio_and_the_search_disc.md) | window size, disc mask, non-maximum suppression, ratio and sharpness | `search.py` |
| [`dino_descriptor_hint.md`](concepts/dino_descriptor_hint.md) | ViT features, attention, tile gallery, using a similarity as a soft prior | `dino_hint.py` (optional) |

## Flow (math form)

Let `Q` be the query features `(8, 56, 56)` with valid-cell mask `c` (0/1), and `W` the map window `(8, R, R)` cut around the prior. For every offset `(u,v)`:

```
dot(u,v)    = Σ_b Σ_ij  W_b(i+u, j+v) · Q_b(i,j)                         (FFT 1)
energy(u,v) = Σ_ij  c(i,j) · Σ_b W_b(i+u, j+v)²                          (FFT 2)
cosine      = dot / (‖Q‖ · √energy)
zncc        = Σ (W − W̄)(Q − Q̄) / (‖Q−Q̄‖ · √variance)      with window means from FFT 3 (one per bin)
```
Positions with too little map (`energy < MIN_ENERGY·n`) or too flat (`variance < MIN_VARIANCE·n`) are set to −1. Then `pick_peaks` repeats 25 times: take the maximum inside the disc, record it, blank a square of ± `gap` cells (`gap = max(TILE/CELL, side/2)` = 56 cells at the default) around it.

## `correlation.py`

* `fast_length(n)` → smallest integer ≥ n whose only prime factors are 2, 3, 5 (FFT is fastest there). `fast_length(2999) = 3000`, `1021 → 1024`, `3001 → 3072`.
* `correlate_everywhere(window, kernel, cells, min_energy=MIN_ENERGY, min_variance=MIN_VARIANCE)`

| argument | shape / type | meaning |
|---|---|---|
| `window` | torch float `(8, H, W)` | map features around the prior |
| `kernel` | torch float `(8, h, w)` | query features |
| `cells` | torch float `(h, w)`, 0/1 | which kernel cells are real |

Returns `{"cosine": (H−h+1, W−w+1), "zncc": same}`, values in `[−1, 1]`, −1 where unusable. Constraints: `h ≤ H`, `w ≤ W`; tensors on the same device; the FFT size is padded to `fast_length` of the window so the circular wrap-around never touches the kept `[:H−h+1, :W−w+1]` region (that is why only valid-region output is returned). Cost is O(8·H·W·log(H·W)) per call, independent of `h, w`.

Key lines:
```python
dot = torch.fft.irfft2((window_fft * kernel_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
```
Multiplying by the *conjugate* of the kernel's transform turns convolution into correlation (no kernel flip needed). Summing over bins before the inverse transform costs one inverse FFT instead of eight.
```python
variance = (energy - squared_sums).clamp_min(0)
```
`energy − Σ_b (local sum_b)²/n` is the window's total variance across the cells, computed without ever forming the window means explicitly.

Try it:
```bash
PYTHONPATH=src python -c "
import torch
from coarse_search.correlation import correlate_everywhere
w = torch.rand(8,60,60); k = w[:,20:30,25:35].clone(); c = torch.ones(10,10)
z = correlate_everywhere(w,k,c)['zncc']; print(z.shape, divmod(int(z.argmax()), z.shape[1]), float(z.max()))"
# torch.Size([51, 51]) (20, 25) 1.0
```

## `search.py`

* `cut_map_features(features, top, left, rows, cols)` → zero-padded crop `(8, rows, cols)` (zeros outside the map, which `correlate_everywhere` then treats as "no map").
* `correlation_maps(map_features, level, query_features, query_cells, prior_m)` → `(maps, geometry)`. Window half-size `reach = DISC_RADIUS/gsd + side·CELL/2` pixels (level 3: 6005 px, 3003 cells wide; level 7: 1585 px, 793 cells), top-left in cells `round((prior/gsd − reach)/CELL)`. `geometry` holds `level`, `side`, `x_m`, `y_m` (the **centre** of the query at each output position, metres) and the boolean `inside` mask (`distance to prior ≤ DISC_RADIUS`).
* `pick_peaks(score_map, geometry, prior_map=None, beta=0.0)` → numpy `(25, 5)`: columns `[score, x_m, y_m, sharpness, ranking value]`. `sharpness = score − mean of the 4 neighbours`. Missing peaks are filled with `(0, 1e9, 1e9, 0, −1)`; downstream code tests `x < 1e8`. `prior_map` & `beta` add `beta·prior` to the *ranking* only; the stored score stays the correlation.
* `search_frame(canvas, valid, level, prior_m, store, variants=(("cosine",0.0),("zncc",0.0)), dino=None)` → `(found, cutoff)`: `found[(mode, beta)]` is an array `(25, 6)` (the sixth column is the DINO tile similarity, −1 without DINO). Inputs: `canvas (224,224)` and `valid` from `render_query`, `prior_m = (x, y)` metres, `store` from `build_map_store`.

Minimal use on real data (after building `store`, `items`, `camera`):
```python
canvas, valid = render_query(items[i], camera, level)
found, _ = search_frame(canvas, valid, level, prior_xy, store, variants=[("zncc", 0.0)])
peaks = found[("zncc", 0.0)]
ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)
```
Constraint: the prior must be in the same metre frame as the map (x east, y south from the map's top-left corner).

## `dino_hint.py` (optional, `python run.py --dino`)

`DinoHint(level_images)` loads a DINOv2 ViT-S/14 via `timm`, cuts each level into 224 px tiles every 112 px (≥ 50 % map), and stores one L2-normalised vector per tile (average of the layer-8 patch tokens over real pixels). `tile_score_raster` returns the cosine similarity of the query to each tile near the prior, `prior_on_grid` interpolates it to the correlation grid, and `pick_peaks(..., beta=0.5)` ranks by `ZNCC + 0.5·similarity`. Needs `timm` and the downloaded weights. Costs minutes per level on first use.

## `self_tests.py`

* `slide_slowly(window, kernel, cells)` → numpy cosine and ZNCC by an explicit double loop, the definition of the maths.
* `fast_equals_slow()` → builds a random window, hides a perturbed kernel at (20, 31), checks `|FFT − loop| < 1e-4` and that both find the hidden position. Runs with no data: `PYTHONPATH=src python -c "from coarse_search.self_tests import fast_equals_slow; fast_equals_slow()"` (differences ≈ 1e-6).
* `map_tests(store, centre_m)` → on real features: (1) a piece cut from the map is found again with score > 0.99 and error < 1 m, (2) a prior far outside the map yields no peaks, (3) the number of positions in the disc matches `π(R/(CELL·gsd))²` within 1 %. Run via `python run.py --tests`.
