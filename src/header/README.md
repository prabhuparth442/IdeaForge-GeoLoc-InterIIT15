# header: the rulebook, the data readers and the scorer

Everything the other sections share lives here: constants, the scoring rule, how to read photos and the map. Nothing in this folder does image processing or searching. If you want to change a number that affects the whole run, it is almost certainly in `settings.py`.

## Concept files (read in this order)

1. [`concepts/scoring_rule.md`](concepts/scoring_rule.md): GSD, σ, the five point tiers, break-even confidence. Explains `scoring.py`.
2. [`concepts/geographic_coordinates_and_map_reading.md`](concepts/geographic_coordinates_and_map_reading.md): latitude/longitude to map pixel to metres; `x` east, `y` south. Explains `satellite_map.py`.
3. [`concepts/stratified_sampling_and_seeds.md`](concepts/stratified_sampling_and_seeds.md): why the test photos are drawn per height group and why every random draw has a seed. Explains `sampling.py`, `SEED`.

## How to run any file here

None of the files has a `__main__`. Import them with the `src` folder on the path:

```bash
cd NadirForge
PYTHONPATH=src python -c "from header.scoring import check_scoring; check_scoring()"
```
Expected: `sigma at 1000 m ... 17.0 m` and `statement's example score: 28.8`. `python run.py` calls `check_scoring()` first for the same reason: a broken scoring file would silently corrupt every number after it.

## Data flow in this section

```
metadata.csv ──read_metadata──► table ──make_items──► items (list of dicts: name, lat, lon, rel_height, yaw, pitch, roll)
        │                          └──Camera(table)──► camera (fx, fy, cx, cy, k1 k2 k3 p1 p2, 320 x 256)
        └ items ──assign_flights──► flight_of_frame ──pick_sample_ids──► sample_ids
*.tif ──find_map_file──► path ──SatelliteMap(path)──► map object (width, height, gsd, transform, gray_max)
error_m, height ──points_for_error──► points ──final_score──► score out of 60
```

## `settings.py`

Plain constants, no functions. Groups: paths (`IMAGES_DIR`, `MAP_DIR`, `MINIMA_FILE`), run switches (`N_PHOTOS`, `USE_DINO`, `USE_MINIMA`, `SEED`), flight groups (`BAND_EDGES`, `FLIGHT_NAMES`), map reading (`MAP_BANDS`, `MAP_GSD`, `STRIP_ROWS`), pyramid (`LEVELS = [3..7]`, `TILE = 224`), geometry guess (`GEOMETRY`), search (`CELL = 4`, `N_BINS = 8`, `DISC_RADIUS = 5000`, `MIN_ENERGY`, `MIN_VARIANCE`, `NUM_PEAKS = 25`, `TILE_STEP`, `DINO_BETA`), fine stage (`CONTEXT_MARGIN`, `USE_BORDER_MASK`, `CTX`, `MATCH_SIZE = 2·CTX`, `RANSAC_PIXELS`, `MAX_SHIFT_CELLS`, `MIN_INLIERS`, `ANSWER_MIN_INLIERS`).
Constraints: `CTX = TILE + 2·CONTEXT_MARGIN`; `LEVELS` must be sorted and each level must exist in the map pyramid; `ANSWER_MIN_INLIERS >= MIN_INLIERS` (see `result_thresholding/concepts/choosing_the_answer.md`). Derived values are computed in the file, so change only the base ones.

## `scoring.py`

| function | input | output |
|---|---|---|
| `sigma_m(altitude_m, focal_px)` | height (m), focal length (px) | σ in metres: `sqrt(8² + (3·h/f)²)` |
| `points_for_error(error_m, altitude_m, focal_px)` | error (m), height, focal | one of +10, +6, +3, 0, −20 |
| `final_score(points_list)` | list of per-photo points (0 for skipped) | `60·max(Σ/(10N), 0)` |
| `check_scoring()` | none | prints σ(1000 m) and the PS example, 28.8 |

Process, in the notation of the concept file: `e = error/σ`; tiers `e ≤ 1 → 10`, `≤ 2 → 6`, `≤ 4 → 3`, `≤ 8 → 0`, else `−20`. Skipped photos must be entered as `0` in the list, not dropped, because `N` is the number of test photos.

```python
PYTHONPATH=src python -c "
from header.scoring import sigma_m, points_for_error, final_score
print(sigma_m(100, 800), points_for_error(5, 100, 800), final_score([10, 10, -20, 0]))"
```
(Focal length 800 px is an example value; the real one is `camera.fx`.)

## `data.py`

* `read_metadata()` → pandas table of `images/metadata.csv`.
* `Camera(table)`: reads the nine intrinsics and **asserts they are identical for all photos** (spread < 1e-6), then stores one value each; the thermal frame size is fixed at 320 × 256. Constraint: a dataset with a changing camera needs one `Camera` per photo.
* `make_items(table)` → list of dicts. Angles `pitch` and `roll` go through `wrap_angle(a) = (a + 180) mod 360 − 180`, so 350° becomes −10°; this removes the wrap-around jump that breaks any later `abs(angle)` test. `yaw` is left as given.
* `assign_flights(items)` → numpy array of group names by height using `np.searchsorted(BAND_EDGES, h, side="right") − 1`; it also writes `item["flight"]`.
* `find_map_file()` → first `.tif` in `MAP_DIR` (alphabetical). Raises `FileNotFoundError` if none.

## `satellite_map.py`

`SatelliteMap(path)` opens the GeoTIFF with rasterio and stores size, the pixel size `gsd = file.res[0]`, a pyproj transformer (EPSG:4326 → the map's CRS) and the inverse affine transform. `gray_max` = 99.5th percentile of the non-zero values of a 1/16-size overview. Methods:

* `read_grey(window=None, out_shape=None, boundless=False)`: average of the bands `MAP_BANDS` as float; optional window and resampling to `out_shape`.
* `read_strip(top, rows)`: a band of full-width rows `top … top+rows` (used to build the pyramid without loading 306 million pixels at once).
* `grey_to_uint8(grey)`: `clip(grey/gray_max, 0, 1)·254 + 1` (values 1…255); pixels with grey ≤ 0 become 0 = "no map".
* `gps_to_map_pixel(lat, lon)` → `(row, col)`: pyproj with `always_xy=True` (note lon first) into the map CRS, then the inverse of the affine transform. Multiply by `gsd` to get metres (x = col·gsd, y = row·gsd).

## `sampling.py`

`pick_sample_ids(flight_of_frame, n_photos, seed)` → sorted numpy array of indices: `n_photos // len(FLIGHT_NAMES)` per group (min 1), without replacement, using `np.random.default_rng(seed)`. Constraint: if a group has fewer photos than requested, all of them are taken. Same inputs and seed always give the same ids; this is what makes two runs comparable.
