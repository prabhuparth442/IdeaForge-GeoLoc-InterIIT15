# header: the shared rulebook, the data readers and the scorer

This folder is the **foundation** of the project. It does no image processing and no searching. It answers four plain questions that every other part needs:

1. *What are the settings?* (`settings.py`)
2. *How do we judge an answer?* (`scoring.py`)
3. *How do we read the photos and their metadata?* (`data.py`, `sampling.py`)
4. *How do we read the satellite map and convert GPS to map pixels?* (`satellite_map.py`)

If a number looks magic anywhere else in the project, the place to look first is `settings.py`.

## Learn the ideas first (concept notes, with worked examples)

| order | note | after reading you can explain |
|---|---|---|
| 1 | [`concepts/scoring_rule.md`](concepts/scoring_rule.md) | GSD, σ, the five point tiers, why a wrong answer must be rare (the 2/3 rule) |
| 2 | [`concepts/geographic_coordinates_and_map_reading.md`](concepts/geographic_coordinates_and_map_reading.md) | how a GPS position becomes a map pixel and a "map metre", why `always_xy=True`, why 0 means "no map" |
| 3 | [`concepts/stratified_sampling_and_seeds.md`](concepts/stratified_sampling_and_seeds.md) | why we test on equal numbers of low/medium/high photos, what a seed is, why angles are wrapped |

## How the pieces connect

```
metadata.csv ──read_metadata──► table ──make_items──► items          (one dict per photo)
                                  │                       │
                                  └── Camera(table)       └── assign_flights ──► flight_of_frame ──pick_sample_ids──► sample_ids
                                         (9 constants)
map.tif ──find_map_file──► path ──SatelliteMap──► map object (gsd, gray_max, read_strip, gps_to_map_pixel)

error in metres + height ──points_for_error──► points ──final_score──► score out of 60
```

## How to run the pieces

These files have no `main`. Import them with `src` on the path, from the repository root:

```bash
PYTHONPATH=src python -c "from header.scoring import check_scoring; check_scoring()"
```

Expected output (the check that the scoring code matches the problem statement):

```
sigma at 1000 m with a 200-pixel focal length: 17.0 m (statement: about 17 m)
statement's example score: 28.8 (statement: 28.8)
```

`python run.py` calls this first, because a wrong scoring file would silently corrupt every later number.

## File by file

### `settings.py`: every constant in one place

There are no functions here, only values. Groups:

| group | names | meaning |
|---|---|---|
| paths | `IMAGES_DIR`, `MAP_DIR`, `MINIMA_FILE` | where the data and the matcher weights are |
| run switches | `N_PHOTOS`, `USE_DINO`, `USE_MINIMA`, `SEED` | how much to run and what to switch on |
| height groups | `BAND_EDGES`, `FLIGHT_NAMES` | the three height groups used for sampling |
| map reading | `MAP_BANDS`, `MAP_GSD`, `STRIP_ROWS` | which bands, pixel size 0.3 m, rows per strip |
| pyramid | `LEVELS = [3,4,5,6,7]`, `TILE = 224` | zoom levels and the standard picture size |
| geometry guess | `GEOMETRY` | yaw offset 180°, scale 1.04, tilt signs found on the dev set |
| search | `CELL = 4`, `N_BINS = 8`, `DISC_RADIUS = 5000`, `NUM_PEAKS = 25`, `MIN_ENERGY`, `MIN_VARIANCE`, `TILE_STEP`, `DINO_BETA` | the coarse search |
| fine stage | `CONTEXT_MARGIN`, `USE_BORDER_MASK`, `CTX`, `MATCH_SIZE`, `RANSAC_PIXELS`, `MAX_SHIFT_CELLS`, `MIN_INLIERS`, `ANSWER_MIN_INLIERS` | the matcher and the decision |

Rules to respect when editing: `CTX = TILE + 2·CONTEXT_MARGIN`; `MATCH_SIZE = 2·CTX`; `ANSWER_MIN_INLIERS ≥ MIN_INLIERS` (explained in `result_thresholding/concepts/choosing_the_answer.md`). Derived values are computed inside the file, so change only the base ones.

The `GEOMETRY` values are **found on the official development set** (see `evaluation/concepts/geometry_hypothesis_tests.md`). A different dataset can use different conventions, so re-run `python run.py --sweep` when the data changes.

### `scoring.py`

| function | input | output |
|---|---|---|
| `sigma_m(altitude_m, focal_px)` | height (m), focal length (pixels) | σ in metres |
| `points_for_error(error_m, altitude_m, focal_px)` | error (m), height (m), focal (px) | one of +10, +6, +3, 0, −20 |
| `final_score(points_list)` | per-photo points; **0 for skipped photos** | score on the 0–60 scale |
| `check_scoring()` | none | prints σ(1000 m) and the problem-statement example (28.8) |

Process in one line: `σ = √(8² + (3·h/f)²)`, `ratio = error/σ`, tier by ratio, `score = 60·max(Σpoints/(10N), 0)`. All derivations are in `scoring_rule.md`.

A quick feel for it:

```bash
PYTHONPATH=src python -c "
from header.scoring import sigma_m, points_for_error, final_score
print(round(sigma_m(100, 200), 2))                       # 8.14
print([points_for_error(e, 100, 200) for e in (6, 15, 30, 90)])   # [10, 6, 3, -20]
print(final_score([10, 6, 3, -20]))                      # 0   (one bad answer cancels the good ones)"
```

Constraints: `focal_px` in pixels (not millimetres); `altitude_m > 0`; the list must not be empty and must contain only 10, 6, 3, 0, −20.

### `data.py`

| name | what it does | notes |
|---|---|---|
| `read_metadata()` | reads `IMAGES_DIR/metadata.csv` into a table | |
| `Camera(table)` | stores `fx, fy, cx, cy, k1, k2, k3, p1, p2`, width 320, height 256 | asserts all rows have the same camera numbers |
| `wrap_angle(a)` | `(a + 180) mod 360 − 180` | 350° → −10° |
| `make_items(table)` | one dict per photo: `name, lat, lon, rel_height, yaw, pitch, roll` | pitch and roll are wrapped |
| `assign_flights(items)` | height group per photo (also stored as `item["flight"]`) | heights must be in [100, 351) |
| `find_map_file()` | first `.tif` in `MAP_DIR` | raises `FileNotFoundError` if none |

### `sampling.py`

`pick_sample_ids(flight_of_frame, n_photos, seed)` returns a sorted array of photo indices: `n_photos // 3` from each height group, without repeats. The same inputs and seed always give the same result. If a group has fewer photos than requested, all of its photos are taken.

```bash
PYTHONPATH=src python -c "
import numpy as np
from header.sampling import pick_sample_ids
groups = np.array(['100-150 m']*50 + ['150-250 m']*50 + ['250-350 m']*50)
ids = pick_sample_ids(groups, 12, 7)
print(len(ids), [int((groups[ids]==g).sum()) for g in ['100-150 m','150-250 m','250-350 m']])   # 12 [4, 4, 4]"
```

### `satellite_map.py`

`SatelliteMap(path)` opens the GeoTIFF with rasterio and stores the size, the pixel size `gsd = file.res[0]`, a pyproj transformer from GPS to the map's projection, and the inverse affine transform. It also computes `gray_max` = the 99.5th percentile of the non-zero values of a 1/16-size overview.

| method | input | output |
|---|---|---|
| `read_grey(window, out_shape, boundless)` | a Window or a target shape | float32 grey (mean of the bands) |
| `read_strip(top, rows)` | first row, number of rows | the full-width strip as grey |
| `grey_to_uint8(grey)` | float grey 2-D array | uint8: `clip(g/gray_max,0,1)·254 + 1`, and 0 where there is no map |
| `gps_to_map_pixel(lat, lon)` | degrees, **latitude first** | `(row, col)` floats; multiply by `gsd` for map metres |

Needs the `rasterio` and `pyproj` packages and a real GeoTIFF; it cannot run on made-up data without a file.

Convention to remember: after loading, positions are **map metres from the top-left corner, `x` east, `y` south** (`x = col·gsd`, `y = row·gsd`).

## Input checks

Each function starts with `assert` lines stating what it accepts (for example "latitude must be in degrees −90..90", "points must be one of 10, 6, 3, 0, −20"). A wrong input stops here with a message instead of producing wrong numbers later.

## Where things go next

Items (photos) and the camera go to `image_processing/`. Sample ids and the test set go to `evaluation/`. The map object goes to `image_processing/map_store.py`, which builds the zoom levels.
