# image_processing: turning two different pictures into comparable ones

## What this section does, in plain words

We start with two pictures that look nothing alike:

* a **thermal photo** (320 × 256 pixels, heat instead of colour, tilted, bent by the lens, pointing wherever the drone's nose pointed), and
* a huge **satellite map** (north-up, straight down, 0.3 m pixels).

This section converts both into the same kind of object so that the next section can search for the photo on the map:

```
THERMAL PHOTO ──► north-up canvas, 224 × 224, nadir at the middle ──► edge-direction cells  (8 × 56 × 56)
SATELLITE MAP ──► five zoom levels ──► edge-direction cells per level (8 × H/4 × W/4)
```

Both outputs are arrays of the same kind ("8 direction bins in each 4 × 4 block"), so they can be compared even though one is heat and the other is sunlight. Nothing here is learned: every step is a formula you can follow by hand.

## Learn the ideas first: read the notes in this order

| step | note | one-line takeaway |
|---|---|---|
| 1 | [`pinhole_camera_and_ground_sample_distance.md`](concepts/pinhole_camera_and_ground_sample_distance.md) | A camera maps ground points to pixels by `u = f·X/Z + c`; one pixel covers `h/f` metres |
| 2 | [`lens_distortion.md`](concepts/lens_distortion.md) | Real lenses bend the picture (our corners by about 46 pixels); five numbers describe it |
| 3 | [`rotations_and_the_north_up_canvas.md`](concepts/rotations_and_the_north_up_canvas.md) | For each canvas pixel we compute where in the photo to read; the nadir point, not the photo centre, is the canvas middle |
| 4 | [`thermal_normalisation.md`](concepts/thermal_normalisation.md) | Percentile stretch removes gain and offset exactly; CLAHE brings out local detail |
| 5 | [`resampling_and_aliasing.md`](concepts/resampling_and_aliasing.md) | Blur or average before shrinking, or fake patterns appear |
| 6 | [`map_zoom_levels.md`](concepts/map_zoom_levels.md) | Five map pictures with pixel size `0.3·2^(L/2)`; each photo uses the level that fits its height |
| 7 | [`edge_direction_features.md`](concepts/edge_direction_features.md) | Compare the direction of edges, not brightness; 8 bins in 4 × 4 cells |

## The data flow with shapes

```
photo.png (uint16, 256×320)
   │ load_thermal_frame            (thermal.py)
   ▼
raw array ── normalise_frame ──► float32 0..1, 256×320
   │
   │ make_query / render_query     (geometry.py)   needs: height, yaw, pitch, roll, camera, zoom level
   ▼
canvas  float32 (224, 224)   +   valid  bool (224, 224)         ← the photo as a north-up map piece
   │ query_edge_features           (edge_features.py)
   ▼
features float32 (8, 56, 56)  +  cell_valid bool (56, 56)       ← goes to coarse_search

map.tif
   │ SatelliteMap.read_strip       (header)
   ▼
build_level_images ──► {3,4,5,6,7: uint8 picture, 0 = no map}   (map_levels.py)
   │ normalise_reference  (CLAHE)
   ▼
level pictures ── map_edge_features ──► {level: float32 (8, H/4, W/4)}
   │
   ▼
build_map_store ──► {"levels": ..., "features": ...}           ← goes to coarse_search (features) and fine_matching (levels)
```

## How to try the pieces (no dataset needed)

```bash
# run from the repository root
PYTHONPATH=src python -c "
import numpy as np
from image_processing.thermal import normalise_frame
f = normalise_frame(np.random.rand(256, 320) * 1000)
print(f.shape, float(f.min()), float(f.max()))"                      # (256, 320) 0.0015... 1.0

PYTHONPATH=src python -c "
from image_processing.map_levels import level_gsd, level_for_height
class Camera: fx = 200.0; width = 320
print([round(level_gsd(l), 2) for l in range(3, 8)])                   # [0.85, 1.2, 1.7, 2.4, 3.39]
print([level_for_height(h, Camera()) for h in (100, 150, 200, 250, 300, 350)])   # [3, 4, 5, 6, 6, 7]"

PYTHONPATH=src python -c "
import numpy as np
from image_processing.edge_features import edge_features, gradient_parts
x = np.arange(32)[None, :].repeat(32, 0).astype(np.float32)
edge = (x >= 16).astype(np.float32)                                   # a vertical edge
bin_id = gradient_parts(edge)[5]
print(bin_id[10, 15], edge_features(edge).shape)                      # 0 (8, 8, 8): bin 0 = vertical edge"
```

The functions that need the real photos or map (`load_thermal_frame`, `make_query`, `SatelliteMap`, `build_map_store`) run through `python run.py`.

## File by file

### `thermal.py`: one photo to a normalised picture

| function | input | output |
|---|---|---|
| `load_thermal_frame(item)` | photo record (`item["name"]`) | raw 2-D array in the file's own range (16-bit) |
| `normalise_frame(frame, low_pct=1, high_pct=99)` | 2-D array | float32 in `[0, 1]`, same shape |

Process: `x' = clip((x − p₁)/(p₉₉ − p₁ + ε), 0, 1)`, convert to 16-bit integers, CLAHE (`clipLimit` 2.0, 8 × 8 tiles), divide by 65535. The photo is read with `cv2.IMREAD_UNCHANGED` so the 16-bit depth is kept. Constraint: the picture must not be constant; the file must exist.

### `camera.py`: the lens model

`distort(camera, x, y)`: normalised coordinates in, distorted normalised coordinates out (arrays of the same shape).
`x_d = x(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁xy + p₂(r² + 2x²)`, with the symmetric formula for `y_d`, where `r² = x² + y²`. The pixel is then `f·x_d + c`. It is applied forwards to canvas pixels, which undistorts the photo for free. Explained with real numbers in `lens_distortion.md`.

### `geometry.py`: the north-up canvas

| function | input | output |
|---|---|---|
| `camera_axes(yaw_deg, pitch_deg, roll_deg)` | three angles in degrees | three world-frame unit vectors: image-right, image-down, viewing axis |
| `make_query(item, camera, canvas_size, pixel_size, geometry)` | photo record, camera, canvas side in pixels, metres per canvas pixel, optional overrides | `canvas` float32 `(S, S)` in `[0, 1]`, `valid` bool `(S, S)` |
| `render_query(item, camera, level, geometry)` | same, with size and pixel size from the zoom level | the same pair with `S = 224` |

Process inside `make_query`, in the order of the code:

1. normalise the photo; if the canvas pixel is larger than a photo pixel, blur first with `σ = 0.5·pixel_size/photo_gsd − 0.25`;
2. height used = `(rel_height + height_offset)·scale`; heading = `yaw + yaw_offset`; pitch and roll multiplied by their signs (or ignored if `attitude` is off);
3. for every canvas pixel: ground point `(east, north, −height)` relative to the nadir point;
4. dot products with the three camera axes give `(Xc, Yc, Zc)`; `in_front = Zc > 0.001`;
5. `x = Xc/Zc`, `y = Yc/Zc`, then `distort`, then `pixel = f·x_d + c`;
6. one `cv2.remap` reads the photo; a second remap of a ones-picture gives the mask (`> 0.99`); empty pixels get the photo mean.

`geometry` is a partial dictionary of overrides on top of `settings.GEOMETRY` (`attitude`, `pitch_sign`, `roll_sign`, `yaw_offset`, `scale`, `height_offset`). Constraints: the camera must look roughly downward (extreme tilt makes `Zc ≤ 0` and the valid area collapses); `level` must be one of `settings.LEVELS`.

### `map_levels.py`: the zoom pyramid

| function | what it does |
|---|---|
| `level_gsd(L)` | pixel size `0.3·2^(L/2)` m |
| `level_scale(L)` | shrink factor `0.3 / level_gsd(L)` |
| `level_for_height(height, camera)` | `ceil(2·log2(footprint / (TILE·MAP_GSD)))`, clipped to 3..7 |
| `build_level_images(satellite_map, levels, strip_rows)` | reads the map in strips of 2048 rows, shrinks each with area averaging, returns `{level: uint8}` |
| `normalise_reference(image)` | CLAHE with about 28-pixel tiles; real pixels stay ≥ 1, no-map stays 0 |
| `build_level_pictures(satellite_map)` | the two above for all levels |
| `map_piece(level_images, level, x_m, y_m, size)` | a `size × size` float piece in `[0, 1]` centred on map metres `(x_m, y_m)`; zeros outside the map |

### `edge_features.py`: the comparison format

| function | input | output |
|---|---|---|
| `gradient_parts(image)` | 2-D picture, ≥ 3 × 3 | blurred picture, `gx`, `gy`, strength, angle, bin id |
| `edge_features(image)` | 2-D picture, ≥ 4 × 4 | `(8, H//4, W//4)` sums of edge strength per direction bin and cell |
| `normalise_cells(features, cells_to_use)` | `(8, rows, cols)` | each cell divided by `length + typical + 0.001` |
| `query_edge_features(canvas, valid)` | square canvas (side a multiple of 4), bool mask | `(8, 56, 56)` features, `(56, 56)` cell mask (cells ≥ 80 % valid after a 7 × 7 erosion) |
| `no_data_cells(image)` | uint8 level picture | cells that contain a 0 pixel, grown by one cell |
| `map_edge_features(image)` | uint8 level picture, must contain map | `(8, H//4, W//4)`, processed in 1024-row strips with 8 rows of padding, no-data cells zeroed, one global `typical` |
| `build_map_features(level_images)` | `{level: uint8}` | `{level: features}` |

### `map_store.py`: everything the search needs, built once

`build_map_store(satellite_map)` returns `{"levels": {L: uint8 (H_L, W_L)}, "features": {L: float32 (8, h_L, w_L)}}`. It prints one line per level (`m/pixel`, size, coverage) and the build time. It checks that the map's pixel size equals `settings.MAP_GSD`. Built on every run for now; it is the slowest set-up step because it reads the 2.6 GB map.

## Input checks

Every function starts with `assert` lines that state what it accepts, for example "frame must be a non-empty 2-D array", "canvas must be square", "level 2 is not one of [3, 4, 5, 6, 7]". They are your quick reference for the constraints above.

## What goes wrong most often

* Geometry values (`yaw_offset`, `scale`, signs) copied to a new dataset without testing; the result is a canvas that does not look like the map, and everything downstream fails quietly. Re-run `python run.py --sweep`.
* A photo height outside the range of the zoom levels (the level is clipped, the photo gets cropped).
* Treating the value 0 in a level picture as "dark ground": it means "no map".
