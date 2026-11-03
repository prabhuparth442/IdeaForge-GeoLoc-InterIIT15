# image_processing: turning two different pictures into comparable ones

Input: a thermal photo (320 × 256, any brightness range) with height and angles, and an orthophoto of the area. Output: (a) for the photo, a north-up 224 × 224 picture with the drone's nadir at its centre and a mask of which pixels are real; (b) for the map, a pyramid of five zoom levels and an 8-channel edge-direction feature grid per level. Everything here is deterministic and has no learned parameters.

## Concept files

| file | what it teaches | code it explains |
|---|---|---|
| [`pinhole_camera_and_ground_sample_distance.md`](concepts/pinhole_camera_and_ground_sample_distance.md) | projection `u = f·X/Z + c`, GSD = h/f | `camera.py`, `geometry.py` |
| [`lens_distortion.md`](concepts/lens_distortion.md) | radial and tangential terms `k1..k3, p1, p2` | `camera.py` |
| [`rotations_and_the_north_up_canvas.md`](concepts/rotations_and_the_north_up_canvas.md) | yaw/pitch/roll matrices, ray–ground intersection, inverse warping | `geometry.py` |
| [`thermal_normalisation.md`](concepts/thermal_normalisation.md) | percentile stretch and CLAHE | `thermal.py`, `map_levels.normalise_reference` |
| [`resampling_and_aliasing.md`](concepts/resampling_and_aliasing.md) | why shrinking needs averaging | `map_levels.py` |
| [`map_zoom_levels.md`](concepts/map_zoom_levels.md) | the ladder `0.3·2^(L/2)` and `level_for_height` | `map_levels.py` |
| [`edge_direction_features.md`](concepts/edge_direction_features.md) | gradient, direction mod π, 8 bins, cells, soft normalisation | `edge_features.py` |

## Flow

```
photo.png ─load_thermal_frame─► normalise_frame ─► photo (float 0..1, 256x320)
item(h, yaw, pitch, roll) + camera ─make_query─► rays → ground → canvas (224x224) + valid mask   ← one cv2.remap
canvas ─query_edge_features─► (features 8 x 56 x 56, cells 56 x 56 bool)

map (GeoTIFF) ─build_level_images─► 5 levels, uint8, 0 = no data ─build_level_pictures (normalise_reference)─► level images
level images ─map_edge_features─► 8 x H/4 x W/4 per level ─build_map_features─► features[level]
build_map_store(satellite_map) ─► {"levels": ..., "features": ...}
```

## How to run (synthetic examples need no data)

```bash
PYTHONPATH=src python -c "
import numpy as np
from image_processing.thermal import normalise_frame
f = normalise_frame(np.random.rand(256,320)*1000)
print(f.shape, f.min(), f.max())"         # (256, 320) ... 1.0
PYTHONPATH=src python -c "
from image_processing.map_levels import level_gsd, level_for_height
print([round(level_gsd(l),2) for l in range(3,8)])"   # [0.85, 1.2, 1.7, 2.4, 3.39]
```
The functions that need real files are exercised by `python run.py` (it builds the store and renders every test photo).

## `thermal.py`

* `load_thermal_frame(item)` → `cv2.imread(..., IMREAD_UNCHANGED)`; if 3 channels, averaged to 2-D. Input: `item["name"]` under `IMAGES_DIR`. Output: raw array, any dtype/range.
* `normalise_frame(frame, low_pct=1, high_pct=99)` → float32 in `[0,1]`.
Process: `x' = clip((x − p₁)/(p₉₉ − p₁ + ε), 0, 1)`; cast to 16 bit; CLAHE (`clipLimit = 2.0`, 8 × 8 tiles); divide by 65535. Percentile limits make the stretch immune to a few hot pixels; CLAHE then equalises contrast per 8 × 8 block with a limited slope so noise is not amplified without bound. Constraint: frame must not be constant (the `+ε` stops division by zero but the output is then all zero).

## `camera.py`

`distort(camera, x, y)`: for normalised coordinates `x = X/Z`, `y = Y/Z`, with `r² = x²+y²`:
`x_d = x(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁xy + p₂(r² + 2x²)` and the symmetric `y_d`. Pixel = `f·x_d + c`. This maps an undistorted ray to where the real lens put it, which is the direction we need for an inverse warp. Input/Output: arrays of the same shape.

## `geometry.py`

* `camera_axes(yaw_deg, pitch_deg, roll_deg)` → three world-frame unit vectors `(right, −forward, down)` of the tilted camera (world axes: east, north, up). Built by rotating `forward = (sin yaw, cos yaw, 0)` and `right = (cos yaw, −sin yaw, 0)` first about the right axis by pitch, then about the forward axis by roll. Their dot product with a world vector gives that vector's camera coordinates.
* `make_query(item, camera, canvas_size=TILE, pixel_size=1.0, geometry=None)` → `(canvas, valid)`:
  1. normalise the photo; if one canvas pixel is bigger than a photo pixel (`pixel_size > photo_gsd`, with `photo_gsd = h/fx`), blur the photo by `σ = 0.5·pixel_size/photo_gsd − 0.25` first, so shrinking does not alias (see `resampling_and_aliasing.md`);
  2. height used = `(rel_height + height_offset)·scale`; heading = `yaw + yaw_offset`; pitch and roll are multiplied by their sign (or ignored if `attitude` is false);
  3. for every canvas pixel: east = (col − middle)·pixel_size, north = −(row − middle)·pixel_size, below = −height; camera coordinates `x_cam, y_cam, z_cam` = dot products with the three axes; `in_front = z_cam > 0.001`;
  4. perspective divide `x_cam/z_cam, y_cam/z_cam` → `distort` → `pixel = f·x_d + c`;
  5. one `cv2.remap` (bilinear) reads the photo at those positions; a second remap of a ones-image tells which canvas pixels fall inside the photo (`> 0.99`); `valid = inside & in_front`; invalid pixels are filled with the photo mean.
* `render_query(item, camera, level, geometry=None)` = `make_query(item, camera, TILE, level_gsd(level), geometry)`.

Inputs: `geometry` dict (`attitude`, `pitch_sign`, `roll_sign`, `yaw_offset`, `scale`, `height_offset`) defaults to `settings.GEOMETRY`. Outputs: `canvas` float32 (224, 224), `valid` bool (224, 224). Constraint: the camera must look downward; a tilt near 90° makes the ground intersection ill-defined (the `in_front` mask handles it by marking such pixels invalid).

## `map_levels.py`

* `level_gsd(L) = MAP_GSD·2^(L/2)`; `level_scale(L) = MAP_GSD/level_gsd(L)` (shrink factor from the 0.3 m map).
* `level_for_height(height, camera)`: `footprint = camera.width·height/camera.fx` (ground width of the photo), `level = ceil(2·log2(footprint/(TILE·MAP_GSD)))`, clipped to `[3, 7]`: the finest level whose 224-pixel tile still covers the footprint. 100 m → 3, 150 → 4, 200 → 5, 250/300 → 6, 350 → 7.
* `build_level_images(satellite_map, levels, strip_rows=2048)` → `{level: uint8 image}`. Allocates each level at `int(size·scale)`, then walks the map in strips of 2048 rows; each strip is shrunk with `cv2.INTER_AREA` (area averaging, so no aliasing) to the matching rows `round(top·scale) … round((top+rows)·scale)` and converted with `grey_to_uint8`. Strip boundaries are rounded to whole output rows, giving a sub-pixel misplacement (documented in `resampling_and_aliasing.md`).
* `normalise_reference(image)`: CLAHE (clip 2.0, tile grid ≈ 28 px per tile so the grid is `(W//28, H//28)`) on the uint8 level, result forced to ≥ 1 where there is map, 0 where there is none.
* `build_level_pictures(satellite_map)` → `{level: normalised uint8 picture}` for all `LEVELS`.
* `map_piece(level_images, level, x_m, y_m, size=TILE)` → float32 `(size, size)` in `[0, 1]` (divided by 255), centred on metres `(x_m, y_m)` at that level, zero outside the map.

## `edge_features.py`

* `gradient_parts(image)`: Gaussian blur (σ = 1 px), central differences `gx = I(x+1) − I(x−1)`, `gy` likewise (borders 0), strength `√(gx²+gy²)`, direction `θ = atan2(gy, gx) mod π` (an edge has no sign), bin `⌊θ/π·8⌋`.
* `edge_features(image)` → `(8, H//4, W//4)` float32: for each bin, the sum of strengths of the pixels in that bin inside each 4 × 4 cell (reshape-and-sum, no loops over pixels).
* `normalise_cells(features, cells_to_use=None)`: divides every cell's 8-vector by `length + typical + 0.001`, where `length = ‖vector‖` and `typical` = 30th percentile of the lengths over the cells in use. "Soft": a cell with strong edges becomes nearly unit length; a cell with weak edges stays small instead of being blown up to unit length.
* `query_edge_features(canvas, valid)` → `(features, cell_valid)`: validity mask eroded with a 7 × 7 box, a cell is valid when ≥ 80 % of its pixels survive; features of invalid cells are set to 0.
* `no_data_cells(image)`: a cell is "no data" if any of its pixels is 0, grown by one cell (3 × 3 dilation) so cells touching the border of the map are excluded too.
* `map_edge_features(image)`: computed in strips of 1024 rows with 8 rows of padding (so no seam at strip borders), no-data cells set to 0, then normalised with one global `typical` taken from every 50th non-zero length.
* `build_map_features(level_images)` → `{level: features}`.

Shapes at the default setting: the query features are `(8, 56, 56)`; a level-3 map of the whole area is several thousand cells on each side, stored as float32.

## `map_store.py`

`build_map_store(satellite_map)` → `{"levels": {L: uint8 (H_L, W_L)}, "features": {L: float32 (8, h_L, w_L)}}`; prints one line per level (`m/pixel`, size, coverage %) and the total build time. Rebuilt on every run; it is the slowest set-up step (reads the full 2.6 GB map). Requires the map at `MAP_DIR`; memory is about the size of the five features arrays.
