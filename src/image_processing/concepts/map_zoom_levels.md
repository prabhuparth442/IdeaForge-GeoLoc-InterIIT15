# The map at several zoom levels

Code: `src/image_processing/map_levels.py` (`level_gsd`, `level_scale`, `level_for_height`, `build_level_images`, `normalise_reference`, `build_level_pictures`, `map_piece`) and `map_store.py`.

---

## 1. Why more than one resolution

The satellite map has 0.3 m pixels. A drone photo from 100 m has 0.5 m pixels, from 350 m 1.75 m pixels. Comparing a 1.75 m photo with a 0.3 m map would make the map's detail useless noise and cost 34× more computation (`(1.75/0.3)² ≈ 34`) for nothing. So we prepare the map **once, in advance**, at several coarser resolutions, and for each photo use the level whose pixel size suits it.

## 2. Definition of a level

Level `L` shrinks the map so that one pixel is

$$g_L=g_0\cdot2^{L/2}\qquad(g_0=0.3\text{ m}).$$

Levels differ by a factor `√2` in pixel size ("half an octave"), so the ladder is dense enough that the level chosen for a photo (section 3) never has a tile more than 41 % bigger than the photo needs.

| L | pixel size g_L | feature cell (4 px) | one 224-px tile covers | full map picture |
|---|---|---|---|---|
| 3 | 0.85 m | 3.4 m | 190 m | 7812 × 4893 |
| 4 | 1.20 m | 4.8 m | 269 m | 5524 × 3460 |
| 5 | 1.70 m | 6.8 m | 380 m | 3906 × 2446 |
| 6 | 2.40 m | 9.6 m | 538 m | 2762 × 1730 |
| 7 | 3.39 m | 13.6 m | 760 m | 1953 × 1223 |

The shrink factor from the original map to level `L` is `scale = 2^{−L/2}`.

## 3. Choosing the level for a photo

The photo's ground footprint is `W·h/f` (see `pinhole_camera_and_ground_sample_distance.md`). We render the canvas as `TILE = 224` pixels of size `g_L`, covering `224·g_L` metres. The canvas must contain the whole photo footprint:

$$224\cdot g_0\cdot2^{L/2}\ \ge\ \frac{W\,h}{f}\ \Longleftrightarrow\ L\ \ge\ 2\log_2\frac{W\,h}{224\,g_0\,f}.$$

Taking the smallest integer that satisfies it (ceiling):

```python
footprint = camera.width * height / camera.fx
level = math.ceil(2 * math.log2(footprint / (settings.TILE * settings.MAP_GSD)))
return int(np.clip(level, min(settings.LEVELS), max(settings.LEVELS)))
```

Worked values (f = 200): h = 100 → footprint 160 m → ratio 160/67.2 = 2.38 → 2·log₂ = 2.50 → **L = 3**; h = 200 → 320/67.2 = 4.76 → 4.50 → **L = 5**; h = 350 → 560/67.2 = 8.33 → 6.12 → **L = 7**. The final `clip` keeps the answer inside the levels we built (3 to 7); heights above ~400 m would silently be forced to level 7 and the photo would be cropped by the canvas. Add levels in `settings.LEVELS` if you ever fly higher.

## 4. Building the pyramid without loading the map

```python
for top in range(0, satellite_map.height, strip_rows):          # strips of 2048 rows
    grey = satellite_map.read_strip(top, rows)                  # read + average the bands
    for level in levels:
        small = cv2.resize(grey, (out_w, y1 - y0), interpolation=cv2.INTER_AREA)
        images[level][y0:y1] = satellite_map.grey_to_uint8(small)
```

* **Input:** a `SatelliteMap`; `levels` such as `[3,4,5,6,7]`.
* **Output:** dict `level → uint8 array`, shape `(map_height·scale, map_width·scale)`, `0` = no map.
* Details of why `INTER_AREA` and how strips are stitched: `resampling_and_aliasing.md`.

## 5. Contrast normalisation of the map (`normalise_reference`)

The thermal photo received CLAHE (`thermal_normalisation.md`). The map gets the same kind of local contrast boost so both live on a comparable scale:

```python
has_map = image > 0
grid = (max(image.shape[1] // 28, 1), max(image.shape[0] // 28, 1))   # tiles of about 28 pixels
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid)
result = np.maximum(clahe.apply(image), 1)    # real pixels never become 0
result[~has_map] = 0                          # but empty stays empty
```

`tileGridSize` is the *number* of tiles, so `size // 28` tiles gives tiles ≈ 28 px wide. A 224-pixel canvas contains 224/28 = 8 such regions per side, the same 8 × 8 regions per picture the thermal photo's CLAHE uses. That is the reason for the number 28: the contrast scale matches.

The `np.maximum(..., 1)` and `result[~has_map] = 0` pair keeps the rule "0 means no map and nothing else is ever 0". CLAHE could otherwise turn a very dark real pixel into 0.

## 6. Cutting a piece: `map_piece`

```python
def map_piece(level_images, level, x_m, y_m, size=settings.TILE):
    gsd = level_gsd(level)
    top  = int(round(y_m / gsd - size / 2))
    left = int(round(x_m / gsd - size / 2))
    piece = np.zeros((size, size), dtype=np.float32)
    ... copy the part that lies inside the picture, divide by 255 ...
```

Given a centre in **map metres** (x east, y south), it returns a `size × size` float picture in `[0,1]` centred there. Parts outside the map stay 0. `x_m / gsd` converts metres to level pixels; subtracting `size/2` gets the top-left corner. Used by the fine matcher (`fine_matching/second_stage.py`) to cut the tile around each candidate.

## 7. `build_map_store`

`map_store.build_map_store(satellite_map)` returns

```python
store = {"levels": {L: uint8 picture}, "features": {L: float32 (8, rows/4, cols/4)}}
```

*Levels* are built here (steps above); *features* are built from the levels by `edge_features.build_map_features` (next file). Both are computed **once** per run before any photo is processed. It takes a few minutes on the 2.6 GB map; the time does not count toward a photo's 30 s budget in flight, because the final system would load a pre-built store from disk (not implemented yet: the store is rebuilt each time `run.py` starts).
