# The map at several zoom levels

**Code:** `level_gsd`, `level_scale`, `level_for_height`, `build_level_images`, `normalise_reference`, `build_level_pictures`, `map_piece` in `src/image_processing/map_levels.py`, and `build_map_store` in `src/image_processing/map_store.py`.

**What you need to know beforehand:** GSD (metres per pixel) and the footprint of a photo, from `pinhole_camera_and_ground_sample_distance.md`; shrinking pictures, from `resampling_and_aliasing.md`.

---

## 1. The big picture in plain words

The satellite map has very fine pixels (0.3 m). A photo taken at 100 m has 0.5 m pixels, at 350 m 1.75 m pixels. Comparing a 1.75 m photo with a 0.3 m map would waste effort: the map's fine detail would be noise, and the computer would do about 34 times more work (`(1.75/0.3)² ≈ 34`) for nothing.

So we prepare the map **once, in advance**, at five coarser resolutions (a **pyramid** of zoom levels), and for each photo we use the level whose pixel size fits it. Think of it as the zoom setting on a map application: you zoom out when you look at a large area.

## 2. What a level is

Level `L` shrinks the map so that one pixel is

$$g_L=g_0\cdot 2^{L/2}\qquad(g_0=0.3\ \text{m}).$$

Neighbouring levels differ by a factor `√2` in pixel size ("half an octave"). That is a dense ladder: the level chosen for a photo (Section 3) never has a tile more than 41 % larger than the photo needs (`√2 = 1.41`).

| L | pixel size `g_L` | feature cell (4 px) | one 224-px tile covers | whole map as one picture |
|---|---|---|---|---|
| 3 | 0.85 m | 3.4 m | 190 m | 7812 × 4893 |
| 4 | 1.20 m | 4.8 m | 269 m | 5524 × 3460 |
| 5 | 1.70 m | 6.8 m | 380 m | 3906 × 2446 |
| 6 | 2.40 m | 9.6 m | 538 m | 2762 × 1730 |
| 7 | 3.39 m | 13.6 m | 760 m | 1953 × 1223 |

The shrink factor from the original map to level `L` is `scale = 2^(−L/2)` (level 3: 0.354, level 7: 0.088). In the code: `level_gsd(L)` returns `g_L`, `level_scale(L)` returns `0.3 / g_L`.

## 3. Choosing the level for a photo

The photo's ground width is `W·h/f` (its footprint). We will render the canvas as `TILE = 224` pixels of size `g_L`, covering `224·g_L` metres. The canvas must contain the whole footprint:

$$224\cdot g_0\cdot 2^{L/2}\ \ge\ \frac{W\,h}{f}\quad\Longleftrightarrow\quad L\ \ge\ 2\log_2\frac{W\,h}{224\,g_0\,f}.$$

(Take the logarithm base 2 of both sides to see this: `L/2 ≥ log₂(Wh / (224·g₀·f))`.) The smallest whole number that satisfies it is the ceiling:

```python
footprint = camera.width * height / camera.fx
level = math.ceil(2 * math.log2(footprint / (settings.TILE * settings.MAP_GSD)))
return int(np.clip(level, min(settings.LEVELS), max(settings.LEVELS)))
```

**Worked values** (`f = 200`, `224 · 0.3 = 67.2 m`):

| height | footprint `320·h/200` | ratio to 67.2 | `2·log₂(ratio)` | level (ceiling, 3..7) |
|---|---|---|---|---|
| 100 m | 160 m | 2.38 | 2.50 | **3** |
| 150 m | 240 m | 3.57 | 3.67 | **4** |
| 200 m | 320 m | 4.76 | 4.50 | **5** |
| 250 m | 400 m | 5.95 | 5.15 | **6** |
| 350 m | 560 m | 8.33 | 6.12 | **7** |

The final `np.clip` keeps the answer inside the levels we built. Above roughly 475 m the footprint no longer fits level 7's tile (760 m wide), and the photo would be cropped; add levels to `settings.LEVELS` if you ever need that.

## 4. Building the pyramid without loading the map

```python
for top in range(0, satellite_map.height, strip_rows):          # strips of 2048 rows
    grey = satellite_map.read_strip(top, rows)                  # read the bands, average them to grey
    for level in levels:
        small = cv2.resize(grey, (out_w, y1 - y0), interpolation=cv2.INTER_AREA)
        images[level][y0:y1] = satellite_map.grey_to_uint8(small)
```

* **Input:** a `SatelliteMap` and the list of levels.
* **Output:** a dictionary `level → uint8 picture` with shape `(map_height·scale, map_width·scale)`; `0` means "no map".
* The map is read one strip at a time (memory), and each strip is shrunk to every level before being discarded. Why `INTER_AREA`, and how strips are stitched: `resampling_and_aliasing.md`.

## 5. Matching the contrast of the map to the photo (`normalise_reference`)

The thermal photo got CLAHE (`thermal_normalisation.md`). The map gets the same kind of local contrast boost, so both pictures live on a comparable scale:

```python
has_map = image > 0
grid = (max(image.shape[1] // 28, 1), max(image.shape[0] // 28, 1))   # about 28-pixel tiles
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid)
result = np.maximum(clahe.apply(image), 1)    # real pixels never become 0
result[~has_map] = 0                          # empty stays empty
```

`tileGridSize` is the **number** of tiles (not their size). With `size // 28` tiles, each tile is about 28 pixels wide. A 224-pixel canvas contains `224/28 = 8` such tiles per side, the same 8 × 8 layout that the photo's CLAHE uses. That is where the number 28 comes from: the contrast scale matches.

The pair `np.maximum(..., 1)` and `result[~has_map] = 0` protects the rule "0 means no map and nothing else is ever 0". Without it, CLAHE could turn a very dark real pixel into 0 and make real ground look like missing data.

## 6. Cutting a piece: `map_piece`

```python
def map_piece(level_images, level, x_m, y_m, size=settings.TILE):
    gsd = level_gsd(level)
    top  = int(round(y_m / gsd - size / 2))
    left = int(round(x_m / gsd - size / 2))
    piece = np.zeros((size, size), dtype=np.float32)
    ... copy the part that lies inside the picture, divide by 255 ...
```

You give a centre in **map metres** (x east, y south). `x_m / gsd` converts metres to level pixels, and subtracting `size/2` moves to the top-left corner. The result is a `size × size` float picture in `[0, 1]`; parts outside the map stay 0.

**Example.** Level 3 (`gsd = 0.85`), centre `(x, y) = (1000 m, 500 m)`, `size = 224`: `left = round(1000/0.85 − 112) = round(1176.5 − 112) = 1064`, `top = round(500/0.85 − 112) = round(588.2 − 112) = 476`. The piece is image rows 476..699 and columns 1064..1287.

This is used by the fine matcher (`fine_matching/second_stage.py`) to cut the tile around each candidate.

## 7. `build_map_store`

```python
store = {"levels": {L: uint8 picture}, "features": {L: float32 (8, rows/4, cols/4)}}
```

The *levels* are built by the steps above; the *features* are built from the levels by `edge_features.build_map_features` (next note). Both are computed once per run, before any photo is processed. It takes a few minutes on the 2.6 GB map. In the final system a prebuilt store would be loaded from disk, which would take no time per photo; saving and loading the store is not implemented yet, so `run.py` rebuilds it every time.

## 8. Mistakes to avoid

* Giving `map_piece` coordinates in pixels or metres from the wrong origin. They are map metres, `x` east and `y` south.
* Forgetting the `0 = no map` convention when computing statistics on a level.
* Mixing levels: a canvas rendered for level 5 must be matched against level-5 features.

## 9. Check yourself

1. Which level for a 300 m photo? *(footprint 480 m; ratio 7.14; `2·log₂ = 5.67` → level 6.)*
2. What is the feature cell size at level 4? *(4 × 1.2 = 4.8 m.)*
3. Why two different maps in the store, `levels` and `features`? *(Levels are pictures, used for cutting tiles for the fine matcher; features are the edge-direction grids used by the coarse search.)*
