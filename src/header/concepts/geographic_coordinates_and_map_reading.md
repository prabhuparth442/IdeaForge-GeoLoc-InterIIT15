# From GPS degrees to map pixels, and how the huge map is read

**Code:** `src/header/satellite_map.py`, `find_map_file` in `src/header/data.py`, `truth_in_metres` in `src/evaluation/testset.py`.

**What you need to know beforehand:** that GPS gives latitude and longitude in degrees, and that a picture is a grid of pixels. We explain everything in between.

---

## 1. The big picture in plain words

The photo's metadata says "I was at latitude 19.12, longitude 73.08". The map is a giant picture made of pixels. To compare them we must answer: **which pixel of the map is latitude 19.12, longitude 73.08?** And later, in the other direction: "the match is at pixel (row 4000, col 7000), what is that in GPS?"

The conversion has three hops:

```
 (lat, lon) in degrees ──project──► (E, N) in metres ──affine──► (row, col) in pixels ──×gsd──► (x, y) "map metres"
  GPS, a sphere              flat grid, metres        the image grid             the project's own frame
```

## 2. Hop 1: from degrees to metres (projection)

### Why degrees are awkward

Latitude lines are evenly spaced: one degree of latitude is about 111 km everywhere. Longitude lines converge towards the poles, so one degree of longitude shrinks with latitude:

$$1^{\circ}\ \text{of longitude}\approx 111.3\ \text{km}\times\cos(\text{latitude}).$$

At latitude 19.12° (our data) `cos(19.12°) = 0.945`, so 1° of longitude is about 105 km, while 1° of latitude is 111 km. A "circle of 5 km" is therefore not a circle in degrees. Pixels are squares and distances are metres, so we need a **flat grid in metres**.

### What a projection does

A map projection flattens a piece of the curved Earth onto a plane. We use the map's own projection, typically **UTM** (Universal Transverse Mercator): the Earth is cut into 60 north–south strips ("zones"), and inside a zone the flattening is accurate to about 0.1 %. The result is two numbers in metres: *easting* `E` (how far east) and *northing* `N` (how far north).

The library `pyproj` does the work:

```python
self.to_map_crs = Transformer.from_crs("EPSG:4326", self.file.crs, always_xy=True)
easting, northing = self.to_map_crs.transform(lon, lat)      # longitude FIRST
```

* `"EPSG:4326"` is the code for plain GPS latitude/longitude.
* `self.file.crs` is the map file's own projection, read from the GeoTIFF.
* **`always_xy=True` is essential.** Officially, EPSG:4326 lists latitude first. With the default setting pyproj would expect `(lat, lon)`, and if you pass `(lon, lat)` the result is a perfectly valid-looking position thousands of kilometres away. The flag makes the order always `(x, y) = (lon, lat)`. This one flag is a classic source of "everything is shifted" bugs.

## 3. Hop 2: from metres to a pixel (the affine transform)

A GeoTIFF stores a small recipe, the **affine transform**, that links pixel indices to map metres:

$$\begin{pmatrix}E\\N\end{pmatrix}=\begin{pmatrix}a&b\\d&e\end{pmatrix}\begin{pmatrix}\text{col}\\\text{row}\end{pmatrix}+\begin{pmatrix}c\\f\end{pmatrix}.$$

For a normal north-up map with square pixels of size `g` (here `g = 0.3` m) the recipe is simple: `a = g`, `e = −g`, `b = d = 0`, and `(c, f)` are the coordinates of the top-left corner. Written out:

$$E=c+g\cdot\text{col},\qquad N=f-g\cdot\text{row}.$$

Why the **minus** in front of `row`? Image rows are counted downwards from the top, but northing grows upwards. So each step down a row reduces `N` by `g`.

Solving for the pixel (this is what `~transform`, the inverse, gives):

$$\text{col}=\frac{E-c}{g},\qquad \text{row}=\frac{f-N}{g}.$$

**Worked example.** Suppose the top-left corner is `(c, f) = (500000, 2000000)` and `g = 0.3`. A point at `E = 500150`, `N = 1999940`:

* col = (500150 − 500000) / 0.3 = 500
* row = (2000000 − 1999940) / 0.3 = 200

So that point is pixel (row 200, col 500). In code:

```python
col, row = self.pixel_from_coords * (easting, northing)
return row, col
```

The `*` here applies the stored inverse matrix to the pair. Note the result order flip: the transform works with `(col, row)` = `(x, y)`, while we return `(row, col)` as numpy users expect. The values are **floats** because a GPS position falls between pixels, and that is fine.

**Assumptions:** the map has no rotation or shear (`b = d = 0`) and its unit is metres. A map stored directly in degrees would break every statement in this project that says "metres".

## 4. Hop 3: the project's own frame, "map metres"

From now on, positions are measured **in metres from the top-left corner of the map**, with **`x` pointing east and `y` pointing south**:

$$x=\text{col}\cdot g,\qquad y=\text{row}\cdot g.$$

We chose *south* for `y` deliberately: it matches how image rows grow downward, so moving down in a picture is a `+y` step and no sign flips are needed when cutting tiles out of the map.

```python
row, col = satellite_map.gps_to_map_pixel(item["lat"], item["lon"])
return np.array([col * satellite_map.gsd, row * satellite_map.gsd])      # (x, y)
```

Continuing the example: `(x, y) = (500 × 0.3, 200 × 0.3) = (150 m, 60 m)`.

To go back (when writing a real answer file) you invert each hop: `row = y/g`, `col = x/g`, then the affine transform to `(E, N)`, then pyproj back to latitude/longitude. The development-set evaluation compares metre positions directly, so this reverse path is not coded yet (see the list of open items in `src/README.md`).

## 5. Reading a 2.6 GB map without loading it

The development map has about 306 million pixels per band and 3 bands. As float32 numbers one band alone would take 1.2 GB, and we would need several copies. So the code never holds the whole map; it reads **pieces**:

* **A strip** (`read_strip(top, rows)`) reads full-width rows `top … top+rows`. The pyramid builder (`image_processing/map_levels.py`) reads 2048 rows at a time, shrinks them to every zoom level and throws the strip away.
* **An overview** (`read_grey(out_shape=(H//16, W//16))`) asks rasterio for a picture 16 times smaller in each direction, averaging each 16 × 16 block. It is cheap and perfectly good for estimating a brightness statistic.
* **A window** (`Window(col_off, row_off, width, height)`) reads any rectangle; with `boundless=True` the parts that hang over the map border are filled with 0 instead of raising an error.

## 6. Bands, and the brightness limit

The satellite picture has red, green and blue bands. The thermal camera is single-channel, so we average the bands into one grey picture (`mean(axis=0)` over `MAP_BANDS = (1, 2, 3)`). The final organiser map is a single panchromatic band: then set `MAP_BANDS = (1,)`.

To store the map in 8 bits we must decide what "full white" is. Taking the largest pixel would be a mistake: a single sun-glint on a metal roof would make all the rest look dark. We take the **99.5th percentile** of a 1/16-size overview, called `gray_max`, and scale:

$$v=\operatorname{clip}\!\Big(\frac{g}{g_{\max}},\,0,\,1\Big)\cdot 254+1\quad\text{for }g>0,\qquad v=0\ \text{for }g\le 0.$$

Real pixels end up in 1 … 255, and **0 is kept for "no map here"** (outside the satellite swath). Why reserve 0? Later, the matching compares edge patterns; an empty area must never look like "very dark ground", or the search would happily match a photo against nothing. The guards in `coarse_search/concepts/template_matching_cosine_and_zncc.md` rely on this convention.

The percentile rule also works unchanged for 11-bit data stored in a 16-bit file (the final Cartosat map), which is why it is used instead of a fixed "divide by 255".

## 7. Code map

| function | input | output |
|---|---|---|
| `SatelliteMap(path, bands)` | path to a `.tif` | object holding `width`, `height`, `gsd`, `gray_max` |
| `read_grey(window, out_shape, boundless)` | a Window, or a target shape | float32 2-D array (mean of bands, 0 = no data) |
| `read_strip(top, rows)` | first row, number of rows | the full-width strip as grey |
| `grey_to_uint8(grey)` | float grey array | uint8 array with the 0 / 1…255 convention |
| `gps_to_map_pixel(lat, lon)` | degrees (lat first, lon second) | `(row, col)` as floats |

`find_map_file()` returns the first `.tif` found in `dataset/small_sat_area/`.

**Input checks (asserts):** latitude must be in −90…90 and longitude in −180…180 (this catches swapped arguments for most places on Earth), strips must lie inside the map, `grey` must be 2-D.

## 8. Mistakes to avoid

* Passing `(lat, lon)` to the transformer instead of `(lon, lat)`.
* Mixing up `(row, col)` with `(x, y)`: `x` is the column, `y` is the row.
* Assuming north is up for `y`: in this project `y` points **south**.
* Forgetting that 0 means "no map" when doing arithmetic on the uint8 map.

## 9. Check yourself

1. Row 1000, column 2000, `g = 0.3`: what are `x` and `y`? *(600 m, 300 m)*
2. Why is `e = −g` in the affine transform? *(Rows grow downward, northing grows upward.)*
3. Why do we use the 99.5th percentile, not the maximum? *(A few bright glints would darken everything else.)*
4. At latitude 19.12°, roughly how many kilometres is one degree of longitude? *(About 105.)*
