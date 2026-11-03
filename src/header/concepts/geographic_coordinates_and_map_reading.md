# From GPS degrees to map pixels, and how the big map is read

Code: `src/header/satellite_map.py`, `src/header/data.py` (`find_map_file`), parts of `evaluation/testset.py`.

---

## 1. Why we cannot just use latitude and longitude

GPS gives angles on a sphere-like Earth: latitude `φ` (north/south) and longitude `λ` (east/west). One degree of latitude is always about 111 km, but one degree of longitude shrinks with latitude:

$$1^\circ\text{ lon}\approx 111\,\text{km}\cdot\cos\varphi\qquad(\varphi=20^\circ\Rightarrow 104\,\text{km}).$$

Distances, circles ("within 5 km") and image pixels need a *flat grid in metres*. So we **project** the sphere onto a plane. Satellite maps ship in such a projection, for example UTM (Universal Transverse Mercator): the Earth is cut into 60 north–south strips (zones); inside a zone the projection to *easting* `E` and *northing* `N` (both metres) has distortion below 0.1 %.

`pyproj.Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)` performs that projection. `EPSG:4326` is plain GPS lat/lon. **`always_xy=True` matters**: it fixes the argument order to `(lon, lat)` = `(x, y)`. Without it pyproj follows the official axis order of EPSG:4326, which is `(lat, lon)`, and every position silently lands in the wrong place. In the code:

```python
easting, northing = self.to_map_crs.transform(lon, lat)   # lon first!
```

## 2. From metres to a pixel: the affine transform

A GeoTIFF stores an *affine transform* from pixel indices `(col, row)` to map metres `(E, N)`:

$$\begin{pmatrix}E\\N\end{pmatrix}=\begin{pmatrix}a&b\\d&e\end{pmatrix}\begin{pmatrix}\text{col}\\\text{row}\end{pmatrix}+\begin{pmatrix}c\\f\end{pmatrix}.$$

For a north-up map with square pixels of size `g` (here `g = 0.3` m): `a = g`, `e = −g` (rows count *downwards* while northing counts *upwards*), `b = d = 0`, and `(c, f)` = coordinates of the top-left corner. So

$$E=c+g\cdot\text{col},\qquad N=f-g\cdot\text{row}.$$

The inverse (`~transform` in rasterio, the matrix inverse) gives the pixel from the metres:

$$\text{col}=\frac{E-c}{g},\qquad \text{row}=\frac{f-N}{g}.$$

```python
col, row = self.pixel_from_coords * (easting, northing)
return row, col
```

`gps_to_map_pixel` returns `(row, col)` as floats (a position *between* pixels is fine).

**Constraint:** this assumes the map has no rotation/shear (`b = d = 0`) and its CRS unit is metres. A map in degrees (EPSG:4326) would break every "metres" statement in this project.

## 3. The project's own coordinate system: "map metres"

Everywhere after loading, positions are measured in **metres from the top-left corner of the map, `x` to the east, `y` to the south**:

$$x=\text{col}\cdot g,\qquad y=\text{row}\cdot g.$$

South (not north) is chosen because it matches how image rows grow downward, so a pixel step down is a `+y` step and no sign flips appear when cutting tiles. See `truth_in_metres` in `evaluation/testset.py`:

```python
row, col = satellite_map.gps_to_map_pixel(item["lat"], item["lon"])
return np.array([col * satellite_map.gsd, row * satellite_map.gsd])
```

At the very end (when writing a real prediction file) the inverse path is `row = y/g`, `col = x/g`, then the transform to `(E, N)`, then pyproj back to lat/lon. That last step is not needed for the development-set evaluation; the evaluation compares metre positions.

## 4. Reading a 2.6 GB map without loading it

The development map is 22096 x 13840 = 306 million pixels per band, 3 bands: 0.3 GB per band as uint8 but 1.2 GB per band as float32 (the type we compute in), and the file on disk is 2.6 GB. We never hold it whole:

* **Windows** (`Window(col_off, row_off, width, height)`) read one rectangle from disk. `read_strip(top, rows)` reads full-width strips of 2048 rows; each strip is shrunk to every zoom level, then discarded (`image_processing/map_levels.py`).
* **Overview read**: `read_grey(out_shape=(H//16, W//16))` asks rasterio to deliver a 16× smaller picture directly with `Resampling.average` (each output pixel is the mean of a 16×16 block). Cheap and good enough to estimate a brightness statistic.
* **`boundless=True, fill_value=0`**: rectangles that hang over the map border are filled with 0 instead of raising an error. Not used in the pipeline's main path today, but available in `read_grey`.

## 5. Bands to one grey picture, and the brightness limit

Satellite RGB bands are averaged into one grey grid (`mean(axis=0)` over `MAP_BANDS=(1,2,3)`). The thermal camera is single-channel, so we compare grey with grey. (The final organiser map is a single panchromatic band: set `MAP_BANDS = (1,)`.)

To store the map in 8 bits we need a "full brightness" value. The maximum pixel is a bad choice because a few glints (a metal roof reflecting the sun) would make everything else look dark. We use the **99.5th percentile** of the overview, `gray_max`, then map

$$v=\operatorname{clip}\!\Big(\frac{g}{g_{\max}},0,1\Big)\cdot254+1\quad(g>0),\qquad v=0\quad(g\le0).$$

Real pixels land in 1…255; **0 is reserved for "no map here"** (outside the satellite swath). Reserving 0 matters later: an empty region must never look like "very dark ground", otherwise correlation would happily match the photo against nothing (see `coarse_search/concepts/template_matching_cosine_and_zncc.md`, "guards").

For 11-bit data in a 16-bit file (the final Cartosat map) the percentile rule works unchanged, which is why it was chosen instead of a fixed `/255`.

## 6. Code map

| function | input | output |
|---|---|---|
| `SatelliteMap(path, bands)` | path to a `.tif` | object holding width, height, `gsd`, `gray_max` |
| `read_grey(window, out_shape, boundless)` | a rasterio `Window` or a target shape | float32 2-D array, mean of the bands, `0` = no data |
| `read_strip(top, rows)` | first row, number of rows | the full-width strip as grey |
| `grey_to_uint8(grey)` | float grey array | uint8 array with the 0/1..255 convention |
| `gps_to_map_pixel(lat, lon)` | degrees | `(row, col)` floats |

`find_map_file()` in `data.py` picks the first `*.tif` inside `dataset/small_sat_area/`.
