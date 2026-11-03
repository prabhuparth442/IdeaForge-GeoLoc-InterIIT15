# The pinhole camera: from 3-D points to pixels

Code: `src/header/data.py` (`Camera`), `src/image_processing/geometry.py` (steps 3 and 4), `src/image_processing/map_levels.py` (`level_for_height`).

---

## 1. The model

A camera is modelled as a **pinhole**: every ray of light passes through one point, the *optical centre* `C`. We place a coordinate system on the camera:

* `Zc` points forward along the viewing direction (depth),
* `Xc` points to the right of the image,
* `Yc` points down in the image.

A scene point with camera coordinates `(Xc, Yc, Zc)` and `Zc > 0` (in front of the camera) is seen along the ray from `C` through that point. Where does the ray hit the sensor?

### Similar triangles

Put the sensor plane at distance `f` from `C` (`f` = *focal length expressed in pixels*; that unit trick removes any need to know the sensor's physical pixel size). The ray hits it at

$$x=\frac{X_c}{Z_c},\qquad y=\frac{Y_c}{Z_c}\qquad(\text{normalised coordinates, unit-free}),$$

$$u=f_x\,x+c_x,\qquad v=f_y\,y+c_y.$$

`(cx, cy)` is the **principal point**, the pixel where the optical axis pierces the sensor (almost the image centre), and `fx, fy` are focal lengths for the horizontal and vertical axes (nearly equal for square pixels).

In matrix form with the *intrinsic matrix* `K`:

$$\begin{pmatrix}u\\v\\1\end{pmatrix}\sim\underbrace{\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}}_{K}\begin{pmatrix}X_c\\Y_c\\Z_c\end{pmatrix}$$

("∼" = equal after dividing the result by its third entry, which is `Zc`. That division is the perspective effect: far things are small.)

For this project (official camera): `320 × 256` pixels, `fx ≈ fy ≈ 200`, `cx = 159.5`, `cy = 127.5`. The principal point is `(W−1)/2, (H−1)/2`, the centre of a 320 × 256 grid whose first pixel is index 0.

## 2. Ground sample distance and footprint

For a nadir (straight-down) camera at height `h`, a ground point `X` metres from the point below the camera has `Zc = h`, `Xc = X`, so `u − cx = f·X/h`:

$$\text{GSD}=\frac{h}{f}\ \text{m/pixel},\qquad \text{footprint}_{\text{width}}=W\cdot\frac{h}{f},\qquad\text{footprint}_{\text{height}}=H\cdot\frac{h}{f}.$$

Field of view across the width: the extreme pixel is `W/2` from the centre, so

$$\text{FOV}=2\arctan\frac{W/2}{f}=2\arctan\frac{160}{200}=77.3^\circ.$$

| h | GSD | footprint (320 × 256 px) |
|---|---|---|
| 100 m | 0.50 m | 160 m × 128 m |
| 200 m | 1.00 m | 320 m × 256 m |
| 350 m | 1.75 m | 560 m × 448 m |

A 77° field of view is a *wide* lens. That is why the lens distortion is strong (see `lens_distortion.md`) and why the corners of the photo hold a lot of ground.

## 3. Why the footprint chooses the zoom level

The coarse search needs the photo to fit inside one 224 × 224 pixel canvas at some map zoom level. `level_for_height` solves this directly; the derivation is in `map_zoom_levels.md`.

## 4. In the code

```python
class Camera:
    def __init__(self, table):
        ...
        self.fx, self.fy = table["cam_fx"][0], table["cam_fy"][0]
        self.cx, self.cy = table["cam_cx"][0], table["cam_cy"][0]
        self.k1, self.k2, self.k3 = ...
        self.p1, self.p2 = ...
        self.width = 320
        self.height = 256
```

The nine camera numbers are read once from the first CSV row (after the assertion that all rows agree). The values are used in two places:

1. `geometry.make_query`: `map_x = fx * x_real + cx` (the pixel coordinate the canvas pixel will read from the photo),
2. `map_levels.level_for_height`: `footprint = camera.width * height / camera.fx`.

**Constraints:** `width`/`height` are hard-coded to the official 320 × 256 camera. A different sensor needs those two lines changed (and `settings.TILE` re-thought), nothing else.
