# Building the north-up canvas: yaw, pitch, roll and the inverse warp

Code: `src/image_processing/geometry.py` (`camera_axes`, `make_query`, `render_query`).

This is the most important geometry in the project. Read it slowly once; after that the code is just the formulas typed in.

---

## 1. What we want

The satellite map is **north-up and looks straight down**. The drone photo has the drone's heading at the top, is tilted a few degrees, and is bent by the lens. To compare them we build a new picture, the **canvas**:

* north is up, east is right (same as the map),
* each pixel is `g` metres (the zoom level's pixel size, `g = level_gsd(L)`),
* **the canvas middle is the point directly below the drone** (nadir). The competition wants the *drone's* position, which is that point, not the photo's centre; when the camera is tilted the photo centre points at ground somewhere else (offset ≈ `h·tan(tilt)`, e.g. 8.7 m for 5° at 100 m).

We do not move photo pixels onto the canvas ("forward" mapping leaves holes where nothing lands and collisions where two pixels land together). Instead, for **every canvas pixel** we compute *where in the photo that spot of ground was seen*, then read the photo there. This is an **inverse warp** (pull, not push). Every canvas pixel gets exactly one value.

## 2. Step 1: canvas pixel to a point on the ground

Canvas of `S × S` pixels. The middle is `m = (S−1)/2` (for `S = 224`, `m = 111.5`). A canvas pixel in column `c` and row `r` lies

$$E=(c-m)\,g\quad\text{metres east},\qquad N=-(r-m)\,g\quad\text{metres north}$$

of the nadir point. (The minus: rows grow downwards, north grows upwards.) Put the origin of a local world frame **at the camera**: east–north–up axes. The camera is at height `h` above the nadir point, so the ground point seen at this canvas pixel is

$$\mathbf P=(E,\;N,\;-h).$$

```python
east  = (cols - middle) * pixel_size
north = -(rows - middle) * pixel_size
below = -height
```

## 3. Step 2: where the camera axes point

We need three unit vectors written in east–north–up coordinates:

* `R`: image-right,
* `B`: image-down,
* `D`: viewing direction (out of the lens).

### Level flight, heading ψ

The heading `ψ` (yaw) is the compass direction of the nose, measured clockwise from north. The nose vector in (east, north, up) is

$$\mathbf F=(\sin\psi,\ \cos\psi,\ 0)$$

(check: `ψ = 0` → north `(0,1,0)`; `ψ = 90°` → east `(1,0,0)`). The right wing points 90° clockwise from the nose:

$$\mathbf R=(\cos\psi,\ -\sin\psi,\ 0)$$

(check: `ψ = 0` → east; `ψ = 90°` → south). The camera looks straight down: `D = (0, 0, −1)`.

A nadir camera shows the nose direction at the **top** of the image, so image-down `B = −F`. Orthonormal check: `R·F = cosψ sinψ − sinψ cosψ = 0`. ✓

### Pitch θ (nose up positive): rotate about the right axis

The nose rises: it gains an upward component, and the viewing direction swings the same way:

$$\mathbf F'=\mathbf F\cos\theta-\mathbf D\sin\theta,\qquad \mathbf D'=\mathbf D\cos\theta+\mathbf F\sin\theta.$$

(`−D` is "up", so `F'` has an up component `sinθ`.) `F'·D' = (Fcosθ − Dsinθ)·(Dcosθ + Fsinθ) = cosθ sinθ − sinθ cosθ = 0`, so they stay perpendicular. The right axis does not change.

### Roll φ (right wing down positive): rotate about the new forward axis

$$\mathbf R''=\mathbf R\cos\varphi+\mathbf D'\sin\varphi,\qquad\mathbf D''=\mathbf D'\cos\varphi-\mathbf R\sin\varphi.$$

The right wing moves towards "down" (`+D'`), and the viewing direction tilts the opposite way.

Pitch first, roll second, is a convention: 3-D rotations do not commute, so the order must match how the autopilot defines the angles. The code:

```python
forward1 = forward * cos(pitch) - down * sin(pitch)
down1    = down * cos(pitch) + forward * sin(pitch)
right2   = right * cos(roll) + down1 * sin(roll)
down2    = down1 * cos(roll) - right * sin(roll)
return right2, -forward1, down2        # image-right, image-down, viewing axis
```

The three returned vectors are the rows of a rotation matrix from the world frame to the camera frame.

## 4. Step 3: the point in camera coordinates

Dot products project the vector `P` onto each camera axis:

$$X_c=\mathbf R''\!\cdot\mathbf P,\qquad Y_c=\mathbf B\cdot\mathbf P,\qquad Z_c=\mathbf D''\!\cdot\mathbf P.$$

Level flight check: `D'' = (0,0,−1)` so `Zc = h` (positive, in front). A canvas pixel is only valid if `Zc > 0` (otherwise the ground point would be behind the camera, which can happen for extreme tilt at the canvas edges). `in_front = z_cam > 1e-3`.

Then `x = Xc/Zc`, `y = Yc/Zc` (pinhole), pass through `distort` (`lens_distortion.md`), and `u = fx x_d + cx`, `v = fy y_d + cy`. Those `(u, v)` are the photo pixel to read for this canvas pixel: `map_x`, `map_y`.

### Worked example

Level flight, heading 0 (north), height 100 m. Canvas pixel exactly 10 m east and 20 m north of the nadir point: `P = (10, 20, −100)`. `R'' = (1,0,0)` → `Xc = 10`. `B = −F = (0,−1,0)` → `Yc = −20`. `D'' = (0,0,−1)` → `Zc = 100`. So `x = 0.1`, `y = −0.2` (ignoring distortion) → `u = 159.5 + 20`, `v = 127.5 − 40`. North of the nadir is *above* the photo centre (smaller `v`), as expected for a north-facing camera. ✓

## 5. Step 4: one resampling

```python
canvas = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
inside = cv2.remap(np.ones_like(photo), map_x, map_y, ...) > 0.99
valid  = inside & in_front
canvas = np.where(valid, canvas, photo.mean())
```

* `cv2.remap(src, map_x, map_y)` produces `dst[i,j] = src(map_y[i,j], map_x[i,j])` with interpolation (details in `resampling_and_aliasing.md`).
* Undistortion, tilt correction, scaling and rotation happen in this **single** lookup, so the photo is interpolated (and so slightly blurred) only once.
* **Mask**: remapping an all-ones image gives 1 where the source was fully inside the photo, between 0 and 1 where interpolation touched the outside, 0 where fully outside. Requiring `> 0.99` keeps only pixels whose neighbours were all real. Empty corners (the rotated rectangle's missing triangles) are filled with the photo mean, so they carry no edges.

## 6. The settings dictionary

`settings.GEOMETRY` has six knobs. They exist because the metadata conventions are not documented, so they are *measured* (see `evaluation/concepts/geometry_hypothesis_tests.md`):

| key | meaning | value found on the official set |
|---|---|---|
| `attitude` | use pitch and roll (True) or pretend straight down | True |
| `pitch_sign`, `roll_sign` | ±1: the stored angle's sign convention | −1, −1 |
| `yaw_offset` | degrees added to the stored heading (constant mounting/convention error) | 180° |
| `scale` | multiplies the height (corrects a height or focal length bias) | 1.04 |
| `height_offset` | metres added to height (e.g. −4 if `alt` is above sea level) | 0 |

```python
height = (item["rel_height"] + setting["height_offset"]) * setting["scale"]
yaw    = item["yaw"] + setting["yaw_offset"]
pitch  = setting["pitch_sign"] * item["pitch"]   # only if attitude is True
```

**These values belong to this dataset.** Another dataset (the SCC-Loc public set) needed scale 0.80 and no heading offset. Re-run the sweep on any new data.

## 7. Function reference

`camera_axes(yaw_deg, pitch_deg, roll_deg)` → three 3-vectors (numpy). Input degrees.

`make_query(item, camera, canvas_size=224, pixel_size=1.0, geometry=None)` → `(canvas, valid)`; `canvas` float32 `(S,S)` in `[0,1]`; `valid` bool `(S,S)`.

`render_query(item, camera, level, geometry=None)` → same, with `canvas_size=TILE` and `pixel_size=level_gsd(level)`.

`geometry` is a partial dictionary of overrides merged on top of `settings.GEOMETRY` (used by the sweep and sensitivity tests).

**Constraints:** the photo file must exist at `settings.IMAGES_DIR / item["name"]`; pitch/roll should be wrapped to ±180 (done in `make_items`); heading in degrees. Tilt beyond ~60° makes `Zc ≤ 0` for many canvas pixels and the valid area collapses.
