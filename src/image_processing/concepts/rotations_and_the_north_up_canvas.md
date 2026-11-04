# Building the north-up canvas: yaw, pitch, roll and the inverse warp

**Code:** `camera_axes`, `make_query` and `render_query` in `src/image_processing/geometry.py`.

**What you need to know beforehand:** the pinhole model (`pinhole_camera_and_ground_sample_distance.md`), the lens distortion function (`lens_distortion.md`), and that a vector can be projected onto an axis with a dot product. Everything else is derived here.

This is the most important geometry in the project. Read it slowly once. After that, the code is just these formulas typed in.

---

## 1. The big picture in plain words

The satellite map is **north-up and looks straight down**. The drone photo is different in four ways: the top of the photo is wherever the drone's nose pointed, the camera is tilted a few degrees, the lens bends the picture, and the pixel size depends on height. To compare the two we build a new picture of the ground, the **canvas**, that behaves exactly like a small piece of the map:

* north is up, east is right (as in the map),
* each pixel is `g` metres wide (`g` is the pixel size of the chosen zoom level),
* **the middle of the canvas is the point directly below the drone** (the nadir point). The competition asks for the *drone's* position, which is the nadir point. When the camera is tilted, the photo's centre looks at a different ground point (see the example in Section 5); the canvas centre does not.

### Why we pull instead of push

The obvious idea is to take each photo pixel and put it on the canvas ("push", forward mapping). That leaves holes where nothing lands and collisions where two pixels land together. We do the opposite: for **every canvas pixel** we compute *where in the photo that spot of ground was seen*, then read the photo there ("pull", an **inverse warp**). Every canvas pixel gets exactly one value, and we never invert any hard formula, because pulling uses the forward camera model only.

The recipe has five short steps:

```
canvas pixel → (1) ground point → (2) camera axes → (3) camera coordinates → (4) pinhole + lens → photo pixel → (5) read value
```

## 2. Step 1: canvas pixel to a point on the ground

The canvas has `S × S` pixels. Its middle is `m = (S − 1)/2` (for `S = 224`: `m = 111.5`). A canvas pixel in column `c` and row `r` lies

$$E=(c-m)\,g\ \text{metres east},\qquad N=-(r-m)\,g\ \text{metres north}$$

of the nadir point. The minus sign: rows grow downwards, north grows upwards.

Put a local world frame at the camera, with axes east, north and up. The camera is at height `h` above the nadir point, so the ground point seen at this canvas pixel is

$$\mathbf P=(E,\ N,\ -h).$$

```python
east  = (cols - middle) * pixel_size
north = -(rows - middle) * pixel_size
below = -height
```

`cols` and `rows` come from `np.meshgrid`, so these three lines compute the ground point for all 50 176 canvas pixels at once.

## 3. Step 2: where do the camera's axes point?

Next we need three unit vectors, written in east–north–up coordinates:

* `R`: the direction of image-right,
* `B`: the direction of image-down,
* `D`: the viewing direction, out of the lens.

### Level flight with heading ψ

The heading ψ (yaw) is the compass direction of the nose, measured clockwise from north. The nose vector is

$$\mathbf F=(\sin\psi,\ \cos\psi,\ 0).$$

Check: ψ = 0 gives `(0, 1, 0)`, north. ψ = 90° gives `(1, 0, 0)`, east. The right wing points 90° clockwise from the nose:

$$\mathbf R=(\cos\psi,\ -\sin\psi,\ 0)$$

Check: ψ = 0 gives east, ψ = 90° gives south. The camera looks straight down, `D = (0, 0, −1)`. A downward camera shows the nose direction at the **top** of the image, so image-down is the opposite of the nose: `B = −F`.

(Sanity check: `R·F = cosψ sinψ − sinψ cosψ = 0`. The axes are perpendicular, as they should be.)

### Pitch θ (nose up is positive): rotate about the right axis

The nose rises, gaining an upward component; the viewing direction swings the same way:

$$\mathbf F'=\mathbf F\cos\theta-\mathbf D\sin\theta,\qquad\mathbf D'=\mathbf D\cos\theta+\mathbf F\sin\theta.$$

(Since `−D` points up, `F'` gets an upward part `sinθ`. The right axis `R` does not change.) They stay perpendicular: `F'·D' = cosθ sinθ − sinθ cosθ = 0`.

### Roll φ (right wing down is positive): rotate about the forward axis

$$\mathbf R''=\mathbf R\cos\varphi+\mathbf D'\sin\varphi,\qquad\mathbf D''=\mathbf D'\cos\varphi-\mathbf R\sin\varphi.$$

The right wing tips towards "down" (`+D'`), and the viewing direction tilts the opposite way.

**Order matters.** Three-dimensional rotations do not commute: pitch-then-roll is not the same as roll-then-pitch. The order here is a convention that must match how the autopilot defines its angles; for the small tilts in this data the difference is tiny.

The code (the variable names carry the step numbers: `1` after pitch, `2` after roll):

```python
forward1 = forward * cos(pitch) - down * sin(pitch)
down1    = down * cos(pitch) + forward * sin(pitch)
right2   = right * cos(roll) + down1 * sin(roll)
down2    = down1 * cos(roll) - right * sin(roll)
return right2, -forward1, down2        # image-right, image-down, viewing axis
```

## 4. Step 3: the ground point in camera coordinates

Project `P` onto each camera axis with a dot product:

$$X_c=\mathbf R''\!\cdot\mathbf P,\qquad Y_c=\mathbf B\cdot\mathbf P,\qquad Z_c=\mathbf D''\!\cdot\mathbf P.$$

For level flight `D'' = (0, 0, −1)`, so `Zc = h`, positive: in front of the camera. A canvas pixel is only usable when `Zc > 0` (otherwise the ground point would be behind the camera, which can happen at the canvas edges for extreme tilt):

```python
in_front = z_cam > 1e-3
```

Then the pinhole model: `x = Xc/Zc`, `y = Yc/Zc`; the lens: `(x_d, y_d) = distort(x, y)`; the pixel: `u = fx·x_d + cx`, `v = fy·y_d + cy`. These `(u, v)` are the photo coordinates to read for this canvas pixel (`map_x`, `map_y` in the code).

## 5. Worked examples (computed with the real code)

All with the camera `fx = 200`, `cx = 159.5`, `cy = 127.5`, no distortion, height `h = 100 m`.

**(a) Level flight, heading north.** Canvas pixel 10 m east and 20 m north of nadir: `P = (10, 20, −100)`. Then `Xc = 10`, `Yc = −20` (north is up in the image, so negative "down"), `Zc = 100`. `x = 0.1`, `y = −0.2`, so `u = 159.5 + 20 = 179.5`, `v = 127.5 − 40 = 87.5`. North of the nadir point is *above* the photo centre, as expected.

**(b) Same ground point, but the drone flies east (ψ = 90°).** Now the photo's top is east. The ground point 20 m north of nadir appears at `(u, v) = (119.5, 127.5)`: to the **left** of centre. And the point 20 m east appears at `(159.5, 87.5)`: at the **top**. The photo is simply rotated; the canvas undoes that rotation.

**(c) Why the nadir point matters: a 5° pitch.** The nadir point is `P = (0, 0, −100)`. With 5° of pitch, it is seen at `(u, v) = (159.5, 145.0)`: 17.5 pixels **below** the photo centre. The photo centre looks at the ground `h·tan 5° = 8.7 m` ahead of the drone. A 5° roll shifts it the same 17.5 pixels sideways, to `(177.0, 127.5)`. If we had centred the canvas on the photo centre instead of the nadir point, our answer would be 8.7 m wrong at 100 m (and 30.6 m wrong at 350 m), already larger than the allowed error σ (see `evaluation/concepts/sensitivity_to_errors.md`).

## 6. Step 4: one resampling

```python
canvas = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
inside = cv2.remap(np.ones_like(photo), map_x, map_y, ...) > 0.99
valid  = inside & in_front
canvas = np.where(valid, canvas, photo.mean())
```

* `cv2.remap(src, map_x, map_y)` produces `dst[i, j] = src(map_y[i, j], map_x[i, j])`, interpolating between pixels (details in `resampling_and_aliasing.md`).
* Undistortion, tilt correction, scaling and rotation all happen in this **single** lookup, so the photo is interpolated, and therefore slightly blurred, only once.
* **The mask.** Remapping a picture of all ones gives 1 where the source was fully inside the photo, something between 0 and 1 where the interpolation touched the outside, and 0 where it was fully outside. Requiring `> 0.99` keeps only canvas pixels whose neighbours were all real photo pixels. The empty corners (the missing triangles of the rotated rectangle) are filled with the photo's average, so they carry no edges.

## 7. The settings dictionary: six knobs found from data

The metadata conventions are not documented (is positive pitch nose-up? does yaw 0 mean north?), so `settings.GEOMETRY` holds six knobs that were **measured** on the data (see `evaluation/concepts/geometry_hypothesis_tests.md`):

| key | meaning | value found on the official set |
|---|---|---|
| `attitude` | use pitch and roll (True) or pretend the camera looks straight down | True |
| `pitch_sign`, `roll_sign` | ±1: the sign convention of the stored angle | −1, −1 |
| `yaw_offset` | degrees added to the stored heading (a constant convention or mounting error) | 180° |
| `scale` | multiplies the height (absorbs a height or focal-length bias) | 1.04 |
| `height_offset` | metres added to the height (for example −4 if `alt` is above sea level) | 0 |

```python
height = (item["rel_height"] + setting["height_offset"]) * setting["scale"]
yaw    = item["yaw"] + setting["yaw_offset"]
pitch  = setting["pitch_sign"] * item["pitch"]      # only if attitude is True
```

**These values belong to this dataset.** Another dataset (the SCC-Loc public set) needed scale 0.80 and no heading offset. Re-run `python run.py --sweep` on any new data.

## 8. Function reference

| function | input | output |
|---|---|---|
| `camera_axes(yaw_deg, pitch_deg, roll_deg)` | three angles in degrees | three numpy 3-vectors: image-right, image-down, viewing axis |
| `make_query(item, camera, canvas_size, pixel_size, geometry)` | one photo record, the camera, canvas size in pixels, metres per canvas pixel, optional overrides | `canvas` float32 `(S, S)` in `[0, 1]`, `valid` bool `(S, S)` |
| `render_query(item, camera, level, geometry)` | the same, with the size and pixel size taken from the zoom level | the same pair, `S = 224` |

`geometry` is a partial dictionary of overrides merged onto `settings.GEOMETRY` (used by the sweep and the sensitivity tests). Constraints: the photo file must exist; pitch and roll are already wrapped to ±180 by `make_items`; angles are in degrees; with tilt beyond about 60° many canvas pixels have `Zc ≤ 0` and the valid area collapses.

## 9. Mistakes to avoid

* Centring the canvas on the photo centre instead of the nadir point.
* Using radians where the code expects degrees (the conversion happens inside `camera_axes`).
* Forgetting that rows grow downwards (the minus sign in `north`).
* Copying the `GEOMETRY` values to a new dataset without testing them.

## 10. Check yourself

1. In example (b), why does the photo's top show east? *(The drone's nose points east, and the photo's top is the nose direction.)*
2. What is `Zc` for level flight at 100 m? *(100: the depth along the viewing direction.)*
3. Why does the mask test use `> 0.99` and not `== 1`? *(Floating-point interpolation can produce values like 0.999.)*
4. Why does one `remap` beat "undistort, then rotate, then scale"? *(Three resamplings blur three times; one remap blurs once, and no holes appear.)*
