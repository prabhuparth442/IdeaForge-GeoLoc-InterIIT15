# The pinhole camera: from 3-D points to pixels

**Code:** `Camera` in `src/header/data.py`, steps 3 and 4 of `make_query` in `src/image_processing/geometry.py`, and `level_for_height` in `src/image_processing/map_levels.py`.

**What you need to know beforehand:** similar triangles. (If you read `header/concepts/scoring_rule.md` you have already seen the key idea, GSD = h/f. This note builds the full picture around it.)

---

## 1. The big picture in plain words

A camera turns a 3-D world into a flat grid of pixels. To place a thermal photo on a map we must be able to answer two questions in both directions:

* *Forward:* "this point on the ground, which pixel of my photo shows it?"
* *Backward:* "this pixel of my photo, which point on the ground is it?"

The **pinhole camera model** answers both with one tiny formula. Everything later in the pipeline (the north-up canvas, the zoom level, the sigma of the score) leans on it.

## 2. The model, step by step

Imagine a closed box with a tiny hole in one side. Light from each point of the scene travels in a straight line through the hole and lands on the opposite wall, which is the sensor. That is a pinhole camera.

Place a coordinate system on the hole (the *optical centre* `C`):

* `Zc` points forward, the direction the camera looks (the *depth* of a point),
* `Xc` points to the right in the image,
* `Yc` points down in the image.

A scene point at `(Xc, Yc, Zc)` with `Zc > 0` (in front of the camera) sends a ray through `C`. Put the sensor at distance `f` behind the hole (in the equations we place it in front, which gives the same numbers without the picture being upside down). Similar triangles give where the ray lands:

```
        sensor plane           hole (C)            scene point
            |                    *------------------> Zc (depth)
      x*f   |<---- f ---->       |\
            |                    | \  ray
                                 |  \
                              Xc |   * (Xc, Zc)
```

$$x=\frac{X_c}{Z_c},\qquad y=\frac{Y_c}{Z_c}\qquad(\text{"normalised coordinates"}, \text{ no units}).$$

Then scale by the focal length and shift to the pixel grid:

$$u=f_x\,x+c_x,\qquad v=f_y\,y+c_y.$$

* `f` is the focal length **in pixels**. Quoting it in pixels is a neat trick: we never need to know the physical size of the sensor's pixels.
* `(cx, cy)` is the **principal point**, the pixel where the optical axis hits the sensor, normally almost the image centre.
* `fx, fy` are separate for the horizontal and vertical directions, nearly equal for square pixels.

The division by `Zc` is the whole "perspective" effect: **twice as far means half as big**.

In matrix form, with the *intrinsic matrix* `K`:

$$\begin{pmatrix}u\\v\\1\end{pmatrix}\sim\underbrace{\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}}_{K}\begin{pmatrix}X_c\\Y_c\\Z_c\end{pmatrix}.$$

The sign "∼" means "equal after dividing the whole result by its third entry (`Zc`)".

**The official camera:** 320 × 256 pixels, `fx ≈ fy ≈ 200`, `cx ≈ 158.9`, `cy ≈ 131.7` (close to the exact centre `(159.5, 127.5)` of a 320 × 256 grid whose first pixel has index 0).

## 3. Worked example: where does a ground point land?

The drone is at height `h = 100 m`, camera pointing straight down. A ground point is 20 m to the east and 10 m to the south of the spot right below the drone. In camera coordinates: `Xc = 20` (right), `Yc = 10` (down in the image), `Zc = 100` (depth).

* `x = 20/100 = 0.2`, `y = 10/100 = 0.1`.
* `u = 200·0.2 + 159.5 = 199.5`, `v = 200·0.1 + 127.5 = 147.5`.

So the point appears at pixel (199.5, 147.5), 40 pixels right and 20 pixels below the centre. 40 pixels for 20 m means **0.5 m per pixel**: the GSD.

## 4. Ground sample distance and the footprint

For a straight-down camera `Zc = h`, so `u − cx = f·Xc/h`. One pixel is therefore

$$\text{GSD}=\frac{h}{f}\ \text{m/pixel}.$$

The whole photo covers a **footprint** on the ground:

$$\text{width}=W\cdot\frac{h}{f},\qquad\text{height}=H\cdot\frac{h}{f}.$$

The **field of view** across the width follows from the extreme pixel being `W/2` from the centre:

$$\text{FOV}=2\arctan\frac{W/2}{f}=2\arctan\frac{160}{200}=77.3^{\circ}.$$

| height h | GSD | footprint (320 × 256 pixels) |
|---|---|---|
| 100 m | 0.50 m | 160 m × 128 m |
| 200 m | 1.00 m | 320 m × 256 m |
| 350 m | 1.75 m | 560 m × 448 m |

A 77° field of view is a **wide** lens. That has two consequences: the corners of the photo hold a lot of ground, and the lens bends the picture a lot (`lens_distortion.md`).

## 5. Why the footprint chooses the map zoom level

We later compare the photo with a map picture 224 × 224 pixels in size. The photo must fit inside that square, so a taller flight (bigger footprint) needs a coarser map level. `level_for_height` computes this directly from `footprint = camera.width * height / camera.fx`; the derivation is in `map_zoom_levels.md`.

## 6. In the code

```python
class Camera:
    def __init__(self, table):
        ...
        self.fx, self.fy = table["cam_fx"][0], table["cam_fy"][0]
        self.cx, self.cy = table["cam_cx"][0], table["cam_cy"][0]
        self.k1, self.k2, self.k3 = ...          # lens distortion (lens_distortion.md)
        self.p1, self.p2 = ...
        self.width = 320
        self.height = 256
```

The nine camera numbers are read once from the first CSV row (after a check that every row has the same numbers). They are used in two places:

1. `make_query` (`geometry.py`): `map_x = fx * x_real + cx` is the photo pixel that a canvas pixel will copy its brightness from.
2. `level_for_height`: `footprint = camera.width * height / camera.fx`.

**Constraints:** the width and height (320 × 256) are written into the class. A different camera needs those two lines changed, and `settings.TILE` reconsidered.

## 7. Mistakes to avoid

* Mixing up `u` (column, horizontal) and `v` (row, vertical).
* Using a focal length in millimetres. Here it is always in pixels.
* Forgetting that the pinhole formula needs `Zc > 0`; points behind the camera produce nonsense, which is why `make_query` has an `in_front` mask.

## 8. Check yourself

1. A ground point 30 m east, 0 m south of nadir, seen from 150 m. Which `u`? *(x = 0.2, u = 200·0.2 + 159.5 = 199.5, the same pixel as in the example: only the ratio X/h matters.)*
2. What is the GSD at 250 m? *(1.25 m/pixel)*
3. Why is the "half" in `W/2` used in the FOV formula? *(The extreme pixel is half a width from the centre.)*
