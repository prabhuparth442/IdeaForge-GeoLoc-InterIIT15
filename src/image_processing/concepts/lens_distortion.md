# Lens distortion: why straight lines curve, and how we cancel it

**Code:** `distort` in `src/image_processing/camera.py`, called from `make_query` in `src/image_processing/geometry.py`.

**What you need to know beforehand:** the pinhole model (`pinhole_camera_and_ground_sample_distance.md`), in particular normalised coordinates `x = Xc/Zc`, `y = Yc/Zc`.

---

## 1. The big picture in plain words

The pinhole model promises that straight lines in the world stay straight in the picture. A real lens, especially a wide one like ours (77° field of view), does not keep that promise: it bends the picture, mostly near the edges. If we ignored this, the ground pattern at the photo's edges would be shifted by tens of pixels and would not line up with the map.

The camera manufacturer measured five numbers that describe the bending. Our job is to use them correctly. The trick, explained in Section 5, is that we apply the distortion **forwards**, which is easy, and get the correction for free.

## 2. Two kinds of bending

* **Radial distortion.** The magnification of the lens depends on the distance from the picture centre. If it decreases outwards the picture gets squeezed towards the corners (**barrel** distortion: straight lines bulge outwards like a barrel). If it increases outwards you get **pincushion**. Our camera has barrel distortion (`k1 ≈ −0.26`).
* **Tangential distortion.** The lens is not perfectly parallel to the sensor, so the picture is skewed a little sideways. For this camera the effect is tiny (`p1 ≈ 0.0001`, `p2 ≈ 0.0015`).

## 3. The equations (Brown–Conrady model)

Start with the ideal normalised coordinates `(x, y)` and let `r² = x² + y²`, the squared distance from the optical axis. The coordinates the lens really delivers are

$$x_d=x\,(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy+p_2(r^2+2x^2)$$
$$y_d=y\,(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy$$

and the pixel is `u = fx·x_d + cx`, `v = fy·y_d + cy`.

### Where the radial polynomial comes from

By symmetry the lens cannot prefer one direction, so the distorted distance depends only on `r`: `r_d = r·g(r)`, where `g` is the "magnification at distance r". `g` must be an **even** function, because a lens treats `+r` and `−r` the same. A smooth even function has a power series with only even powers:

$$g(r)=1+k_1r^2+k_2r^4+k_3r^6+\dots$$

Truncating after three terms is flexible enough for wide lenses. Since `x_d = x·g` and `y_d = y·g`, every point moves along the line to the centre. Radial distortion never moves a point sideways.

### Where the tangential terms come from

Decentring (the lens axis being slightly off the sensor axis) is a first-order effect in the offset. It adds terms linear in `p1, p2` that are polynomials of degree 2 in `x, y`. Brown's derivation gives exactly the products in the equations: `2xy`, `r² + 2x²`, `r² + 2y²`.

## 4. A worked example with the real camera

The metadata gives `fx = 200.27`, `fy = 200.20`, `cx = 158.88`, `cy = 131.69`, `k1 = −0.2570`, `k2 = 0.0483`, `k3 = 0`, `p1 = 0.00012`, `p2 = 0.00147`.

Take the ideal ray that a pinhole would show in the very corner pixel (0, 0): `x = (0 − 158.88)/200.27 = −0.793`, `y = (0 − 131.69)/200.20 = −0.658`.

* `r² = 0.629 + 0.433 = 1.062`
* `g = 1 − 0.2570·1.062 + 0.0483·1.062² = 1 − 0.273 + 0.054 = 0.782`
* So the real lens puts that ray at 78 % of the pinhole distance from the centre: `x_d ≈ −0.62`, `y_d ≈ −0.51`.

In pixels, the corner moves by about **46 pixels** towards the centre. At the edge middle the shift is smaller (about 20 pixels along the top edge). Put differently: the ray that falls in the corner of the sensor is 57.6° off-axis, far wider than a pinhole with `f = 200` would suggest. Ignoring this would misplace the photo's edge by tens of pixels, which is metres on the ground. Lens distortion matters.

## 5. Why we use the formula forwards

Usually people want to **undistort** a photo, which requires inverting the polynomial. There is no closed form, so one iterates. Our pipeline avoids it:

For each pixel of the **canvas** (the north-up picture we are building) we know the ideal ray `(x, y)` from the geometry. We want to know *which photo pixel recorded that ray*. That is exactly the forward model: ideal ray → apply `distort` → pixel. So undistortion comes out for free, without inverting anything, in the same step in which we rotate and scale (`rotations_and_the_north_up_canvas.md`).

## 6. The code

```python
def distort(camera, x, y):
    r2 = x * x + y * y
    radial = 1 + camera.k1 * r2 + camera.k2 * r2 ** 2 + camera.k3 * r2 ** 3
    x_d = x * radial + 2 * camera.p1 * x * y + camera.p2 * (r2 + 2 * x * x)
    y_d = y * radial + camera.p1 * (r2 + 2 * y * y) + 2 * camera.p2 * x * y
    return x_d, y_d
```

* `x` and `y` are whole numpy arrays, one entry per canvas pixel, so one call distorts about 50 000 points at once.
* `r2 ** 2` is `r⁴` and `r2 ** 3` is `r⁶`. Working with `r²` avoids taking square roots.
* The function does not know about pixels. It takes and returns **normalised** coordinates; `make_query` then applies `fx, cx` etc.

## 7. Constraints and mistakes

* Inputs must be normalised coordinates (dimensionless, around ±1), not pixels. Passing pixels makes `r² ≈ 10⁴` and the result nonsense. The assert checks only that `x` and `y` have the same shape.
* The polynomial is only trustworthy inside the calibrated field of view. Far outside (|x| beyond about 1.2) it can curl backwards and fold the picture. The validity mask in `make_query` discards canvas pixels whose photo coordinate falls outside the 320 × 256 sensor.
* The sign convention of `k1` is "distorted = ideal × (1 + k1 r² …)". Another library might define the inverse; always check the definition against the calibration source.

## 8. Check yourself

1. Which of `k1, k2, k3, p1, p2` move points only along the line to the centre? *(`k1, k2, k3`.)*
2. If `k1 < 0`, do corner points move towards or away from the centre? *(Towards: barrel distortion.)*
3. Why is `distort` applied to the canvas pixel's ideal ray rather than to the photo's pixels? *(Because we run the warp backwards: for each output pixel we ask where to read from; this makes the correction need no inversion.)*
