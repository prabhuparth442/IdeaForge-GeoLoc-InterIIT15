# Lens distortion: the Brown–Conrady model

Code: `src/image_processing/camera.py` (`distort`), used in `geometry.make_query`.

---

## 1. Why a real lens is not a pinhole

A pinhole maps straight lines in the world to straight lines in the image. A real lens (especially a wide one) bends them. Two physical causes:

* **Radial distortion**: magnification depends on distance from the optical axis. If magnification falls with distance the picture is squeezed towards the corners ("barrel", looks like a barrel's bulge); if it grows, corners are stretched ("pincushion").
* **Tangential distortion**: the lens is not perfectly parallel to the sensor (decentring), which skews the picture sideways.

The camera manufacturer measured five numbers `k1, k2, k3, p1, p2` that describe this. The model is the Brown–Conrady model, used by OpenCV, ROS, COLMAP and practically every calibration toolbox.

## 2. The equations

Take the *ideal* normalised coordinates `(x, y) = (Xc/Zc, Yc/Zc)` (section 1 of `pinhole_camera_and_ground_sample_distance.md`) and let `r² = x² + y²` be the squared distance from the optical axis. The **distorted** coordinates, the ones the lens really delivers, are

$$x_d=x\,(1+k_1r^2+k_2r^4+k_3r^6)\;+\;2p_1xy+p_2(r^2+2x^2)$$
$$y_d=y\,(1+k_1r^2+k_2r^4+k_3r^6)\;+\;p_1(r^2+2y^2)+2p_2xy$$

and the pixel is `u = fx·x_d + cx`, `v = fy·y_d + cy`.

### Where the radial polynomial comes from

By symmetry of the lens the distortion cannot depend on direction, only on `r`. Write the distorted radius as `r_d = r·g(r)` where `g` is the "magnification". `g` must be an even function (the lens treats `+r` and `−r` equally), and a smooth even function has a power series with only even powers: `g(r) = 1 + k1 r² + k2 r⁴ + k3 r⁶ + …`. Truncating after three terms gives enough flexibility for wide lenses. Since `x_d = x·g`, `y_d = y·g`, the point moves along the line to the centre (radially), never sideways.

### Where the tangential terms come from

Decentring is first-order in the offset between the lens axis and the sensor, which adds terms linear in `(p1, p2)` that are polynomials of degree 2 in `x, y`. Brown's decentring-distortion derivation gives exactly the products above: `2xy`, `r² + 2x²`, `r² + 2y²`.

### A number to feel it

The official camera has `k1 ≈ −0.26`. A corner pixel of the photo has `x ≈ 0.8, y ≈ 0.64`, so `r² ≈ 1.05` and (using only `k1`) `g ≈ 1 − 0.26·1.05 ≈ 0.73`: the corner of the *ideal* image lands about 27 % closer to the centre in the real photo (the higher-order terms modify this somewhat). In pixels: the corner is at radius r = 1.02, moved by r(1 - g) ~ 0.28 normalised units = about 55 pixels at f = 200. A 55-pixel error at the corner is impossible to ignore.

## 3. In the code

```python
def distort(camera, x, y):
    r2 = x * x + y * y
    radial = 1 + camera.k1 * r2 + camera.k2 * r2 ** 2 + camera.k3 * r2 ** 3
    x_d = x * radial + 2 * camera.p1 * x * y + camera.p2 * (r2 + 2 * x * x)
    y_d = y * radial + camera.p1 * (r2 + 2 * y * y) + 2 * camera.p2 * x * y
    return x_d, y_d
```

`x` and `y` are whole numpy arrays (one entry per canvas pixel), so the function distorts a million points in one call. `r2 ** 2` is `r⁴` and `r2 ** 3` is `r⁶`; using `r2` powers avoids ever taking a square root.

## 4. Why we apply the distortion forwards

Usually people want to *undistort* a photo, which needs the inverse of the polynomial (no closed form; iterative). Our warp avoids that. For each **canvas** pixel we know the ideal ray (`x, y`) from the geometry, and we want to know *which photo pixel recorded that ray*. That is exactly the forward model: ideal → distorted → pixel. So undistortion falls out for free, with no inversion, while we also rotate and rescale (see `rotations_and_the_north_up_canvas.md`).

## 5. Constraints

* Inputs are normalised coordinates (dimensionless), not pixels. Passing pixel values here gives nonsense because `r² ≈ 10⁴`.
* The polynomial is only valid inside the calibrated field of view. Rays far outside it (|x| > ~1.2 or so) can bend backwards (the polynomial is not monotone), which would fold the picture; the validity mask later removes canvas pixels whose photo coordinate lies outside the 320 × 256 sensor.
