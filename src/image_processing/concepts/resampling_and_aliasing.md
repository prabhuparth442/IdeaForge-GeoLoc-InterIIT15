# Resampling: interpolation, aliasing, and why we blur before shrinking

Code: `geometry.make_query` (the blur and `cv2.remap`), `map_levels.build_level_images` (`cv2.resize` with `INTER_AREA`).

---

## 1. Reading a picture between its pixels

The warp asks for the photo's value at non-integer positions `(u, v)`. A digital picture only knows integer positions. **Bilinear interpolation** guesses the value by blending the four surrounding pixels.

Let `u0 = ⌊u⌋`, `v0 = ⌊v⌋`, `a = u − u0`, `b = v − v0` (both in `[0,1)`). With neighbours `I00, I10, I01, I11`:

$$I(u,v)=(1-a)(1-b)\,I_{00}+a(1-b)\,I_{10}+(1-a)\,b\,I_{01}+a\,b\,I_{11}.$$

It is "linear in `u`, then linear in `v`", so the four weights sum to 1, and a point exactly on a pixel returns that pixel. The cost: interpolation averages, which slightly **blurs**. That is another reason the whole chain (undistort + rotate + scale) is done as one `remap`.

## 2. Aliasing: what goes wrong when you shrink

Suppose the canvas pixel is bigger than a photo pixel (`g` larger than the photo's GSD `h/f`), as happens at higher zoom levels. Simply sampling every `s`-th photo pixel (`s = g/GSD_photo`) throws away detail, and worse, **fine patterns masquerade as coarse patterns**. A picket fence with a 1.5-pixel period sampled once per 2 pixels shows a fake coarse pattern. This is aliasing. The sampling theorem says a signal sampled every `g` metres can only represent spatial frequencies below the **Nyquist frequency** `1/(2g)`; higher frequencies must be removed *before* sampling.

The removal is a low-pass filter, i.e. a **blur**. A Gaussian blur of standard deviation `σ` multiplies each spatial frequency `f` by

$$H(f)=e^{-2\pi^{2}\sigma^{2}f^{2}}.$$

Choose `σ ≈ g/2` (in ground units). At the Nyquist frequency `f = 1/(2g)`:

$$H=\exp\!\big(-2\pi^2\cdot\tfrac{g^2}{4}\cdot\tfrac{1}{4g^2}\big)=e^{-\pi^2/8}\approx0.29,$$

so content right at the limit is cut to about a quarter and anything above vanishes quickly. In photo pixels the width is `σ = 0.5·g/GSD_photo = 0.5·s`.

The code uses this idea with a small empirical correction:

```python
photo_gsd = item["rel_height"] / camera.fx
blur = max(0.0, 0.5 * pixel_size / photo_gsd - 0.25)
if blur > 0.2:
    photo = cv2.GaussianBlur(photo, (0, 0), blur)
```

`0.5·s` is the Nyquist rule above; `−0.25` subtracts blur the photo already has from its own pixel footprint and from the bilinear interpolation; `blur > 0.2` skips pointless tiny blurs. Values: `s = 1 → 0.25` (blurred slightly), `s = 2 → 0.75`, `s = 3.5 → 1.5`. When the canvas pixel is *smaller* than the photo pixel (upsampling) `s < 1` and the rule gives `0`.

This is a heuristic, not an exact filter design; what matters is that it grows with the shrink factor and is zero when no shrinking occurs.

`cv2.GaussianBlur(img, (0, 0), sigma)`: the `(0, 0)` kernel size tells OpenCV to derive the kernel size from `sigma`.

## 3. Shrinking the map: `INTER_AREA`

For the satellite map (0.3 m pixels shrunk to 0.85–3.4 m) we need the same anti-aliasing but for big arrays. `cv2.resize(..., interpolation=cv2.INTER_AREA)` replaces each output pixel by the area-weighted mean of the source pixels it covers: a box filter whose width equals the shrink factor. A box is a poorer low-pass filter than a Gaussian but costs nothing extra and never invents values. The notebook text calls this "averaging, so nothing aliases."

Order matters: first **average** the grey values (`resize`), *then* convert to 8-bit (`grey_to_uint8`). Converting to 8-bit first and then averaging would round every source pixel before averaging.

## 4. Strip-wise shrinking and the rounding of rows

The map is shrunk in horizontal strips of 2048 rows (memory). For level `L` the factor is `scale = 0.3/gsd_L = 2^{-L/2}`. A strip starting at row `top` with `rows` rows goes to output rows

```python
y0 = int(round(top * scale))
y1 = min(int(round((top + rows) * scale)), out_h)
small = cv2.resize(grey, (out_w, y1 - y0), interpolation=cv2.INTER_AREA)
```

Each strip is resized to *exactly* `y1 − y0` output rows, which keeps strips seam-free (no gaps, no overlaps) even though `2048·scale` is not an integer. The consequence: inside one strip the vertical scale can differ from `scale` by up to one output pixel over the strip's height (1 pixel is 0.85 m at level 3 up to 3.4 m at level 7), so a map position can be off by up to one level pixel. That is below the allowed error sigma (8 to 10 m), though at level 7 it is already a third of it, so it is a real (bounded) approximation. If it ever matters, build each level with one `cv2.resize` over the whole map instead of per strip (this needs the whole grey map in memory, about 1.2 GB per band as float32).

## 5. Summary table

| operation | where | why |
|---|---|---|
| Gaussian blur σ = 0.5·s − 0.25 | `make_query`, before the warp | remove frequencies the coarser canvas cannot hold |
| bilinear `remap` | `make_query` | read the photo at fractional positions, all geometry in one step |
| `INTER_AREA` resize | `build_level_images` | alias-free shrinking of the huge map |
| `BORDER_CONSTANT`, mask | `make_query` | outside the photo there is no data; remember where |
