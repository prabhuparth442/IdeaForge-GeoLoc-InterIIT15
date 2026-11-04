# Resampling: interpolation, aliasing, and why we blur before shrinking

**Code:** the blur and `cv2.remap` in `make_query` (`geometry.py`), and `cv2.resize` with `INTER_AREA` in `build_level_images` (`map_levels.py`).

**What you need to know beforehand:** that a picture is a grid of numbers. A little signal-processing intuition helps (a "frequency" is how fast something changes from pixel to pixel), but it is explained as we go.

---

## 1. The big picture in plain words

Whenever we move or resize a picture, the new pixel centres fall *between* the old ones, or the new pixels are *bigger* than the old ones. Two things must be handled with care:

1. **Reading between pixels** (interpolation). We use bilinear interpolation.
2. **Making pixels bigger without creating fake patterns** (aliasing). We blur or average *before* shrinking.

Both matter because our matching compares small edge patterns. A fake pattern produced by careless shrinking would be matched against real structure on the map.

## 2. Reading a picture between its pixels (bilinear interpolation)

The warp asks for the photo's value at non-integer positions `(u, v)`, for example `(100.3, 57.8)`. A digital picture only has values at integer positions. **Bilinear interpolation** blends the four surrounding pixels.

Let `u0 = ⌊u⌋`, `v0 = ⌊v⌋`, and `a = u − u0`, `b = v − v0` (both in `[0, 1)`). With the four neighbours `I00, I10, I01, I11`:

$$I(u,v)=(1-a)(1-b)\,I_{00}+a(1-b)\,I_{10}+(1-a)\,b\,I_{01}+a\,b\,I_{11}.$$

It is "linear in `u`, then linear in `v`". The four weights are never negative and sum to 1, so the result always lies between the neighbours' smallest and largest value, and a point exactly on a pixel returns that pixel.

**Worked example.** Position `(100.3, 57.8)`: `a = 0.3`, `b = 0.8`. Neighbour values `I00 = 10`, `I10 = 20`, `I01 = 30`, `I11 = 40`. The weights are `0.7·0.2 = 0.14`, `0.3·0.2 = 0.06`, `0.7·0.8 = 0.56`, `0.3·0.8 = 0.24`. Result: `0.14·10 + 0.06·20 + 0.56·30 + 0.24·40 = 1.4 + 1.2 + 16.8 + 9.6 = 29`.

The price of interpolating is that it averages, which **slightly blurs** the picture. That is one reason the whole chain (undistort + rotate + scale) is done in a single `remap`: the photo is interpolated once, not three times.

## 3. Aliasing: what goes wrong when you shrink

Suppose the canvas pixel is bigger than a photo pixel: the canvas pixel size `g` is larger than the photo's GSD `h/f`. Let the shrink factor be `s = g / GSD_photo` (for example 2 means "each canvas pixel covers 2 photo pixels across").

If we simply picked every `s`-th photo pixel we would throw detail away, and worse: **fine patterns turn into fake coarse patterns**. A fence whose posts are 1.5 pixels apart, sampled once every 2 pixels, shows posts at a made-up spacing. This is **aliasing**. The same effect makes wagon wheels look as if they turn backwards in films.

The sampling theorem says: if you sample every `g` metres you can only represent spatial frequencies below the **Nyquist frequency** `1/(2g)` (that is, patterns with at least two samples per period). Anything finer must be removed **before** sampling. Removing fine detail is a **low-pass filter**, in other words a **blur**.

A Gaussian blur of width `σ` multiplies each spatial frequency `f` by

$$H(f)=e^{-2\pi^{2}\sigma^{2}f^{2}}.$$

Choose `σ ≈ g/2` (in ground units). At the Nyquist frequency `f = 1/(2g)`:

$$H=\exp\!\Big(-2\pi^2\cdot\frac{g^2}{4}\cdot\frac{1}{4g^2}\Big)=e^{-\pi^2/8}\approx0.29,$$

so content right at the limit is cut to about 29 %, and anything finer vanishes quickly. In photo pixels the width is `σ = 0.5·s`.

The code uses this rule with a small empirical correction:

```python
photo_gsd = item["rel_height"] / camera.fx
blur = max(0.0, 0.5 * pixel_size / photo_gsd - 0.25)
if blur > 0.2:
    photo = cv2.GaussianBlur(photo, (0, 0), blur)
```

* `0.5·s` is the Nyquist rule above.
* `− 0.25` subtracts blur that the photo already has from its own pixel footprint and from the bilinear interpolation.
* `blur > 0.2` skips pointless tiny blurs.
* When the canvas pixel is smaller than the photo pixel (upsampling), `s < 1` and the rule gives 0.

| shrink factor s | blur σ (photo pixels) |
|---|---|
| 1 | 0.25 (a very light blur is applied, since 0.25 > 0.2) |
| 2 | 0.75 |
| 3.5 | 1.5 |

This is a heuristic, not an exact filter design. What matters is that it grows with the shrink factor and is zero when nothing is shrunk. `cv2.GaussianBlur(img, (0, 0), sigma)`: the kernel size `(0, 0)` tells OpenCV to derive it from `sigma`.

## 4. Shrinking the map: `INTER_AREA`

The satellite map has 0.3 m pixels, and the zoom levels are 0.85 to 3.4 m. That is a shrink factor between 2.8 and 11.3 on a huge array. We need the same anti-aliasing, cheaply. `cv2.resize(..., interpolation=cv2.INTER_AREA)` replaces each output pixel by the **area-weighted mean** of the source pixels it covers: a box filter as wide as the shrink factor. A box is a poorer low-pass filter than a Gaussian, but it is fast, never invents values and leaves nothing aliased in practice.

Order matters: first **average** the grey values (`resize`), *then* convert to 8 bits (`grey_to_uint8`). Converting first would round every source pixel to a whole number before averaging, which loses information.

## 5. Strip-wise shrinking and the rounding of rows

To save memory, the map is shrunk in horizontal strips of 2048 rows. For level `L` the factor is `scale = 0.3 / gsd_L = 2^(−L/2)`. A strip that starts at row `top` and has `rows` rows goes to these output rows:

```python
y0 = int(round(top * scale))
y1 = min(int(round((top + rows) * scale)), out_h)
small = cv2.resize(grey, (out_w, y1 - y0), interpolation=cv2.INTER_AREA)
```

Each strip is resized to *exactly* `y1 − y0` output rows. This keeps strips seamless (no gaps, no overlaps), even though `2048·scale` is rarely a whole number.

The consequence: inside one strip the vertical scale can differ from `scale` by up to one output pixel over the strip's height. One output pixel is 0.85 m at level 3 and 3.4 m at level 7. That is smaller than the allowed error σ (8 to 10 m), but at level 7 it is already about a third of it, so it is a real, bounded approximation. If it ever matters, build each level with one `cv2.resize` over the whole map instead of strip by strip. That needs the whole grey map in memory (about 1.2 GB per band as float32).

## 6. Summary

| operation | where | why |
|---|---|---|
| Gaussian blur, σ = 0.5·s − 0.25 | `make_query`, before the warp | remove frequencies the coarser canvas cannot hold |
| bilinear `remap` | `make_query` | read the photo at fractional positions, all geometry in one step |
| `INTER_AREA` resize | `build_level_images` | alias-free shrinking of the huge map |
| `BORDER_CONSTANT` plus a mask | `make_query` | outside the photo there is no data, and we remember where |

## 7. Mistakes to avoid

* Shrinking with plain sampling (`INTER_NEAREST`, or `INTER_LINEAR` for a big shrink). It aliases.
* Interpolating several times in a row (undistort, then rotate, then resize) when one `remap` can do it all.
* Converting to 8 bits before averaging.

## 8. Check yourself

1. What does bilinear interpolation return at `a = 0, b = 0`? *(Exactly `I00`.)*
2. Why blur *before* shrinking and not after? *(After shrinking, the fake patterns already exist and cannot be told from real ones.)*
3. Roughly how much blur for a canvas pixel 3 times bigger than the photo pixel? *(`0.5·3 − 0.25 = 1.25` photo pixels.)*
