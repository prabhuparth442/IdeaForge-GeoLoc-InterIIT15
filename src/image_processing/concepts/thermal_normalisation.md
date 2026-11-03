# Making a thermal photo comparable: percentile stretch and CLAHE

Code: `src/image_processing/thermal.py`.

---

## 1. The problem

A thermal camera records 16-bit counts that depend on temperature, but also on gain, offset, and drift. Two photos of the same place minutes apart may use totally different ranges. One real example from the dataset: all pixels of a photo lay between 20386 and 21825 out of 65535, so shown directly the picture is a flat grey rectangle.

We therefore normalise **every photo by itself**. Only the *pattern* of light and dark should survive, not the absolute numbers. (The edge-direction features in `edge_direction_features.md` add a second layer of insensitivity.)

## 2. Step 1: percentile stretch

Let `I` be the photo, `a` its 1st percentile and `b` its 99th percentile. Map `[a, b]` linearly onto `[0, 1]` and cut off the rest:

$$I'=\operatorname{clip}\!\Big(\frac{I-a}{b-a+\varepsilon},\,0,\,1\Big),\qquad\varepsilon=10^{-6}.$$

**Why percentiles instead of min/max?** One dead pixel or a hot chimney sets the maximum. Min/max stretching would then squeeze all useful structure into a small part of the range. Percentiles ignore the most extreme 1 % on each side, so a few outliers cannot ruin the contrast. The `ε` prevents a division by zero for a perfectly flat photo.

**Why this is "affine invariant":** if the camera changed the photo from `I` to `αI + β` (gain `α > 0`, offset `β`), the percentiles transform the same way, `a → αa+β`, `b → αb+β`, so `(I−a)/(b−a)` is *unchanged*. The stretch removes any linear brightness change exactly.

## 3. Step 2: CLAHE

After the stretch, large dark or bright regions can still hide local detail. **Histogram equalisation** spreads the brightness values so that all grey levels are used equally.

### Plain histogram equalisation, derived

Let brightness `r` have probability density `p(r)` and cumulative distribution `F(r) = ∫₀^r p(t)dt`. Define a new brightness `s = F(r)`. Then `s` is uniform on `[0, 1]`:

$$P(s\le t)=P\big(F(r)\le t\big)=P\big(r\le F^{-1}(t)\big)=F\big(F^{-1}(t)\big)=t.$$

(This is the *probability integral transform*.) Practically: sort the pixels, replace each value by its rank divided by the number of pixels. Dark gradients get stretched, flat regions get compressed.

### Making it local and tame (CLAHE)

*Adaptive*: do the above separately in a grid of tiles (here 8 × 8 tiles over the 320 × 256 photo), then smooth the tile boundaries by interpolating between neighbouring tile mappings.

*Contrast limited*: equalisation amplifies noise in flat areas (a flat area's histogram is a narrow spike, its CDF a steep step, so tiny noise becomes a big jump). CLAHE clips each tile's histogram at a ceiling (`clipLimit = 2.0` times the average bin height) and redistributes the clipped excess evenly over all bins before computing the CDF. This caps the slope of the CDF, therefore the maximum local contrast gain.

OpenCV's CLAHE needs integer input, so the code converts to `uint16` first:

```python
as_16bit = (stretched * 65535).astype(np.uint16)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
return clahe.apply(as_16bit).astype(np.float32) / 65535
```

## 4. Complete function

```python
def normalise_frame(frame, low_pct=1, high_pct=99):
    frame = frame.astype(np.float32)
    low, high = np.percentile(frame, [low_pct, high_pct])
    stretched = np.clip((frame - low) / (high - low + 1e-6), 0, 1)
    ...CLAHE...
```

**Input:** 2-D array of any numeric type (the code first averages 3-channel files into one channel in `load_thermal_frame`). **Output:** float32 array in `[0, 1]`, same shape (256 × 320).

**Caveats**

* CLAHE also amplifies sensor noise. The later steps blur lightly (σ = 1 px before the gradient, and a further anti-aliasing blur before resampling) to remain robust.
* Both the thermal photo and the satellite map receive CLAHE (the map version in `map_zoom_levels.md`); they are different *sizes of tile* but chosen so the contrast scale in the 224-pixel canvas is comparable (8 regions across).
* Reading the file: `cv2.imread(path, cv2.IMREAD_UNCHANGED)` keeps the 16-bit depth; the default flag would silently convert to 8-bit.
