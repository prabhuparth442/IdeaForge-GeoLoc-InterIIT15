# Making a thermal photo comparable: percentile stretch and CLAHE

**Code:** `src/image_processing/thermal.py`. The same idea is applied to the map in `map_levels.normalise_reference`.

**What you need to know beforehand:** what a histogram is (how many pixels have each brightness).

---

## 1. The big picture in plain words

A thermal camera records numbers (16-bit counts) that depend on temperature, but also on the camera's gain, offset and drift. Two photos of the same place minutes apart may sit in completely different numeric ranges. One real photo of this dataset has all its pixels between 19789 and 21537 out of a possible 65535: shown directly, it is a nearly uniform grey rectangle.

We want only the **pattern** (which area is brighter than its neighbour), not the absolute numbers. So we normalise **each photo by itself** in two steps:

1. a **percentile stretch** to use the full brightness range,
2. **CLAHE**, a local contrast boost that makes small details visible.

(The edge-direction features in `edge_direction_features.md` add a third layer of insensitivity to brightness.)

## 2. Step 1: the percentile stretch

Let `I` be the photo, `a` its 1st percentile (1 % of pixels are darker than `a`) and `b` its 99th percentile. Map `[a, b]` linearly onto `[0, 1]` and cut off whatever lies outside:

$$I'=\operatorname{clip}\!\Big(\frac{I-a}{b-a+\varepsilon},\,0,\,1\Big),\qquad\varepsilon=10^{-6}.$$

**Worked example.** A tiny photo with pixel values `[100, 102, 104, 106, 108, 110, 112, 114, 116, 400]`. The 400 is a hot pixel. With min/max stretching: `(104 − 100)/(400 − 100) = 0.013`, so every ordinary pixel ends up in `0 … 0.05`, almost black. Now imagine the same scene as a real photo with 80 000 pixels, where the hot pixel is 1 pixel in 80 000: the 99th percentile is then about 116, and `(104 − 100)/(116 − 100) = 0.25`; the hot pixel is simply clipped to 1. The picture keeps its structure.

**Why percentiles instead of the minimum and maximum?** One dead pixel or one hot chimney decides the max. Percentiles ignore the most extreme 1 % at each end.

**Why this removes gain and offset exactly.** If the camera changed the photo from `I` to `αI + β` (gain `α > 0`, offset `β`), then the percentiles change the same way: `a → αa + β`, `b → αb + β`. Hence

$$\frac{(\alpha I+\beta)-(\alpha a+\beta)}{(\alpha b+\beta)-(\alpha a+\beta)}=\frac{\alpha (I-a)}{\alpha (b-a)}=\frac{I-a}{b-a},$$

unchanged. The stretch cancels any linear brightness change exactly. The `ε` only prevents a division by zero for a perfectly flat photo.

## 3. Step 2: CLAHE

After the stretch, a large dark region or a large bright region can still hide fine detail inside it. **Histogram equalisation** spreads the brightness values so that all grey levels are used about equally often.

### Plain equalisation, derived

Let the brightness `r` have density `p(r)` and cumulative distribution `F(r) = ∫₀^r p(t) dt` (the share of pixels darker than `r`). Define the new brightness `s = F(r)`. Then `s` is uniform on `[0, 1]`, because

$$P(s\le t)=P\big(F(r)\le t\big)=P\big(r\le F^{-1}(t)\big)=F\big(F^{-1}(t)\big)=t.$$

This is the *probability integral transform*. In practice: sort the pixels and replace each value by its rank divided by the number of pixels. Dense brightness ranges get stretched, sparse ones get compressed.

**Tiny example.** Values `[10, 10, 10, 12, 50]` have ranks (as shares) `0.2, 0.4, 0.6, 0.8, 1.0`. After equalisation (using the share of pixels ≤ each value): `10 → 0.6`, `12 → 0.8`, `50 → 1.0`. The two close values 10 and 12 are now clearly separated from each other, which is the effect we want.

### Making it local and tame: CLAHE

CLAHE stands for **C**ontrast **L**imited **A**daptive **H**istogram **E**qualisation.

* **Adaptive:** do the equalisation separately in a grid of tiles (here 8 × 8 tiles across the photo) and blend neighbouring tiles' mappings smoothly, so each region gets its own contrast stretch.
* **Contrast limited:** plain equalisation amplifies noise in flat areas. A flat area's histogram is a narrow spike, its cumulative curve is a steep step, so tiny noise becomes a big jump. CLAHE clips each tile's histogram at a ceiling (`clipLimit = 2.0` times the average bin height) and spreads the clipped excess evenly over all bins before building the cumulative curve. A capped histogram means a capped slope of the curve, which means a capped local contrast gain.

OpenCV's CLAHE needs integer input, so the code first converts to 16-bit integers:

```python
as_16bit = (stretched * 65535).astype(np.uint16)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
return clahe.apply(as_16bit).astype(np.float32) / 65535
```

## 4. The full function

```python
def normalise_frame(frame, low_pct=1, high_pct=99):
    frame = frame.astype(np.float32)
    low, high = np.percentile(frame, [low_pct, high_pct])
    stretched = np.clip((frame - low) / (high - low + 1e-6), 0, 1)
    as_16bit = (stretched * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(as_16bit).astype(np.float32) / 65535
```

**Input:** a 2-D array of any numeric type (`load_thermal_frame` first averages three-channel files into one channel). **Output:** a float32 array in `[0, 1]` of the same shape (256 × 320).

`load_thermal_frame` reads with `cv2.IMREAD_UNCHANGED`. This matters: the default flag converts the 16-bit picture to 8 bits and silently throws away most of the information.

## 5. What can go wrong

* A constant photo gives an all-zero output (the `ε` prevents a crash, but there is no pattern to keep).
* CLAHE also amplifies sensor noise. That is why the next steps blur lightly: σ = 1 pixel before computing gradients, and an anti-aliasing blur before resampling (`resampling_and_aliasing.md`).
* The satellite map gets the same treatment (CLAHE with about 28-pixel tiles) so that the contrast scale in the 224-pixel canvas is similar for both pictures: about 8 regions across.
* Do not change `clipLimit` on one picture and not the other. Both pictures must have the same kind of processing, otherwise edge strengths differ systematically.

## 6. Check yourself

1. A photo has pixel values between 20000 and 22000. After the stretch, what happens to 21000 (assuming `a = 20000`, `b = 22000`)? *(0.5)*
2. Why are the 1st and 99th percentiles used instead of the min and max? *(Outliers like a hot pixel would squeeze everything else.)*
3. Why is plain histogram equalisation risky for a nearly flat region? *(It amplifies noise: the cumulative curve is very steep there.)*
4. Which step removes gain and offset exactly? *(The percentile stretch.)*
