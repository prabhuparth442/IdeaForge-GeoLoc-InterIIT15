# How much does an input error cost? Sensitivity analysis

**Code:** [`../sensitivity.py`](../sensitivity.py) (`share_for`, `run_sensitivity`).
**You need first:** the geometry note just before this one, and `image_processing/concepts/rotations_and_the_north_up_canvas.md`.

---

## 1. The idea

Every input of the pipeline has an error: the heading from the compass, the tilt angles, the height, the camera scale. Two practical questions follow:

1. **How big an error can we tolerate?**
2. **Which input is the dangerous one?**

We answer with two tools: simple geometry that predicts the effect on the ground (this note, sections 2 and 3), and a measurement with the real pipeline (sections 4 and 5).

## 2. Geometry first: what a given error does on the ground

### Tilt

A camera tilted by angle `δ` at height `h` sees the ground point under it displaced horizontally from the nadir by `h·tan δ`. If the tilt is ignored (or has the wrong sign), the position error is about

$$\Delta\approx h\tan\delta.$$

| height | δ = 2° | δ = 5° | δ = 10° |
|---|---|---|---|
| 100 m | 3.5 m | 8.7 m | 17.6 m |
| 200 m | 7.0 m | 17.5 m | 35.3 m |
| 350 m | 12.2 m | 30.6 m | 61.7 m |

Compare with σ = 8.1 m at 100 m and 9.6 m at 350 m: a 5° tilt left uncorrected at 350 m costs about **three σ**, which drops the answer from 10 points to 3 or 0. This is why the geometry includes the tilt correction and why the sweep tests its signs.

### Heading

An angular error `δ` (in radians) rotates the canvas around its centre. A feature at distance `d` from the centre moves along an arc of length `d·δ`. The centre does not move at all; the edge moves most. Half the canvas is 112 pixels:

| level | metres per pixel | centre-to-edge distance | shift at the edge for a 5° error |
|---|---|---|---|
| 3 | 0.849 | 95 m | 8.3 m |
| 4 | 1.200 | 134 m | 11.7 m |
| 5 | 1.697 | 190 m | 16.6 m |
| 6 | 2.400 | 269 m | 23.5 m |
| 7 | 3.394 | 380 m | 33.2 m |

(5° = 0.0873 rad; for instance 95 m × 0.0873 = 8.3 m.) These shifts are all within a few canvas cells, so the edge-direction cells blur them out, and the *nadir point* at the canvas centre does not move at all. The coarse search is therefore expected to tolerate several degrees of heading error.

### Scale (and height)

If the pixel size is wrong by factor `s`, a feature at distance `d` from the centre appears at `s·d`, so the position error there is `(s − 1)·d`. For `s = 1.04`: 3.8 m at 95 m from the centre, 15.2 m at 380 m. Because the ground pixel size is proportional to height, a **height error** `Δh` acts as a scale error of `1 + Δh/h`: a 10 m error at 200 m height is the same as scale 1.05.

## 3. Summary of the geometry

| input error | effect on a feature at distance `d` | worst place |
|---|---|---|
| tilt ignored | whole canvas shifted by `h tan δ` | everywhere |
| heading δ | moved by `d·δ` | canvas edge |
| scale s | moved by `(s−1)·d` | canvas edge |
| height Δh | same as scale `1 + Δh/h` | canvas edge |

The nadir point (canvas centre) is untouched by heading and scale errors, but shifted by a tilt error. That is why tilt is the input to be most careful about.

## 4. The measurement

```python
for delta in [-12, -6, -3, 0, 3, 6, 12]:
    share = share_for(testset, store, ids, dict(base, yaw_offset=base["yaw_offset"] + delta))
for factor in [0.85, 0.92, 1.0, 1.08, 1.15]:
    share = share_for(testset, store, ids, dict(base, scale=base["scale"] * factor))
```

Procedure:

1. pick 18 random sample photos (seeded);
2. for each heading offset (degrees) and then each scale factor, render the canvases with the **deliberately wrong** value, run the coarse search, and compute the share of photos where the top peak is within 4σ (`share_for`);
3. print one line per setting.

The curve is the tolerance. Where it falls from its plateau is the largest error that the first stage survives.

## 5. Reading the output

* A **flat plateau** around 0 and a drop at ±6° or ±12° tells you the system tolerates heading noise of that size.
* **A peak not at 0**: the configured `yaw_offset` or `scale` is slightly off, and the best delta is the correction.
* **18 photos means steps of 0.056** in share. Ignore differences below about 0.1.
* The expectation (not a measured result): because cells are 4 × 4 pixels, a small rotation or scale error changes the cell statistics little, so the curve should be flat near zero and then fall. Real numbers need a run: `python run.py --sensitivity`.

## 6. Extending it

`run_sensitivity` sweeps **heading and scale**. To test height or tilt, copy the function and change the key being varied (`height_offset`, `pitch_sign`, or a new `tilt_scale` if you add one). Each setting costs 18 coarse searches.

## 7. Common mistakes

* Sweeping on photos where the effect cannot appear (very flat photos hide tilt effects).
* Comparing shares from different photo samples.
* Concluding "robust" from a flat curve that only reaches ±3°: test beyond the realistic error range.

## 8. Check yourself

1. A drone at 200 m height has an uncorrected 3° tilt. What is the position error? (`200·tan 3° ≈ 10.5 m`, about 1.2σ.)
2. A feature is 200 m from the canvas centre and the heading is off by 2°. How far does it move? (`200·0.0349 = 7.0 m`.)
3. Why does a heading error not move the nadir point?
