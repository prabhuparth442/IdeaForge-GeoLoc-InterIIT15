# How much does an input error cost? Sensitivity analysis

Code: `src/evaluation/sensitivity.py` (`share_for`, `run_sensitivity`).

---

## 1. Idea

Every input to the pipeline (heading, tilt, height, camera scale) has an error. We need to know **how much error we can tolerate** and which input is the dangerous one. Two tools: calculate the effect on the ground (geometry), and measure it with the pipeline (what the code does).

## 2. Geometric estimates

**Tilt.** A camera tilted by angle `δ` while the drone is at height `d` sees the ground point offset horizontally by `d·tan δ` from nadir. If the tilt is ignored (or has the wrong sign) the position error is about

$$\Delta\approx h\tan\delta.$$

* h = 100 m, δ = 5°: 8.7 m; h = 350 m: 30.6 m (about three σ).
* So tilt matters more for high flights; this is why tilt correction is in the geometry and is tested in the sweep.

**Heading.** An angular error `δ` rotates the canvas around its centre. A feature at radius `d` from the centre moves by the arc `d·δ` (δ in radians). With `d` = 100 m and δ = 5° = 0.0873 rad → 8.7 m. At the canvas edge features shift most, the centre not at all.

**Scale.** If the pixel size is wrong by factor `s`, a feature at distance `d` from the centre appears at `s·d`: error `(s−1)·d`. For `s = 1.04` and `d = 100` m: 4 m, growing linearly with distance from the centre.

**Height error.** The same as a scale error: pixel size is proportional to height, so a height error `Δh` gives scale `1 + Δh/h`.

## 3. The measurement

```python
for delta in [-12, -6, -3, 0, 3, 6, 12]:
    share = share_for(testset, store, ids, dict(base, yaw_offset=base["yaw_offset"] + delta))
```

Uses 18 random sample photos; for each heading error, the share of photos where the best correlation peak is within 4σ. The curve is the tolerance: where it drops from its plateau is the maximum heading error the first stage survives. The edge-direction cells are 4 × 4 pixels (3.4 to 13.6 m), so a small rotation barely changes the cell statistics: expect the curve to be flat near zero, then fall. (Expectation, not a measured result.)

## 4. Reading it

* Flat curve around 0, drop at ±6 or ±12: the system tolerates metadata noise of that size.
* Peak not at 0: the configured `yaw_offset` is slightly off; the best delta is the correction.
* 18 photos means steps of 0.056 in share: ignore differences below ~0.1.

## 5. Running

`python run.py --sensitivity`. Only heading is swept by the code (despite `run.py`'s help text mentioning scale); to sweep scale or height copy `run_sensitivity` and change the dictionary key (`scale`, `height_offset`).
