# Testing geometry guesses: yaw offset, tilt signs, height, scale

Code: `src/evaluation/geometry_sweep.py` (`share_within_4_sigma`, `sweep_geometry`).

---

## 1. Why this exists

The metadata has angles and heights, but conventions are not documented (is positive pitch nose up? does yaw 0 mean north? is the height above ground or sea level?). A wrong sign makes the north-up canvas wrong, and the correlation will fail even though the code is right. Instead of guessing, we **let the data choose**: try each hypothesis, run the first stage, see which one finds the truth most often.

## 2. The test statistic

```python
errors_in_sigma = error_m / sigma
return (errors_in_sigma <= 4).mean(), errors_in_sigma
```

Share of photos whose top correlation peak is within 4σ of the truth. Only the coarse stage is needed (fast, no network), and 4σ is generous enough to count a close-but-not-perfect lock as success while wrong geometry gives roughly random positions inside the disc (share near 0).

## 3. The sweep, step by step

1. **Yaw offset** on nearly flat photos (|pitch|, |roll| < 8°, tilt ignored): try 0, 90, 180, 270°. If the yaw convention is off by a quarter turn, only one offset aligns the canvas with the map. Choose the best share (ties: smaller median error).
2. **Tilt signs** on photos with tilt ≥ 6° (up to 40): five hypotheses: no tilt correction, and the four sign combinations of (pitch, roll). Photos with large tilt are used because on near-flat photos all five render almost the same picture and cannot be told apart.
3. **Height offset:** compare the best hypothesis with the same one using `height_offset = −4` m; switch only if the share improves by more than 0.05 (to avoid chasing noise).
4. **Scale:** 0.96 … 1.16; keep the best.
5. Print the chosen geometry and compare it with `settings.GEOMETRY`; the final line says "same" or tells you to update the settings.

## 4. Sampling noise

Samples are 24 to 40 photos, so a share difference of 0.05 is 1–2 photos. A hypothesis wins reliably only when the separation is large (e.g. 0.6 vs 0.1). Treat a close call as "not determined" and look at the median error in σ too.

## 5. Running it

`python run.py --sweep` (needs map data and the first-stage code only; no GPU strictly required but slow on CPU). Cost = (#hypotheses × #photos) searches ≈ (4×24 + 5×40 + 1×40 + 6×40) coarse searches.

## 6. Pitfall fixed in the code

Each stage builds on the geometry chosen so far via `dict(geometry, ...)`. A previous version merged a *partial* geometry with the default scale 1.04; the current code starts every hypothesis from a full dictionary.
