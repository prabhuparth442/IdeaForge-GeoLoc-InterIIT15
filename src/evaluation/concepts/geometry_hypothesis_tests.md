# Testing geometry guesses: yaw offset, tilt signs, height, scale

**Code:** [`../geometry_sweep.py`](../geometry_sweep.py) (`share_within_4_sigma`, `sweep_geometry`).
**You need first:** `image_processing/concepts/rotations_and_the_north_up_canvas.md` (what yaw, pitch and roll do to the canvas).

---

## 1. Why this exists

The photo metadata has angles and heights, but the **conventions are not documented**. Is positive pitch nose up or nose down? Does yaw 0 mean north? Is a roll to the right positive? Is the height above ground, or above the take-off point? Each wrong guess builds a wrong north-up canvas, and then the search fails even though every line of code is right. That failure is silent and looks like "the method does not work".

Instead of guessing, we **let the data choose**: try each hypothesis, run the first stage with it, and see which hypothesis finds the truth most often.

## 2. The test statistic: share within 4σ

```python
errors_in_sigma = error_m / sigma
return (errors_in_sigma <= 4).mean(), errors_in_sigma
```

For each hypothesis and each test photo: render the canvas, run the coarse search against the photo's simulated prior, take the **top peak**, and compute its error in σ units. The statistic is the share of photos for which that error is at most 4σ.

Why this statistic works:

* It needs **only the coarse stage**: fast, no neural network.
* A correct geometry gives a lock close to the truth (share high, e.g. 0.6 or more).
* A wrong geometry makes the canvas look unlike the map; the top peak is then essentially a random place in the 5 km disc, which is almost never within 4σ (≈ 10 to 40 m) of the truth. The share is then near 0 (the chance of landing within 40 m by luck in a disc of area `π·5000²` is `(40/5000)² ≈ 6·10⁻⁵`).

That huge gap between "right" and "wrong" is what lets us decide with few photos.

## 3. The sweep, step by step

The code starts from a neutral geometry (`attitude=True`, signs +1, `yaw_offset=0`, `scale=1.0`, `height_offset=0`) and fixes one thing at a time, each time keeping what was chosen before.

1. **Yaw offset**, on nearly flat photos (|pitch| and |roll| below 8°, tilt correction off, up to 24 photos). Try offsets 0, 90, 180, 270°. If the heading convention is off by a quarter turn, only one of the four makes the canvas line up with the map. Choose the best share (ties: smaller median error). Flat photos are used so that the tilt cannot confuse the result.
2. **Tilt signs**, on photos with tilt ≥ 6° (|pitch| + |roll|), up to 40. Five hypotheses: no tilt correction, and the four sign combinations of (pitch, roll). Tilted photos are used because on near-flat photos all five render almost the same picture and cannot be told apart.
3. **Height offset.** Compare the best hypothesis against the same one with `height_offset = −4` m. Switch only if the share improves by more than 0.05, to avoid chasing noise.
4. **Scale.** Try 0.96, 1.0, 1.04, 1.08, 1.12, 1.16 and keep the best.
5. Print the chosen geometry and compare it with `settings.GEOMETRY`. The last line says "same" or tells you to update `header/settings.py`.

## 4. Sampling noise: how to read a close call

The samples are 24 to 40 photos, so a difference of 0.05 in share is 1 to 2 photos. A hypothesis wins reliably only when the gap is large (for example 0.6 against 0.1). Treat a close call as "not determined" and look at the median error in σ as well. The 0.05 margin in step 3 and the preference for simple defaults exist for the same reason.

## 5. What it costs

`python run.py --sweep` runs `4×24 + 5×40 + 1×40 + 6×40 = 576` coarse searches. Each is one FFT correlation over the disc; this is fast on a GPU and slow on a CPU. It needs the map store but no matcher.

## 6. A pitfall that was fixed

Every stage builds on the geometry chosen so far through `dict(geometry, …)`. An earlier version merged a *partial* geometry with the default scale 1.04 and tested the wrong thing. The current code starts every hypothesis from a **complete** dictionary and only overrides the keys under test.

## 7. When to run it again

Re-run it when the data changes: a different drone, flight company or firmware may flip a sign convention. The values it prints replace the ones in `settings.GEOMETRY`. Do not carry them over to a new dataset without testing.

## 8. Common mistakes

* Judging a hypothesis on near-flat photos (all tilt hypotheses look the same there).
* Changing two things at once (then you cannot tell which helped).
* Believing a winner from a 0.05 difference.
* Forgetting that a wrong yaw offset can hide a tilt effect and the other way round, which is why the order of the steps matters.

## 9. Check yourself

1. Why do we test tilt signs on tilted photos only?
2. Why is chance agreement with the truth within 4σ so unlikely, and why does that help?
3. Which setting would you re-test first if the drone model changed?
