# The scoring rule: sigma, points, and why we sometimes say "not matched"

Code: `src/header/scoring.py`. Used by: every part of `evaluation/`, and by the decision logic's design (the `2/3` rule).

Everything the team does is judged by one number, so we derive it from scratch before touching any pipeline code.

---

## 1. The pinhole camera gives us "metres per pixel" (GSD)

A camera looking straight down at flat ground from height `h` metres. Take one point on the ground `X` metres sideways from the point directly below the camera. Draw the ray from the camera centre to that point. The ray crosses the image plane (a distance `f` behind the pinhole, `f` measured in pixels) at `u` pixels from the picture centre.

```
            camera (pinhole)
               *
              /|\
         f   / | \          similar triangles:
   image  ---/--+--\---          u / f  =  X / h
   plane    u       .
                    |  h
        ground ------+---------*-------
                  nadir  X    ground point
```

The small triangle (`f` tall, `u` wide) and the big triangle (`h` tall, `X` wide) have the same angles, so their sides are in the same ratio:

$$\frac{u}{f}=\frac{X}{h}\quad\Longrightarrow\quad u=f\,\frac{X}{h}.$$

One pixel is `u = 1`, so the ground it covers is

$$\boxed{\text{GSD}=\frac{X}{u}=\frac{h}{f}}\qquad(\text{metres per pixel}).$$

**Numbers for this project.** The official camera has `f ≈ 200` pixels, so a photo from 100 m has GSD = 0.5 m/pixel and from 350 m GSD = 1.75 m/pixel. The problem statement also says the height is never taken below 100 m for scoring (a drone flying at 60 m is treated as if it were at 100 m), which is why the code has `max(altitude_m, 100.0)`.

## 2. Sigma: how much error is "allowed"

The allowed error `σ` is built from two independent error sources:

* a fixed 8 m (reference-map and GPS tolerance), and
* 3 pixels of ground size (`3·GSD`): we cannot be trusted to find a position more accurately than a few image pixels.

Independent errors add in quadrature (variances add, not standard deviations):

$$\sigma=\sqrt{8^{2}+(3\,\text{GSD})^{2}}\quad\text{metres}.$$

| height h | GSD = h/200 | σ |
|---|---|---|
| 100 m | 0.50 m | 8.14 m |
| 200 m | 1.00 m | 8.54 m |
| 350 m | 1.75 m | 9.57 m |
| 1000 m | 5.00 m | 17.0 m (the problem statement's own example, "about 17 m") |

So on our 100–350 m data σ is always 8 to 10 m. The matching cell of the coarse search is 3.4 to 13.6 m wide (see `image_processing/concepts/map_zoom_levels.md`), the same order as σ. That is the reason a second, finer stage exists at all.

## 3. Points per photo

Let `e` be the distance between our answer and the true position (metres). Define `ratio = e/σ`.

| ratio = e/σ | points |
|---|---|
| ≤ 1 | **+10** |
| ≤ 2 | +6 |
| ≤ 4 | +3 |
| ≤ 8 | 0 |
| > 8 | **−20** |

or "Not Matched" = 0 points. The score for a set of `N` photos:

$$\text{score}=60\cdot\max\!\Big(\frac{\sum_{k}\text{points}_k}{10\,N},\;0\Big).$$

`10·N` is the maximum possible total (every photo +10), so the fraction is "share of the maximum"; times 60 puts it on a 0–60 scale; `max(·,0)` means a bad run cannot go below zero.

**Check by hand (the problem statement's example).** 100 photos: 40 with +10, 20 with +6, 2 with −20, 38 with 0.

`Σ = 400 + 120 − 40 + 0 = 480`, `480/(10·100) = 0.48`, `60·0.48 = 28.8`. `check_scoring()` prints exactly this and the code must print 28.8.

## 4. Why a wrong answer must be rare: the 2/3 rule

We may *skip* a photo (0 points) or *answer* it. Suppose that if we answer, we are right (and earn `r` points) with probability `p` and wrong (−20) with probability `1−p`. The expected gain of answering is

$$E=p\,r-20\,(1-p).$$

Answering beats skipping exactly when `E > 0`:

$$p\,r>20(1-p)\;\Longleftrightarrow\;p>\frac{20}{20+r}.$$

| reward r when right | break-even p |
|---|---|
| +10 | 2/3 ≈ 0.67 |
| +6 | 20/26 ≈ 0.77 |
| +3 | 20/23 ≈ 0.87 |

The famous "2/3" is therefore the *most optimistic* case (all correct answers earn +10). In practice some right answers only earn +3 or +6, so the real break-even is higher. This is the whole reason the project has a *confidence signal* (the number of matcher inliers, see `fine_matching/` and `result_thresholding/`): we must know when we are probably wrong.

## 5. Code walk-through (`scoring.py`)

```python
def sigma_m(altitude_m, focal_px):
    height = max(altitude_m, 100.0)          # heights below 100 m count as 100 m
    gsd = height / focal_px                  # section 1
    return math.sqrt(8.0 ** 2 + (3.0 * gsd) ** 2)   # section 2
```

```python
def points_for_error(error_m, altitude_m, focal_px):
    ratio = error_m / sigma_m(altitude_m, focal_px)
    if ratio <= 1:  return 10
    if ratio <= 2:  return 6
    if ratio <= 4:  return 3
    if ratio <= 8:  return 0
    return -20
```

The if-chain walks from the best tier to the worst; the first test that passes returns. The order matters: `ratio <= 8` must come after `<= 4`.

```python
def final_score(points_list):
    accuracy = sum(points_list) / (10 * len(points_list))
    return 60 * max(accuracy, 0)
```

Pass `0` for every skipped photo, so skipped photos still count in the denominator `N`: skipping everything gives score 0, not "undefined".

**Constraints:** `focal_px` must be the camera's focal length in pixels (`camera.fx`), not millimetres. `altitude_m` is height above ground. `final_score` raises `ZeroDivisionError` for an empty list.
