# From matched points to a shift: similarity transform and RANSAC

**Code:** [`../ransac.py`](../ransac.py) (`fit_transform`).
**You need first:** [`keypoint_matching_superpoint_lightglue_minima.md`](keypoint_matching_superpoint_lightglue_minima.md) (where the point pairs come from).

---

## 1. The question this note answers

The matcher hands us a list of point pairs: point `k` in the canvas and its partner in the map tile. Some pairs are right, many are wrong. We want to know three things:

1. **Is there one consistent way of placing the canvas on the tile?** (a transform)
2. **How many pairs agree with it?** (the inlier count; our confidence signal)
3. **How far did the canvas centre move?** (the shift; our refined answer)

## 2. Which transform? The similarity transform

Both pictures are already north-up with the same pixel size. If the tile is centred exactly on the drone, a canvas point and its partner differ by nothing. If the tile is a bit off, they differ by a constant shift. Because heading, height and the lens are not perfect there may also be a tiny rotation and a tiny scale error. So we allow:

> rotation + uniform scale + shift: a **similarity transform**, 4 unknown numbers.

We deliberately do not allow squeezing, shearing or perspective: they would let a wrong alignment find a "good" fit by contorting itself.

### A neat way to write it: complex numbers

Write a point `(x, y)` as the complex number `z = x + iy`. In complex arithmetic, multiplying by a number rotates and scales; adding moves:

$$z'=a\,z+b,\qquad a=s\,e^{i\theta}=s\cos\theta+i\,s\sin\theta,\qquad b=t_x+i\,t_y.$$

`s` is the scale, `θ` the rotation, `b` the shift. In ordinary matrix form (`a_r = s cos θ`, `a_i = s sin θ`):

$$\begin{pmatrix}x'\\y'\end{pmatrix}=\begin{pmatrix}a_r&-a_i\\a_i&a_r\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}+\begin{pmatrix}t_x\\t_y\end{pmatrix}.$$

OpenCV returns exactly this as the 2 × 3 matrix `[[a_r, −a_i, t_x], [a_i, a_r, t_y]]`, and the code reads the scale and angle back off it:

```python
scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)       # s = |a|
angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))    # theta = arg(a)
```

### Two pairs are enough: a worked example

Take two pairs: canvas `(10, 20) → (13, 24)` and canvas `(50, 20) → (53, 24)`. As complex numbers `z₁ = 10+20i`, `z₁' = 13+24i`, `z₂ = 50+20i`, `z₂' = 53+24i`. Subtract the two equations `z' = a z + b` to remove `b`:

$$a=\frac{z_1'-z_2'}{z_1-z_2}=\frac{-40}{-40}=1,\qquad b=z_1'-a\,z_1=(13+24i)-(10+20i)=3+4i.$$

So `a = 1` (no rotation, no scale change) and the shift is `(3, 4)` pixels. A second example with a rotation: `(0,0) → (10, 5)` and `(100, 0) → (114.85, 10.50)`. Then `a = (−104.85 − 5.5i)/(−100) = 1.0485 + 0.055i`, giving `s = |a| = 1.05` and `θ = 3.0°`: a 5 % scale error and a 3° rotation, which the gate in §5 would accept.

Two pairs form the **minimal sample**: 4 unknowns, and each pair gives 2 equations.

### Many pairs: least squares

With `n` pairs we pick `a, b` that minimise the total squared error `Σ |a z_i + b − z_i'|²`. With the centroids `z̄` and `z̄'` the best answer is

$$a=\frac{\sum_i\overline{(z_i-\bar z)}\,(z_i'-\bar z')}{\sum_i|z_i-\bar z|^2},\qquad b=\bar z'-a\,\bar z.$$

(Set the derivative with respect to `b` to zero to get `b = z̄' − a z̄`; substitute and differentiate with respect to `a`.) Using all points averages out the small errors of single points; that is why the final shift is accurate to a fraction of a pixel. But there is a catch.

## 3. The catch: wrong pairs ruin the average

Many of LightGlue's pairs are wrong (**outliers**): a window corner matched to a road corner elsewhere. One pair that is 100 pixels off drags a least-squares fit far from the truth. We need a method that finds the fit supported by the *majority-consistent* pairs and ignores the rest.

## 4. RANSAC: guess, check, repeat

**RANSAC** (RANdom SAmple Consensus) in four steps:

1. Pick **2 pairs at random** (a minimal sample).
2. Compute the transform from just those two (§2).
3. Count **inliers**: pairs whose error `|a z + b − z'|` is below a threshold (`RANSAC_PIXELS = 3.0` canvas pixels).
4. Remember the transform with the most inliers. Repeat many times. At the end, refit using all inliers of the best one (least squares), which gives the accurate final answer.

```python
matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC,
    ransacReprojThreshold=RANSAC_PIXELS, maxIters=3000, confidence=0.995)
```

(`estimateAffinePartial2D` is OpenCV's name for "similarity transform with 4 degrees of freedom".)

If the two pairs picked happen to be correct, the resulting transform is good and many other correct pairs will vote for it. If one of them is wrong, the transform is nonsense and few pairs vote. So we only need to be lucky once.

### How many draws are enough?

Let `w` be the fraction of correct pairs. One draw of 2 is all-correct with probability `w²`. After `k` draws, the probability that at least one was all-correct is `1 − (1 − w²)^k`. Setting that to at least `P = 0.995` and solving gives

$$k\ \ge\ \frac{\ln(1-P)}{\ln(1-w^2)}.$$

| fraction of correct pairs `w` | `w²` | draws needed (rounded up) |
|---|---|---|
| 0.5 | 0.25 | 19 |
| 0.2 | 0.04 | 130 |
| 0.1 | 0.01 | 528 |

`maxIters = 3000` is far above these numbers, so RANSAC finds the right transform even when 90 % of the pairs are wrong. OpenCV also stops early once its own confidence criterion is met.

## 5. Why the inlier count is strong evidence

Suppose the candidate is a **wrong** place. Then partners are essentially random positions within the 224 × 224 tile. For a given transform, a random pair agrees within 3 pixels with probability about the area of a 3-pixel disc divided by the tile area:

$$p=\frac{\pi\cdot 3^2}{224^2}\approx5.6\cdot10^{-4}.$$

With 300 pairs we expect `λ = 300·p ≈ 0.17` accidental inliers. The chance of **12 or more** is the Poisson tail

$$P(N\ge12)\approx e^{-\lambda}\frac{\lambda^{12}}{12!}\approx 10^{-18}.$$

So for random mismatches, 12 inliers essentially never happen, even though RANSAC tries thousands of transforms.

**The honest caveat.** Real mismatches are not uniformly random. In a repetitive scene (rows of identical houses, a regular field grid) wrong matches are *coherent*: they agree on one shift because the scene repeats. Then many inliers appear at a wrong place. So the count is strong evidence, not proof, and that is why the project's decision rule (`result_thresholding/`) is tuned on data, not derived from this calculation.

## 6. The sanity gate

```python
if abs(scale - 1) > 0.15 or abs(angle) > 12:
    return None, np.zeros(len(points_a), dtype=bool)
```

Canvas and tile are both north-up at the same pixel size, so the true transform is nearly a pure shift: `s ≈ 1`, `θ ≈ 0`. A fit that stretches by more than 15 % or rotates by more than 12° found its inliers by assuming a wrong geometry. It is thrown away and the candidate gets **zero inliers**. The limits are generous because heading and height have small errors. (The 3° / 5 % example above passes; a 20° fit would not.)

If there are fewer than 4 pairs, or OpenCV returns nothing, the function also returns `(None, all False)`.

## 7. Run it without any model

```bash
PYTHONPATH=src python -c "
import numpy as np
from fine_matching.ransac import fit_transform
rng = np.random.default_rng(0)
a = (rng.random((40,2))*200).astype(np.float32)
b = a + np.array([5,-3], dtype=np.float32); b[:10] = (rng.random((10,2))*200)   # 10 outliers
M, inl = fit_transform(a, b); print(M.round(2).tolist(), int(inl.sum()))"
# [[1.0, -0.0, 5.0], [0.0, 1.0, -3.0]] 30
```

Forty pairs, 30 of which are shifted by (5, −3) and 10 replaced by random points. The recovered matrix is the pure shift (5, −3) (`a = 1`, `b = 5 − 3i`), and the 30 true pairs are the inliers. The 10 outliers did not disturb it.

## 8. Inputs, outputs, constraints

`fit_transform(points_a, points_b)`

* `points_a`, `points_b`: float32 arrays `(n, 2)`, columns `x, y` in canvas pixels; row `k` of A is the partner of row `k` of B (asserted).
* Returns `(matrix 2×3 float64 or None, inliers bool (n,))`.
* Determinism: RANSAC uses OpenCV's random generator, which `localise_photo` seeds per photo (`cv2.setRNGSeed(SEED)`; see `result_thresholding/concepts/one_photo_at_a_time.md`), so repeated runs give the same result.

## 9. Common mistakes

* Reading the inlier count of a candidate whose fit was rejected by the gate: it is 0 by design.
* Using a threshold in metres instead of pixels (`RANSAC_PIXELS` is in canvas pixels, which is 0.85 m at level 3 and 3.4 m at level 7, so 3 pixels is 2.5 to 10 m).
* Believing 12 inliers is always safe: see the repetitive-scene caveat.

## 10. Check yourself

1. Given the pairs `(0,0) → (2,1)` and `(10,0) → (12,1)`, what are `a` and `b`? (`a = 1`, `b = 2 + i`.)
2. Why does RANSAC need only about 19 draws when half the pairs are correct?
3. Why is a pure least-squares fit on all pairs a bad idea here?
