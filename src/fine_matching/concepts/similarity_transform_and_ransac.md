# From matched points to a shift: similarity transform and RANSAC

Code: `src/fine_matching/ransac.py` (`fit_transform`).

---

## 1. What we need to estimate

Both pictures (the canvas and the map tile) are already **north-up at the same pixel size**. If the tile is centred on the true position, a canvas point `p` and its map partner `p'` differ only by a translation. In practice there are small errors (a degree or two of heading, a percent of scale), so we allow a **similarity transform**: rotation, uniform scale and translation, 4 unknowns.

## 2. The similarity transform in complex numbers

Write a 2-D point `(x, y)` as the complex number `z = x + iy`. Multiplying by a complex number rotates and scales; adding translates:

$$z'=a\,z+b,\qquad a=s\,e^{i\theta}=s\cos\theta+i\,s\sin\theta,\quad b=t_x+it_y.$$

In real matrix form, with `a_r = s cosθ`, `a_i = s sinθ`:

$$\begin{pmatrix}x'\\y'\end{pmatrix}=\begin{pmatrix}a_r&-a_i\\a_i&a_r\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}+\begin{pmatrix}t_x\\t_y\end{pmatrix}.$$

OpenCV's `cv2.estimateAffinePartial2D` returns the 2 × 3 matrix `[[a_r, −a_i, t_x],[a_i, a_r, t_y]]`. From it:

```python
scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)       # s = |a|
angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))    # theta = arg(a)
```

### Two point pairs determine it

Given pairs `(z₁, z₁')` and `(z₂, z₂')`: subtract the two equations to remove `b`: `z₁' − z₂' = a(z₁ − z₂)`, so

$$a=\frac{z_1'-z_2'}{z_1-z_2},\qquad b=z_1'-a\,z_1.$$

Two pairs are the **minimal sample** for a similarity (4 unknowns, 2 equations per pair).

### Many pairs: least squares

Minimise `Σ |a z_i + b − z_i'|²`. With centroids `z̄, z̄'` the optimum is

$$a=\frac{\sum_i\overline{(z_i-\bar z)}\,(z_i'-\bar z')}{\sum_i|z_i-\bar z|^2},\qquad b=\bar z'-a\,\bar z.$$

(Derivation: set the derivative with respect to `b` to zero → `b = z̄' − a z̄`; substitute and differentiate with respect to `a`.) This uses all points, so errors of individual points average out: that is why the final shift is accurate to a fraction of a pixel.

## 3. Outliers and RANSAC

Many matched pairs are **wrong** (outliers): LightGlue pairs a window corner with a road corner somewhere else. A least-squares fit to all pairs is ruined by even a few wrong ones.

**RANSAC** (RANdom SAmple Consensus):

1. Draw a random minimal sample (2 pairs).
2. Compute the transform from just those.
3. Count the **inliers**: pairs whose error `|a z + b − z'|` is below a threshold (`RANSAC_PIXELS = 3.0` canvas pixels).
4. Remember the transform with the most inliers; repeat; finally refit on all inliers.

```python
matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC,
    ransacReprojThreshold=RANSAC_PIXELS, maxIters=3000, confidence=0.995)
```

### How many draws are enough?

Let `w` be the fraction of pairs that are correct, `s = 2` the sample size. One draw is all-correct with probability `w^s`. After `k` draws the probability of having had at least one all-correct sample is

$$P=1-(1-w^s)^k\ \Longrightarrow\ k\ge\frac{\ln(1-P)}{\ln(1-w^s)}.$$

For `P = 0.995`:

| fraction correct w | w² | draws k needed |
|---|---|---|
| 0.5 | 0.25 | 18 |
| 0.2 | 0.04 | 130 |
| 0.1 | 0.01 | 527 |

`maxIters = 3000` is far above these numbers, so RANSAC finds the right transform even if 90 % of the pairs are wrong. OpenCV additionally stops early once its own confidence criterion is met.

## 4. Why the inlier count is trustworthy evidence

Suppose the candidate is a *wrong* place. For a given transform, a random pair agrees within 3 pixels with probability about

$$\frac{\pi\cdot3^2}{224^2}\approx5.6\cdot10^{-4}$$

(area of the 3-pixel disc over the tile area; the match is wrong so partners are essentially uniform in the tile). With 300 pairs the expected number of accidental inliers is `λ = 300 · 5.6·10⁻⁴ ≈ 0.17`. The chance of 12 or more is the Poisson tail

$$P(N\ge12)\approx e^{-\lambda}\frac{\lambda^{12}}{12!}\approx10^{-18}.$$

Even with RANSAC trying thousands of transforms, accidents with 12 inliers are astronomically rare. **Caveat:** real mismatches are not uniformly random. A repetitive scene (rows of identical houses) produces *coherent* false matches that do agree on one shift; so the count is strong evidence, not proof. This is why the project's decision rule is tuned on data, not derived from this calculation.

## 5. The sanity gate

```python
if abs(scale - 1) > 0.15 or abs(angle) > 12:
    return None, np.zeros(len(points_a), dtype=bool)
```

Canvas and tile are both north-up at the same pixel size, so the true transform is close to a pure shift (`s ≈ 1`, `θ ≈ 0`). A transform that stretches by more than 15 % or rotates more than 12° found its inliers by fitting a wrong geometry. It is rejected and the candidate gets zero inliers. The thresholds are generous because the geometry (heading, height) carries small errors.

If fewer than 4 pairs exist (`len(points_a) < 4`) no estimate is attempted.

## 6. Inputs, outputs, constraints

`fit_transform(points_a, points_b)`

* `points_a`, `points_b`: float32 arrays `(n, 2)` in canvas pixels, columns `x, y`, row `k` of A matches row `k` of B.
* Returns `(matrix 2×3 float64 or None, inliers bool array (n,))`.
* Determinism: RANSAC uses OpenCV's random generator, seeded per photo by `cv2.setRNGSeed(SEED)` in `localise_photo` (see `result_thresholding/concepts/one_photo_at_a_time.md`).
