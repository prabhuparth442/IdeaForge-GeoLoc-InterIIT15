# Template matching: cosine similarity, its floor, and ZNCC

Code: `src/coarse_search/self_tests.py` (`slide_slowly`, the readable definition) and `src/coarse_search/correlation.py` (the fast version of the same formulas).

---

## 1. The task

After `image_processing/` we have:

* the **query** `Q`, an array `8 × h × w` (h = w = 56 cells) of edge-direction numbers, with a mask `M ∈ {0,1}^{h×w}` saying which cells are real (the canvas has empty corners). `Q` is already zero where `M = 0`. Let `n = Σ M` (about 2000 real cells).
* the **window** `W`, the map's edge cells around the prior, `8 × H × Wd` with `H, Wd` in the thousands.

*Template matching*: put the query on the window at a shift `(i, j)`, score how alike the two are, do it for every shift, and look at the best. The collection of scores over all shifts is the **heat map**.

Write `P_{ij}` for the part of the window under the query: `P_{ij} = W[:, i:i+h, j:j+w] · M` (masked the same way as `Q`).

## 2. Score 1: cosine similarity

Treat `Q` and `P` as long vectors (8·56·56 numbers). Their cosine is

$$\cos(Q,P)=\frac{\langle Q,P\rangle}{\|Q\|\,\|P\|},\qquad\langle Q,P\rangle=\sum_{b,r,c}Q_{b,r,c}P_{b,r,c}.$$

It is the cosine of the angle between the two vectors. By the Cauchy–Schwarz inequality its value lies in `[−1, 1]`; for non-negative data it lies in `[0, 1]`. It equals 1 exactly when `P` is a positive multiple of `Q`. It is blind to overall strength (scaling `P` by 3 changes nothing), which is useful.

### The floor problem

Edge strengths are never negative, so two completely unrelated places still have a large cosine. A quick estimate: let the entries of two independent vectors be non-negative with mean `m` and variance `v`. Then `E[ab] = m²` and `E[a²] = m² + v`, so

$$\cos\approx\frac{m^2}{m^2+v}=\frac{1}{1+\mathrm{CV}^2},\qquad\mathrm{CV}=\frac{\sqrt v}{m}.$$

For entries uniform on `[0,1]`: `m = ½`, `v = 1/12`, `CV² = 1/3`, so `cos ≈ 0.75`. (A numeric check with 100000 random pairs gives 0.749.) The smoother and more clustered the data, the lower the coefficient of variation and the higher the floor: in the toy example below the *median* cosine over all positions is 0.938.

So a correct match with cosine 0.996 sits only 0.06 above the bulk of wrong positions. Noise or one look-alike easily beats it.

## 3. Score 2: ZNCC (zero-mean normalised cross-correlation)

Remove the average of each direction layer from both sides, then take the cosine. For layer `b`:

$$\mu^Q_b=\frac{1}{n}\sum_{r,c}Q_{b,r,c},\qquad Q^0_{b,r,c}=(Q_{b,r,c}-\mu^Q_b)\,M_{r,c}$$
$$\mu^P_b=\frac{1}{n}\sum_{r,c}P_{b,r,c},\qquad P^0_{b,r,c}=(P_{b,r,c}-\mu^P_b)\,M_{r,c}$$

$$\mathrm{ZNCC}=\frac{\langle P^0,Q^0\rangle}{\|P^0\|\,\|Q^0\|}.$$

It is a Pearson correlation coefficient: how much do the *fluctuations around the average* of the two patterns agree? The mask is applied after subtracting so the masked-out cells stay exactly 0 and do not leak into sums. Per-layer means are used (not one global mean) because different directions have different average amounts of edge.

Properties:

* Range `[−1, 1]`. Unrelated patterns give about **0** (the floor has been subtracted), a perfect fit gives 1.
* Invariant to `P → αP + β_b` for `α > 0` and any per-layer offsets `β_b`: it ignores contrast and local brightness-like offsets of the edge maps.
* For independent random fields its standard deviation is roughly `1/√N` with `N` independent values (here ≈ 1/√(n·8)). In real (spatially smooth) data the effective `N` is smaller, so the spread is larger; the toy example shows about ±0.1.

### Measured on the toy example

A toy map (8 layers, 48 × 56, smooth random blobs), a 14 × 14 circular query cut at (20, 31) plus noise (0.15·uniform). Computed by `slide_slowly`:

| score | best position | best value | median over all positions | 5th–95th percentile |
|---|---|---|---|---|
| cosine | (20, 31) ✓ | 0.996 | 0.938 | 0.925 to 0.948 |
| ZNCC | (20, 31) ✓ | 0.934 | −0.012 | −0.142 to 0.118 |

Both find the hidden position; the gap between the peak and the bulk is 0.06 for cosine but 0.95 for ZNCC. On real photos the numbers differ, but the behaviour is the same, which is why the pipeline uses ZNCC (`("zncc", 0.0)`).

## 4. Guards: positions that cannot be trusted

Two situations make both scores meaningless, so the fast implementation sets the score to **−1** ("no answer here"):

1. **No map under the query**: `energy = Σ M·Σ_b P_b² < 0.05·n`. A window beyond the map's edge is all zeros; dividing zero by zero would produce garbage. (`MIN_ENERGY = 0.05` is a first guess.)
2. **Too flat (ZNCC only)**: the local variance `Σ P0² < 0.005·n`. Dividing by a tiny `‖P0‖` blows up noise to large scores.

## 5. The slow reference: `slide_slowly`

```python
for row in range(out_h):
    for col in range(out_w):
        piece = window[:, row:row+kernel_h, col:col+kernel_w] * cells
        cosine[row, col] = (piece*kernel).sum() / (kernel_length*np.sqrt((piece**2).sum()) + 1e-9)
        piece_zero = (piece - piece.sum(axis=(1,2), keepdims=True)/n_valid) * cells
        zncc[row, col] = (piece_zero*kernel_zero).sum() / (kernel_zero_length*np.sqrt((piece_zero**2).sum()) + 1e-9)
```

Two plain loops, one score per position: exactly the definitions above (`1e-9` guards division by zero). It is the **definition** the fast FFT version must reproduce (`run.py --tests` checks the maximum difference is below `1e-4`; typically ~1e-6).

**Cost, and why a loop is impossible for real use.** With a 5 km radius at level 3 the cell is 3.39 m, so the disc has `π·(5000/3.39)² ≈ 6.8 million` positions; each costs `8·56·56 ≈ 25 000` multiplications: about `1.7·10¹¹` operations per photo. Plain loops would take hours; `fft_cross_correlation.md` explains the speed-up.
