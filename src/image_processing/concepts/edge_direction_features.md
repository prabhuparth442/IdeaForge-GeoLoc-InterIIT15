# Edge-direction cells: what we actually compare

**Code:** `src/image_processing/edge_features.py`.

**What you need to know beforehand:** what a derivative is, informally (how fast something changes), and what an angle is. Nothing else; the rest is built here.

---

## 1. The big picture in plain words

A thermal photo records heat. A satellite photo records reflected sunlight. A road can be warm and bright in the thermal image and dark in the satellite image, or the other way round after sunset. Comparing brightness directly does not work.

What both pictures share is the **geometry of the scene**: where roads, roofs and field boundaries run. More precisely, the *direction* in which these boundaries run. A north–south road is a vertical line in both pictures, whether it is bright or dark.

So we turn each picture into a map of "which way do the edges point, and how strong are they", in small blocks. That map is the same for both sensors, even though their brightness values are not.

```
picture ──blur──► gradient (gx, gy) ──► strength + direction ──► 8 direction bins ──► 4×4 cell sums ──► soft normalisation
```

## 2. Step 1: the gradient (how fast brightness changes)

For a picture `I(x, y)` the gradient `∇I = (∂I/∂x, ∂I/∂y)` points in the direction in which brightness increases fastest, and its length tells how steep the change is. At an edge the gradient is **perpendicular to the edge line**: for a vertical road the brightness changes left to right, so the gradient points sideways.

On a pixel grid we approximate the derivatives with **central differences**: right neighbour minus left neighbour.

$$g_x[i,j]=I[i,j+1]-I[i,j-1],\qquad g_y[i,j]=I[i+1,j]-I[i-1,j].$$

(Dividing by 2 would give the exact derivative; we skip it because the later normalisation removes any constant factor.) The one-pixel border stays 0:

```python
gx[:, 1:-1] = blurred[:, 2:] - blurred[:, :-2]
gy[1:-1, :] = blurred[2:, :] - blurred[:-2, :]
```

Before differentiating, the picture is blurred with a Gaussian of `σ = 1` pixel. Differentiating amplifies noise (it measures *changes*, and noise is a change), so we smooth first. Real edges, which are several pixels wide, survive; single-pixel noise does not.

## 3. Step 2: strength and direction

$$s=\sqrt{g_x^2+g_y^2},\qquad \alpha=\operatorname{atan2}(g_y,\,g_x)\bmod \pi.$$

* `s` is the **edge strength**. Flat regions have `s ≈ 0`.
* `α` is the direction of the gradient. The `mod π` identifies `α` with `α + π`: a light-to-dark edge and a dark-to-light edge count as the same. The next paragraph shows why this is exactly what we need.

### Why brightness differences do not matter (a proof)

Suppose the other sensor sees `I' = a·I + b`: a gain `a ≠ 0` and an offset `b`. The offset cancels in every difference, so `g_x' = a·g_x` and `g_y' = a·g_y`. Therefore

$$s'=|a|\,s,\qquad \alpha'=\operatorname{atan2}(a\,g_y,\ a\,g_x)=\begin{cases}\alpha & a>0\\ \alpha+\pi & a<0\end{cases}\ \equiv\ \alpha\pmod\pi .$$

The direction does not change, even for `a < 0` (contrast reversal, e.g. a thermal inversion). The strength is scaled by `|a|`, which the per-cell normalisation in Section 6 removes. Brightness differences that are not of the form `aI + b` (nonlinear ones) are not covered exactly, but monotonic ones come close.

**Check with the real function.** A vertical edge (dark left, light right) gives gradient angle 0° and bin 0. The same edge with reversed contrast gives bin 0 as well. The same edge with gain 3 and offset 5 gives bin 0, with the strength multiplied by 3. A horizontal edge gives angle 90° and bin 4.

## 4. Step 3: eight direction bins

The angle range `[0, π)` (that is 0° to 180°) is split into 8 bins of 22.5°:

```python
bin_id = np.minimum((angle / np.pi * N_BINS).astype(int), N_BINS - 1)
```

* `angle / π · 8` maps `[0, π)` onto `[0, 8)`.
* `astype(int)` rounds down: the bin number.
* `np.minimum(…, 7)` guards the single value `α = π` that floating-point rounding can produce.

| bin | gradient angle | edge looks like |
|---|---|---|
| 0 | 0° – 22.5° | vertical edge (gradient points sideways) |
| 2 | 45° – 67.5° | diagonal edge |
| 4 | 90° – 112.5° | horizontal edge (gradient points up/down) |
| 6 | 135° – 157.5° | the other diagonal |

Limitation: the bins are hard and do not overlap. An edge at 22.4° and one at 22.6° fall into different bins although they are almost parallel. An angle exactly on a boundary (like a perfect 45° diagonal) lands in either neighbouring bin depending on floating-point rounding; in a test it fell in bin 1 even though 45°/22.5° is exactly 2. The 4 × 4 cell sums in the next step soften this, but it is a known weakness (soft binning would be a possible improvement).

## 5. Step 4: cells of 4 × 4 pixels

For each bin `b` we add up the edge strengths of the pixels in that bin, block by block:

```python
weight = (strength * (bin_id == b))[:rows*CELL, :cols*CELL]
features[b] = weight.reshape(rows, CELL, cols, CELL).sum(axis=(1, 3))
```

* `strength * (bin_id == b)` keeps the strength only where the pixel's direction is `b`.
* `reshape(rows, 4, cols, 4)` cuts the picture into 4 × 4 blocks without copying data; `sum(axis=(1, 3))` adds up the two within-block axes.

A 224 × 224 canvas becomes `8 × 56 × 56` numbers: for every 4 × 4 block, a little 8-bar histogram of "how much edge in each direction".

Why use cells instead of single pixels?

1. **Tolerance.** If the geometry is off by up to 4 pixels (it never is perfect), an edge moves inside its cell but the cell's content barely changes.
2. **Cost.** 16 times fewer positions to test in the correlation.
3. **Robustness.** Summing many pixels averages out noise.

The price: a position is only resolved to one cell (3.4 to 13.6 m depending on the level). The fine stage (`fine_matching/`) recovers the part below one cell.

## 6. Step 5: normalising every cell ("soft" normalisation)

Let `v` be one cell's 8 numbers, `‖v‖ = √(Σ v_b²)` its length, and `t` a *typical* strength: the 30th percentile of `‖v‖` over the valid cells. Then

$$\tilde v=\frac{v}{\|v\|+t+10^{-3}}.$$

What this does:

* **Strong cell** (`‖v‖ ≫ t`): `‖ṽ‖ = ‖v‖/(‖v‖+t) → 1`. Only the *mix of directions* matters; the strength is saturated. This removes sensor gain.
* **Weak cell** (`‖v‖ ≪ t`): `ṽ ≈ v/t`, small. Noise-like cells contribute little.
* **Empty cell:** `ṽ ≈ 0`.

So `‖ṽ‖ = ‖v‖/(‖v‖+t) ∈ [0, 1)`, a smooth "soft threshold". The `+10⁻³` avoids division by zero for completely flat blocks.

**Worked example** (three cells; bin 0 has the strengths 40, 4, 0.1, and the first cell also has 30 in bin 1). The lengths are 50, 4 and 0.1, and the 30th percentile `t` is 2.44.

| cell | length ‖v‖ | ‖ṽ‖ = length/(length + t) |
|---|---|---|
| strong | 50 | 0.953 |
| medium | 4 | 0.621 |
| faint | 0.1 | 0.039 |

The strong cell is almost saturated, the faint cell almost vanishes, and a huge gain in the sensor (all values × 100) would change the strong cell's result only slightly.

```python
def normalise_cells(features, cells_to_use=None):
    length = np.sqrt((features ** 2).sum(axis=0))
    ...
    typical = np.percentile(length[cells_to_use], 30)
    return features / (length + typical + 1e-3)
```

## 7. The query side (the photo's canvas)

```python
def query_edge_features(canvas, valid):
    features = edge_features(canvas)
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8))
    cell_valid = safe.reshape(side, 4, side, 4).mean(axis=(1, 3)) >= 0.8
    features = normalise_cells(features, cell_valid) * cell_valid
    return features, cell_valid
```

* The canvas has empty corners (filled with the average grey). Their borders create fake edges. `erode` with a 7 × 7 box shrinks the valid area by 3 pixels on each side to discard the contaminated pixels.
* A cell is valid only if at least 80 % of its pixels survived.
* Invalid cells are set to 0 after normalising, and `typical` is computed from the valid cells only.

**Output:** `features` with shape `(8, 56, 56)` and `cell_valid` with shape `(56, 56)` (bool).

## 8. The map side

The map levels get the same recipe (blur, gradient, 8 bins, 4 × 4 cells, normalise), so query and map are made of the same material. Differences caused by the size:

* **Strips.** A level picture is processed 1024 rows at a time with 8 extra rows above and below. The blur and the gradient reach about 4 pixels, so 8 are enough; the padding rows are thrown away afterwards. There are no seams at the strip joins.
* **No-data borders.** Where the map ends (value 0) next to real pixels there is a huge fake step edge. `no_data_cells` marks every 4 × 4 cell that contains a 0 pixel, and a 3 × 3 `cv2.dilate` marks the neighbours as well. Their features are set to 0.
* **One `typical` for the whole level:** the 30th percentile of the non-zero cell lengths, sampled every 50th cell (`[::50]`) to save time. A window cut anywhere from the level is then on the same scale.

```python
border = no_data_cells(image)[:features.shape[1], :features.shape[2]]
features[:, border] = 0
length = np.sqrt((features ** 2).sum(axis=0))
typical = np.percentile(length[length > 0][::50], 30)
return features / (length + typical + 1e-3)
```

## 9. Inputs, outputs, constraints

| function | input | output |
|---|---|---|
| `gradient_parts(image)` | 2-D float picture, at least 3 × 3 | blurred picture, `gx`, `gy`, strength, angle, bin id |
| `edge_features(image)` | 2-D picture, at least 4 × 4 | `(8, H//4, W//4)` float32 |
| `normalise_cells(features, cells_to_use)` | `(8, rows, cols)` and optionally a `(rows, cols)` bool mask | normalised features |
| `query_edge_features(canvas, valid)` | square canvas, side a multiple of 4, bool mask | `(8, 56, 56)` features and `(56, 56)` mask |
| `no_data_cells(image)` | 2-D uint8 level picture | bool cell mask |
| `map_edge_features(image)` | 2-D uint8 level picture | `(8, H//4, W//4)` normalised |
| `build_map_features(level_images)` | dictionary `level → uint8 picture` | dictionary `level → features` |

## 10. Mistakes to avoid

* Skipping the blur (noise then dominates the gradient directions).
* Treating angles over the full 360° (this would make contrast-reversed pictures differ).
* Mixing normalised and unnormalised features between the query and the map.
* Forgetting that zero cells mean "no information", not "no edges".

## 11. Where this leads

Both sides are now non-negative arrays of shape `8 × h × w`. Finding where the query fits in the map is a **template matching** problem. Because every entry is non-negative, plain cosine similarity has a high floor and ZNCC is needed: see `coarse_search/concepts/template_matching_cosine_and_zncc.md`.

## 12. Check yourself

1. A vertical road has a gradient pointing which way, and which bin? *(Sideways, 0°: bin 0.)*
2. Why `mod π` and not a full circle? *(Dark-to-light and light-to-dark must count as the same edge.)*
3. A cell has length 50 and `t = 2.44`. What is its length after normalising? *(0.953.)*
4. Why are 8 padding rows enough in the strip processing? *(Blur and gradient reach about 4 pixels.)*
