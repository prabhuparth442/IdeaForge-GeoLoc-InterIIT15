# Template matching: how do we score "does the photo fit here?"

**Code:** [`../self_tests.py`](../self_tests.py) (`slide_slowly`, the slow readable version) and [`../correlation.py`](../correlation.py) (the fast version of the same formulas).
**You need first:** [`edge_direction_features.md`](../../image_processing/concepts/edge_direction_features.md), because this note starts where that one ends.

---

## 1. Where we are in the story

After `image_processing/` we hold two arrays that describe edges in the same language:

* the **query** `Q`, the photo's fingerprint: 8 layers (one per edge direction) of 56 × 56 cells. The canvas has empty corners (where the tilted photo does not reach), so there is also a **mask** `M` of 56 × 56 ones and zeros: 1 = this cell has real data. About 2000 of the 3136 cells are real. `Q` is already 0 where `M` is 0.
* the **window** `W`, the same kind of fingerprint of the *map*, cut around the place where we think the drone is. It is thousands of cells wide.

The idea of **template matching** is as simple as it sounds:

> Lay the query on top of the window at one position, compute a single number saying "how alike are the two?", slide the query one cell, compute again, and so on for every position. Write all the numbers into a grid. That grid is the **heat map**. Its brightest spot is our best guess for where the photo was taken.

The whole question of this note is: **what is the single number?** The obvious choice (cosine) turns out to be a poor one, and the better choice (ZNCC) takes a short walk to understand.

## 2. Notation, kept small

* `P` = the part of the window currently under the query (same shape as `Q`, masked by `M`).
* `n` = number of real cells in the mask (about 2000).
* We treat `Q` and `P` as long lists of numbers (8 × 56 × 56 of them). Their **dot product** is "multiply the lists entry by entry and add everything up": `Q·P`.
* Their **length** is `‖Q‖ = √(Q·Q)`.

## 3. First idea: cosine similarity

$$\cos(Q,P)=\frac{Q\cdot P}{\|Q\|\,\|P\|}$$

Picture two arrows from the origin. The cosine is large when they point in nearly the same direction, and it ignores how long they are. In words: "does `P` have the same *shape* as `Q`, regardless of how strong the edges are?" Ignoring strength is useful, because a bright and a dim photo of the same street have the same edge layout.

### A tiny example you can do by hand

Use one layer and four cells so the arithmetic fits on a line. The query is `Q = [1, 3, 1, 3]`. Here are five candidate windows:

| window `P` | what it is | cosine |
|---|---|---|
| `[1, 3, 1, 3]` | identical | 1.000 |
| `[2, 6, 2, 6]` | same pattern, twice as strong | 1.000 |
| `[3, 1, 3, 1]` | pattern **reversed** (high where Q is low) | **0.600** |
| `[2, 2, 2, 2]` | flat, no pattern at all | **0.894** |
| `[5, 5, 5, 6]` | almost flat, a bit of noise | **0.913** |

Look at rows 3 to 5. A reversed pattern, a flat patch and noise all score 0.6 to 0.9. The cosine only sinks to 0.6 for the exact opposite of the pattern. **The score never gets anywhere near 0.**

### Why: edge strengths are never negative

Every number in `Q` and `P` is 0 or more (an edge is either there or not; there is no "negative edge"). Two lists of non-negative numbers always have a positive dot product, so the arrows always point roughly the same way. A rough estimate of the floor: if the entries have average `m` and spread `v` (variance), two unrelated lists have `Q·P ≈ m²` per entry and `Q·Q ≈ m² + v`, so

$$\cos\approx\frac{m^2}{m^2+v}.$$

For numbers spread evenly between 0 and 1 this is about 0.75 (a numeric check with 100 000 random pairs gives 0.749). Real edge maps are smoother and more clustered than that, so the floor is higher. In the toy test used in this repo (below), the **median** cosine over all positions is 0.938.

Consequence: a correct match scoring 0.996 sits only 0.06 above the crowd of wrong positions. A bit of noise, or one look-alike street corner, can easily beat it. We want a score where "no relationship" means about **0**, so that a good match stands far above the crowd.

## 4. Second idea: ZNCC (zero-mean normalised cross-correlation)

The name is a mouthful; the recipe is two steps.

1. **Zero-mean:** in each of the 8 layers, subtract that layer's average from every cell (so every layer now averages 0).
2. Take the **cosine** of the two corrected lists.

Formally, for layer `b`:

$$\mu^Q_b=\frac1n\sum_{\text{real cells}}Q_{b},\qquad Q^0_b=(Q_b-\mu^Q_b)\cdot M,$$

and the same for `P` with its own averages `μᴾ_b`. Then

$$\mathrm{ZNCC}=\frac{P^0\cdot Q^0}{\|P^0\|\,\|Q^0\|}.$$

The mask `M` is multiplied in *after* subtracting, so the empty cells go back to exactly 0 and do not pollute the sums (subtracting a mean from an empty cell would otherwise give it a non-zero value).

### What the subtraction does, on the same toy numbers

`Q = [1, 3, 1, 3]` has average 2, so `Q⁰ = [−1, +1, −1, +1]`: "low, high, low, high". Now every candidate is judged on whether it goes *up and down at the same places*:

| window `P` | `P⁰` (after subtracting its own average) | ZNCC |
|---|---|---|
| `[1, 3, 1, 3]` | `[−1, +1, −1, +1]` | **+1.000** |
| `[2, 6, 2, 6]` | `[−2, +2, −2, +2]` | **+1.000** (strength ignored, as before) |
| `[3, 1, 3, 1]` | `[+1, −1, +1, −1]` | **−1.000** (opposite pattern, now clearly bad) |
| `[2, 2, 2, 2]` | `[0, 0, 0, 0]` | undefined (0/0), see the guard in §6 |
| `[5, 5, 5, 6]` | `[−.25, −.25, −.25, +.75]` | **+0.577** |

Two things changed. The reversed pattern is now −1 instead of +0.6. And the flat patch has *no ups and downs at all*, so there is nothing to compare; ZNCC refuses to give a number (more on that in §6).

ZNCC is the **Pearson correlation coefficient**, the same statistic used to ask "do these two series move together?". It has three properties we rely on:

* range −1 to +1, and **unrelated patterns give about 0**;
* it ignores contrast: `P → α·P + β` (stronger, or shifted up) gives the same score, so a hazy photo and a crisp map still compare fairly;
* the subtraction is per layer, not one global average, because different directions have different amounts of edge (in a city, horizontal and vertical edges dominate; diagonals are rare).

### How big is "about 0"?

For two random patterns of `N` independent numbers, the score typically lands within about `±1/√N`. Here `N` is around 8 × 2000 = 16 000, so `±0.008` if the cells were independent. They are not: neighbouring cells are strongly related, so the *effective* `N` is far smaller and the spread is bigger. That is why the toy example below shows a spread of about ±0.1. Real photos behave similarly.

## 5. Measured on the toy example used by the repo's self-test

A toy map (8 layers, 48 × 56 cells of smooth random blobs); a 14 × 14 circular query cut out at position (20, 31) and spoiled with noise. Both scores computed at every position by `slide_slowly`:

| score | best position | best value | median over all positions | 5th to 95th percentile |
|---|---|---|---|---|
| cosine | (20, 31) correct | 0.996 | 0.938 | 0.925 to 0.948 |
| ZNCC | (20, 31) correct | 0.934 | −0.012 | −0.142 to 0.118 |

Both find the hidden place. But look at the **gap between the winner and the crowd**: for cosine 0.996 versus 0.938 (a gap of 0.06); for ZNCC 0.934 versus about 0 (a gap of 0.93). When the picture is noisy or the scene is repetitive, a big gap is what keeps the winner the winner. That is why the pipeline uses `("zncc", 0.0)`.

## 6. Guards: positions where the formula cannot be trusted

Both formulas divide by lengths. If a length is near zero the division is meaningless, so the fast code writes **−1** ("no answer here") in two situations:

1. **No map under the query.** Window cells outside the map are zeros. If the part of the window under the query has almost no edge energy (`Σ P² < 0.05·n`), the score is −1. (0.05 is a first guess, `settings.MIN_ENERGY`.)
2. **Flat patch (ZNCC only).** The window part has too little variation to correlate against (`variance < 0.005·n`, `settings.MIN_VARIANCE`). Dividing by a tiny `‖P⁰‖` would blow the noise up into a huge score; the toy `[2, 2, 2, 2]` above is the extreme case.

Thresholds are set *per cell* so they behave the same however many cells of the query are real. Later code reads −1 as "unusable" and never lets such a position become a peak.

## 7. The slow reference: `slide_slowly`

The slow version in `self_tests.py` is just the definition written as two loops:

```python
for row in range(out_h):
    for col in range(out_w):
        piece = window[:, row:row+kernel_h, col:col+kernel_w] * cells      # window part under the query
        cosine[row, col] = (piece*kernel).sum() / (kernel_length*np.sqrt((piece**2).sum()) + 1e-9)
        piece_zero = (piece - piece.sum(axis=(1,2), keepdims=True)/n_valid) * cells
        zncc[row, col] = (piece_zero*kernel_zero).sum() / (kernel_zero_length*np.sqrt((piece_zero**2).sum()) + 1e-9)
```

Read it line by line:

* `piece = window[...] * cells` cuts the window part under the query and zeroes the cells the query does not use.
* The cosine line is dot product over (length × length). The `1e-9` only prevents 0/0.
* `piece_zero` subtracts each layer's average (`piece.sum(axis=(1,2))/n_valid` gives 8 averages) and re-applies the mask.
* The ZNCC line is the same cosine on the corrected lists.

This is the **definition**. It is slow, and that is the point: its only job is to be obviously right so the fast version can be checked against it.

### Why a loop is impossible for real use

At map level 3 one map *pixel* is 0.849 m and one cell is 4 pixels = 3.39 m. A 5 km radius disc therefore holds about π·(5000/3.39)² ≈ **6.8 million positions**. Each costs 8·56·56 ≈ 25 000 multiplications for the dot product alone. That is around 1.7 × 10¹¹ operations for one photo, hours with plain loops. [`fft_cross_correlation.md`](fft_cross_correlation.md) explains how the same numbers are obtained in seconds.

## 8. Common mistakes

* **Using cosine and believing 0.95 means "found it".** On this kind of data 0.94 is the crowd.
* **Subtracting one global average instead of one per layer.** The per-layer version is what removes each direction's own floor.
* **Forgetting to re-apply the mask after subtracting the average.** Empty cells would then carry the value `−μ` and add fake terms to every sum.
* **Reading the −1 as a real score.** It is a flag, not a measurement.

## 9. Check yourself

1. For `Q = [2, 4, 2, 4]` and `P = [10, 30, 10, 30]`, what are the cosine and the ZNCC? (Cosine: 0.99. ZNCC: 1.0.)
2. Why does a flat window have a high cosine but undefined ZNCC?
3. Which of the two scores would you trust to say "this place is clearly better than the rest"? Why?
