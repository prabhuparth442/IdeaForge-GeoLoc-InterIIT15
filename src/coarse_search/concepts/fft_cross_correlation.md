# Scoring every position at once with the FFT

**Code:** [`../correlation.py`](../correlation.py) (`fast_length`, `correlate_everywhere`).
**You need first:** [`template_matching_cosine_and_zncc.md`](template_matching_cosine_and_zncc.md) (what the two scores are).

---

## 1. The problem in one sentence

We have to compute the same score at about **6.8 million** positions (level 3), and each score needs about 25 000 multiplications. Doing that with two loops takes hours. The **Fast Fourier Transform (FFT)** gives exactly the same numbers in seconds, and this note walks through why that is possible and how the code uses it.

You do not need to know what a Fourier transform "is" to follow the logic. Only one fact is needed, the *convolution theorem*, and we build up to it with a tiny example.

## 2. What exactly has to be computed: sums of products at every shift

Take one layer, one dimension, to see the pattern. Window `W`, kernel (the query) `K`:

$$C[i]=\sum_{u}W[i+u]\,K[u].$$

"Put the kernel at shift `i`, multiply entry by entry, add up." Real life is the same with two dimensions (`W[i+u, j+v]·K[u, v]`) and 8 layers whose results are added.

### A tiny example, by hand

```
W = [1, 2, 5, 3, 1, 0, 2]      (window, 7 numbers)
K = [5, 3]                      (kernel, 2 numbers)
```

| shift i | W under K | products | C[i] |
|---|---|---|---|
| 0 | 1, 2 | 1·5 + 2·3 | 11 |
| 1 | 2, 5 | 2·5 + 5·3 | 25 |
| 2 | 5, 3 | 5·5 + 3·3 | **34** |
| 3 | 3, 1 | 3·5 + 1·3 | 18 |
| 4 | 1, 0 | 1·5 + 0·3 | 5 |
| 5 | 0, 2 | 0·5 + 2·3 | 6 |

The biggest value is at shift 2, where the kernel `[5, 3]` sits on `[5, 3]`. This is a "how well does the kernel fit here" curve. This `C` is the *numerator* of both our scores (the dot product); the rest of the score only needs a few more sums of the same kind (§5).

Cost: for each of the (window length) shifts we do (kernel length) multiplications. For 2-D with a 3000 × 3000 window and a 56 × 56 kernel that is 9 million × 3136 ≈ 2.8 × 10¹⁰ per layer.

## 3. The shortcut: the convolution theorem

A **Fourier transform** rewrites a list of numbers as a list of "how much of each wave frequency is in it". Call the transform of `W` by `Ŵ` and of `K` by `K̂`. The theorem we need says:

> The sums-of-products curve `C` is obtained by transforming `W` and `K`, multiplying the two transforms **entry by entry** (with a small twist: use the *complex conjugate* of `K̂`), and transforming back.

$$C=\mathcal F^{-1}\big\{\hat W\cdot\overline{\hat K}\big\}.$$

Why this is believable, in two sentences: shifting a signal only multiplies its transform by a simple phase (a rotation in the complex plane); and a sum over all shifts of "signal times shifted signal" turns, in the frequency world, into plain multiplication. The conjugate on `K̂` is what makes it *correlation* ("slide and compare") rather than *convolution* ("slide a flipped copy"): it saves us flipping the kernel.

The **FFT** is a clever algorithm that computes a transform of `N` numbers in about `N·log₂N` steps instead of `N²`. So the plan "transform both, multiply, transform back" costs about three FFTs rather than `N × (kernel size)` multiplications.

Rough size for one layer, 3000 × 3000 window: `N = 9 × 10⁶`, `log₂N ≈ 23`, so about 2 × 10⁸ steps per FFT, times a small constant, against 2.8 × 10¹⁰ for the direct method. That is between one and two orders of magnitude, and FFT also runs very well on a GPU.

### The same example through the FFT

Running the 7-number example through `numpy.fft` (transform length 7):

```
direct sums:              [11, 25, 34, 18, 5, 6]
FFT result (first 6):     [11, 25, 34, 18, 5, 6]     identical
FFT result (all 7):       [11, 25, 34, 18, 5, 6, 13]
```

The first six match. There is a **seventh number, 13**, which has no counterpart in the table. That leads to the one thing you must understand about FFT correlation.

## 4. The "wrap-around" trap and why we only keep part of the output

The FFT secretly treats every list as **repeating forever** (periodic). So when the kernel hangs over the right end of the window, the part sticking out re-enters from the left. Position 6 in the example is "kernel at the last window entry plus the first window entry":

$$C[6]=W[6]\cdot5+W[0]\cdot3=2\cdot5+1\cdot3=13.$$

It is a legitimate number for a ring, but meaningless for a straight window: the real kernel does not straddle the end and the beginning.

**The fix is bookkeeping, not extra calculation.** For a window of length `H` and a kernel of length `h`, the shifts `i = 0 … H−h` have the kernel fully inside, and for those no wrapped term enters the sum. In the example `H = 7`, `h = 2`, so only `i = 0 … 5` are kept. The seventh value is discarded. That is exactly what the code does:

```python
out_h = height - kernel_h + 1       # number of valid shifts
dot = torch.fft.irfft2(...)[:out_h, :out_w]       # keep only those
```

A short proof that nothing wrapped in what we keep: the sum for shift `i` uses window indices `i + u` with `u ≤ h−1`; if `i ≤ H−h` the index never exceeds `H−1`, so "mod N" never does anything. ∎ (The FFT length `N` is at least `H`, padded with zeros, which is the `s=size` argument.)

Consequence for the output size: a window of `H × W` cells and a kernel of `h × w` give `(H−h+1) × (W−w+1)` valid positions.

### `rfft2` and `irfft2`

The map data is real numbers. The transform of a real signal has a mirror symmetry (the second half is a conjugate copy of the first), so `rfft2` stores only half. It is the same maths with half the memory and time. `irfft2` is the matching inverse.

### Why `fast_length`

FFT programs are fastest when the length factorises into small primes. Lengths like 3001 (prime) can be several times slower than 3072 (= 2¹⁰·3). `fast_length(n)` therefore returns the smallest number ≥ n whose only prime factors are 2, 3 and 5:

```python
def fast_length(n):
    while True:
        rest = n
        for factor in (2, 3, 5):
            while rest % factor == 0:
                rest = rest // factor
        if rest == 1:
            return n
        n = n + 1
```

Walk through `n = 2999`: divide out 2s, 3s and 5s from 2999: nothing divides, `rest = 2999 ≠ 1`, try 3000. 3000 = 2³·3·5³, everything divides out, `rest = 1`, return 3000. Verified values: `2999 → 3000`, `1021 → 1024`, `3001 → 3072`, `3003 → 3072`, `793 → 800`.

The extra zeros are harmless: padding goes at the end, and we keep only the first `H−h+1` outputs.

## 5. Assembling the two scores from FFT pieces

Notation as before: layers `b = 1…8`, query `Q`, mask `M`, `n = ΣM`, window `W`. All the sums run over the query's footprint (the real cells).

$$\mathrm{dot}[i,j] = \sum_b C[W_b,Q_b][i,j] = \mathcal{F}^{-1}(\sum_b \hat{W}_b \overline{\hat{Q}_b})$$

### 5.1 Dot product (the cosine's numerator)

For each layer, correlate `W_b` with `Q_b`; then add the 8 layers. Because the transform is *linear*, adding in the frequency world gives the same thing as adding after, and lets us invert **once** instead of eight times:

```python
dot = torch.fft.irfft2((window_fft * kernel_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
```

Read inside out: `window_fft` and `kernel_fft` have shape `(8, ·, ·)`; the product is entry by entry; `.sum(dim=0)` adds the 8 layers; `irfft2` goes back; `[:out_h, :out_w]` keeps only valid shifts.

### 5.2 Energy (the denominators' common part)

`‖P‖²` is "the sum of squares of all the window numbers under the query's real cells". That is itself a correlation: of the layer-summed squares with the mask.

```python
strength_fft = torch.fft.rfft2((window ** 2).sum(dim=0), s=size)
energy = torch.fft.irfft2(strength_fft * cells_fft.conj(), s=size)[:out_h, :out_w].clamp_min(0)
cosine = dot / (kernel.pow(2).sum().sqrt() * energy.sqrt() + 1e-6)
```

`clamp_min(0)` removes tiny negative values that rounding errors produce before the square root. `+ 1e-6` stops 0/0.

### 5.3 ZNCC numerator, with a nice simplification

The ZNCC numerator is `Σ (P_b − μᴾ_b) · Q⁰_b`. Expand the bracket:

$$\sum P_b\,Q^0_b\;-\;\mu^P_b\sum Q^0_b .$$

The second term is zero, because `Q⁰` was built to have average zero in every layer (`ΣQ⁰_b = 0`). So **we do not need the window's average at all in the numerator**: a plain correlation of the window with the zero-mean kernel does the job.

```python
kernel_mean = kernel.sum(dim=(1, 2), keepdim=True) / n_cells
kernel_zero = (kernel - kernel_mean) * cells
dot_zero = torch.fft.irfft2((window_fft * zero_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
```

### 5.4 ZNCC denominator: variance without ever forming the means

We need `Σ (P_b − μ)²` over real cells, where `μ = (ΣP_b)/n`. Expand:

$$\sum(P_b-\mu)^2=\sum P_b^2-2\mu\sum P_b+n\mu^2=\sum P_b^2-\frac{(\sum P_b)^2}{n}.$$

Summed over the 8 layers: `variance = energy − Σ_b (local_sum_b)² / n`, where `local_sum_b[i, j] = Σ over real cells of W_b` is one more correlation, with the mask.

```python
squared_sums = torch.zeros_like(energy)
for b in range(n_bins):
    local_sum = torch.fft.irfft2(window_fft[b] * cells_fft.conj(), s=size)[:out_h, :out_w]
    squared_sums += local_sum ** 2 / n_cells
variance = (energy - squared_sums).clamp_min(0)
zncc = dot_zero / (kernel_zero.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)
```

All the heavy work is FFT calls; the 8-iteration loop is only 8 cheap extra inverse transforms.

### Transform count

Forward `rfft2`: window stack, mask, kernel stack, layer-summed squares, zero-mean kernel stack (5 calls). Inverse `irfft2`: `dot`, `energy`, `dot_zero`, plus 8 `local_sum` layers (11 calls). Sixteen big transforms per photo. Measured here on a CPU: **2.65 s at level 7** (793 × 793 cells); level 3 is about 14 times bigger, so expect tens of seconds on a CPU and far less on a GPU (neither measured on the real map yet).

## 6. Guards (last step)

```python
no_map   = energy   < min_energy   * n_cells
too_flat = variance < min_variance * n_cells
cosine[no_map] = -1.0
zncc[no_map | too_flat] = -1.0
```

Same reasons as in the template-matching note: no map under the query, or nothing to correlate against. Thresholds are per query cell.

## 7. A full-size check you can run

```bash
PYTHONPATH=src python -c "
import torch
from coarse_search.correlation import correlate_everywhere
w = torch.rand(8, 60, 60); k = w[:, 20:30, 25:35].clone(); c = torch.ones(10, 10)
z = correlate_everywhere(w, k, c)['zncc']
print(z.shape, divmod(int(z.argmax()), z.shape[1]), float(z.max()))"
# torch.Size([51, 51]) (20, 25) 1.0
```

A random window, a piece cut out of it at row 20, column 25, and the highest ZNCC lands exactly there with value 1.0. The output is 51 × 51 because 60 − 10 + 1 = 51. A larger run (793 × 793 window, 56 × 56 piece cut at (100, 200)) gives a 738 × 738 map, peak (100, 200) with value 1.0, **median cosine 0.75 and median ZNCC 0.00**, which shows the floor effect from the previous note on random data.

`python run.py --tests` additionally compares the FFT code with the slow loops on a toy problem: the largest difference must stay below 1e-4 (observed about 1e-6).

## 8. Inputs, outputs, constraints

| argument | shape | meaning |
|---|---|---|
| `window` | torch float32 `(8, H, W)` | map cells around the prior (zeros outside the map) |
| `kernel` | torch float32 `(8, h, w)` | query cells (zero where invalid) |
| `cells` | torch float32 `(h, w)` | 1 where the query cell is real |

Returns `{"cosine": (H−h+1, W−w+1), "zncc": same}`, values in [−1, 1] with −1 meaning "unusable". Requirements, all enforced by `assert`: `h ≤ H`, `w ≤ W`; same number of layers; same device; at least one real cell.

Memory is the practical limit: the level-3 window is about 3003 × 3003 × 8 floats = 288 MB, and the complex spectra are several times larger. If a small GPU runs out, run level 3 on the CPU (`settings.DEVICE`).

## 9. Common mistakes

* Taking the full output instead of cropping to the valid part (the wrapped numbers look plausible and are wrong near the far edges).
* Forgetting the conjugate (you then get a flipped-kernel convolution, whose peak sits in the wrong place).
* Padding to a size smaller than the window.
* Mixing devices (kernel on GPU, window on CPU).

## 10. Check yourself

1. In the 7-number example, what would the 7th output mean if the window really were a ring?
2. For a window of 100 cells and a kernel of 30, how many valid shifts are there? (71.)
3. Why do we sum the 8 layers before the inverse transform rather than after?
