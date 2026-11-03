# Correlating everywhere at once with the FFT

Code: `src/coarse_search/correlation.py` (`fast_length`, `correlate_everywhere`).

---

## 1. Cross-correlation is a sum of products at every shift

For one layer, window `W[r, c]` (size `H × Wd`) and kernel `K[u, v]` (size `h × w`):

$$C[i,j]=\sum_{u=0}^{h-1}\sum_{v=0}^{w-1}W[i+u,\,j+v]\;K[u,v],\qquad 0\le i\le H-h,\;0\le j\le Wd-w.$$

That is exactly "sum of map × query" at shift `(i,j)`, the numerator of both scores. Directly it costs `h·w` multiplications per output value.

## 2. The convolution theorem

The 2-D discrete Fourier transform of a signal `x` of size `N × M` is `X[p,q] = Σ x[r,c] e^{-2πi(pr/N + qc/M)}`. Two facts:

1. A **shift** of the signal multiplies its transform by a phase: `x[r+a, c+b] ↔ X[p,q]·e^{+2πi(pa/N+qb/M)}`.
2. Therefore the transform of a **circular cross-correlation** is

$$\mathcal F\{C\}=\mathcal F\{W\}\cdot\overline{\mathcal F\{K\}}$$

(element-wise product with the complex conjugate of the kernel's transform). Proof sketch: write `C[i] = Σ_u W[i+u]K[u]`, take the DFT over `i`, substitute `i' = i+u`, and the double sum factorises into `Ŵ[p]·conj(K̂[p])`.

So: transform both arrays (`O(N log N)`), multiply, transform back. The Fast Fourier Transform computes a transform of `N` points in about `N log₂N` operations instead of `N²`.

Speed-up: direct = `(H·Wd)·(h·w)` multiplications; FFT ≈ `(H·Wd)·log₂(H·Wd)·const`. With `h·w = 3136` and `log₂(9·10⁶) ≈ 23`, the FFT saves roughly one to two orders of magnitude per layer (the exact factor depends on constants), and it runs well on a GPU.

## 3. Circular vs linear, and the padding

The DFT assumes the signal repeats forever (periodic). Correlation computed that way is **circular**: when the kernel hangs off the right edge of the window, its right part wraps around to the left. We want only positions where the kernel lies fully inside the window.

Claim: if both arrays are zero-padded to a common length `N ≥ H`, the circular correlation equals the linear one for all outputs `i ≤ H − h`.

Proof: `C_circ[i] = Σ_{u<h} W[(i+u) mod N]·K[u]`. For `i ≤ H − h` and `u ≤ h − 1`, the index `i+u ≤ H − 1 < N`, so `mod N` does nothing: no wrap-around term enters. ∎

Hence `torch.fft.rfft2(window, s=size)` (zero-padding to `size`) and cutting the result `[:out_h, :out_w]` with `out_h = H − h + 1` is correct, and everything beyond is thrown away.

`rfft2` is the transform for *real* inputs: the transform of a real signal is conjugate-symmetric, so only half is stored (about half the memory and time).

### Why `fast_length`

FFT algorithms are fastest for lengths made only of small prime factors. `fast_length(n)` returns the smallest `m ≥ n` whose only prime factors are 2, 3 and 5:

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

It divides out 2s, 3s, 5s; if nothing is left, the size is good, else try `n+1`. Examples: `2999 → 3000`, `1021 → 1024`, `3001 → 3072`.

## 4. The pieces of the two scores

Notation (as in `template_matching_cosine_and_zncc.md`): layers `b = 1..8`, query `Q`, mask `M`, `n = ΣM`, window `W`. All correlation sums below are over the query footprint, i.e. correlations with `M` or `Q`.

**Dot product (both scores' numerator, cosine).**

$$
\operatorname{dot}[i,j]
=
\sum_b C[W_b,Q_b][i,j]
=
\mathcal{F}^{-1}\left(
\sum_b \widehat{W}_b \cdot \overline{\widehat{Q}_b}
\right)
$$

Because the transform is linear, we sum over the 8 layers *in the frequency domain* and invert only once:

```python
dot = torch.fft.irfft2((window_fft * kernel_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
```

**Energy (cosine denominator).** `‖P_{ij}‖² = Σ_b Σ_{valid cells} W_b[i+u, j+v]²`. That is a correlation of the layer-summed squares with the mask:

```python
strength_fft = torch.fft.rfft2((window ** 2).sum(dim=0), s=size)
energy = torch.fft.irfft2(strength_fft * cells_fft.conj(), s=size)[:out_h, :out_w].clamp_min(0)
cosine = dot / (kernel.pow(2).sum().sqrt() * energy.sqrt() + 1e-6)
```

`clamp_min(0)` removes tiny negative values created by floating-point rounding before the square root.

**ZNCC numerator.** With `Q^0_b = (Q_b − μ^Q_b)M`:

$$\langle P^0,Q^0\rangle=\sum_b\sum_{valid}(P_b-\mu^P_b)\,Q^0_b=\sum_b\sum_{valid}P_b\,Q^0_b-\sum_b\mu^P_b\underbrace{\sum_{valid}Q^0_b}_{=0}=\langle P,Q^0\rangle.$$

The second term vanishes because `Q^0` sums to zero over the valid cells in each layer *by construction*. So we do **not** need to subtract the map's local mean for the numerator; a plain correlation with the zero-mean kernel suffices (`dot_zero`).

**ZNCC denominator.** `‖P^0‖² = Σ_b Σ_valid (P_b − μ_b)²`. Expand with `μ_b = (Σ_valid P_b)/n`:

$$\sum_{valid}(P_b-\mu_b)^2=\sum_{valid}P_b^2-2\mu_b\sum_{valid}P_b+n\mu_b^2=\sum_{valid}P_b^2-\frac{\big(\sum_{valid}P_b\big)^2}{n}.$$

Summing over layers: `variance = energy − Σ_b (local_sum_b)²/n`, where `local_sum_b[i,j] = Σ_valid W_b[i+u, j+v]` is a correlation of layer `b` with the mask. The code loops over the 8 layers:

```python
squared_sums = torch.zeros_like(energy)
for b in range(n_bins):
    local_sum = torch.fft.irfft2(window_fft[b] * cells_fft.conj(), s=size)[:out_h, :out_w]
    squared_sums += local_sum ** 2 / n_cells
variance = (energy - squared_sums).clamp_min(0)
zncc = dot_zero / (kernel_zero.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)
```

Count of transforms: forward `rfft2` of the 8-layer window stack, of the mask, of the 8-layer kernel stack, of the layer-summed squares and of the 8-layer zero-mean kernel stack (five calls); inverse `irfft2` for `dot`, `energy`, `dot_zero` and the 8 `local_sum` layers (eleven). On a GPU the whole thing takes a fraction of a second; on a CPU several seconds at level 3.

## 5. Guards (after the maths)

```python
no_map   = energy   < min_energy   * n_cells
too_flat = variance < min_variance * n_cells
cosine[no_map] = -1.0
zncc[no_map | too_flat] = -1.0
```

Thresholds are per query cell (`0.05` and `0.005`) so they do not depend on how many cells are valid. The `-1` is a sentinel; `pick_peaks` treats `score > -1` as "usable".

## 6. Inputs, outputs, constraints

`correlate_everywhere(window, kernel, cells, min_energy, min_variance)`

| argument | shape / type | meaning |
|---|---|---|
| `window` | torch float32 `(8, H, Wd)` | map cells around the prior (zeros outside the map) |
| `kernel` | torch float32 `(8, h, w)` | query cells (already zero where invalid) |
| `cells` | torch float32 `(h, w)` | 1 where the query cell is real, else 0 |

Returns `{"cosine": (H−h+1, Wd−w+1), "zncc": same}` torch tensors on the same device.

Constraints: `H ≥ h`, `Wd ≥ w`. Memory: the window for a 5 km radius at level 3 is about 3000 × 3000 × 8 floats = 288 MB, plus complex spectra several times that; on a small GPU use fewer levels or process level 3 on the CPU. Everything must live on one device (`settings.DEVICE`).

## 7. Verifying it

`python run.py --tests` runs `fast_equals_slow()`: a 48 × 56 toy window, a hidden 14 × 14 circular query, the loop version and the FFT version side by side; the largest difference must be below `1e-4` (observed ≈ 1e-6) and both must find the hidden position.
