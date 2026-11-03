# Which photos we test on, and why every run gives the same numbers

Code: `src/header/sampling.py`, `src/header/data.py`, `random.seed(...)` calls in `run.py`.

---

## 1. Why sample at all

The development set has 500 photos. Evaluating one photo runs the correlation over a 5 km disc plus up to 25 matcher calls, and the run is repeated for many experiments (geometry sweeps, parameter changes). `--n-photos N` therefore tests on a subset.

## 2. Why *stratified*: height decides everything

Photo height controls (a) the zoom level used (3 to 7), (b) the ground size of a pixel, (c) how many landmarks fit in one photo. A random subset of the 500 photos could by chance contain mostly low photos and give a flattering or unfair picture. **Stratified sampling** forces every height group to be represented equally:

| group | heights |
|---|---|
| `100-150 m` | 100 ≤ h < 150 |
| `150-250 m` | 150 ≤ h < 250 |
| `250-350 m` | 250 ≤ h < 351 |

`assign_flights` finds the group of a height `h` with a binary search over the edges `[100, 150, 250, 351]`:

```python
np.searchsorted(BAND_EDGES, h, side="right") - 1
```

`searchsorted(..., side="right")` returns the number of edges `≤ h`. Example: `h = 150` → edges ≤ 150 are `100, 150` → 2 → minus 1 → group index 1 (`150-250 m`). The `−1` converts "how many edges passed" into "which gap I am in". With `side="left"` a photo at exactly 150 m would wrongly fall into group 0.

`pick_sample_ids` then takes `per_group = max(1, N // 3)` photos from each group:

```python
chosen = rng.choice(members, size=min(per_group, len(members)), replace=False)
```

`replace=False` = no photo twice. `min(...)` handles a group smaller than the request. Result `sorted(...)` so photo order is by index, not by group (the order does not matter to correctness because the pipeline is stateless; see `result_thresholding/concepts/one_photo_at_a_time.md`).

**Resulting precision.** With `N = 60`, 20 photos per group: one photo moves a group's share by 5 percentage points. The standard error of a measured proportion `p` from `n` photos is `√(p(1−p)/n)`; for `p = 0.7, n = 20` that is `±10 %`. Only large gaps between experiments mean something.

## 3. Reproducibility: what a "seed" is

Computers produce *pseudo*-random numbers: a deterministic sequence generated from an internal state. Fixing the starting state (the **seed**) makes the entire sequence repeatable. `run.py` seeds:

| call | controls |
|---|---|
| `random.seed(SEED)` | Python's `random` module |
| `np.random.seed(SEED)` | numpy's legacy global generator |
| `torch.manual_seed(SEED)` | PyTorch (CPU and GPU) |
| `np.random.default_rng(seed)` inside functions | a *separate* generator object; unaffected by global seeds, good practice |
| `cv2.setRNGSeed(SEED)` inside `localise_photo` | OpenCV's generator, used by RANSAC |

`np.random.default_rng(SEED)` creates a generator that you hold in a variable and pass around. Two such generators with different seeds never disturb each other, which is why `TestSet` uses `default_rng(0)` for priors and `pick_sample_ids` uses `default_rng(7)` for the sample: changing one does not change the other.

## 4. Metadata to photo records (`data.py`)

`make_items(table)` turns each CSV row into a plain dict (`name, lat, lon, rel_height, yaw, pitch, roll`). Plain dicts were kept instead of classes so a record can be printed and inspected at a glance.

**Angle wrapping.** Pitch and roll are stored as 0…360 in the file but must be small signed tilts:

$$\text{wrap}(a)=\big((a+180)\bmod 360\big)-180\in[-180,180).$$

Derivation: adding 180 moves the target interval `[−180, 180)` to `[0, 360)`; `mod 360` forces any angle into it; subtracting 180 moves back. Example: `350 → (530 mod 360) − 180 = 170 − 180 = −10`. A drone pitched 10° down is not "350°" for the geometry formulas.

**Camera constants.** `Camera.__init__` checks, for each of the 9 camera columns, that max − min over all rows is below 1e-6. The same camera took all photos, so the numbers must agree; an assertion stops the run early if a different dataset has per-photo calibrations (which this code would silently ignore otherwise).

## 5. Constraints

* The CSV must contain: `filename, lat_gtruth, lon_gtruth, alt, yaw, pitch, roll, cam_fx, cam_fy, cam_cx, cam_cy, cam_k1, cam_k2, cam_k3, cam_p1, cam_p2`.
* `lat_gtruth/lon_gtruth` are only used by `evaluation/` (as the truth and to simulate the prior). In a real flight the system receives a prior position instead; nothing in `image_processing/`, `coarse_search/`, `fine_matching/`, `result_thresholding/` reads them.
* `rel_height` is `alt` as given in the CSV (height above ground; the sweep in `evaluation/geometry_sweep.py` can test `alt − 4`).
