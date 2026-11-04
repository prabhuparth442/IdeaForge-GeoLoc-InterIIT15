# Which photos we test on, and why every run gives the same numbers

**Code:** `src/header/sampling.py`, `src/header/data.py`, and the `seed` calls in `run.py`.

**What you need to know beforehand:** what a random number is. Nothing else.

---

## 1. The big picture in plain words

Running the whole pipeline on a photo takes a while, and we repeat the run many times (for every experiment and every setting we try). So we often test on a **subset** of the 500 development photos (`--n-photos 60`). Two questions follow:

1. *Which* photos do we pick, so that the result is fair?
2. How do we make sure two runs pick the same photos, so that a difference between two experiments is caused by the experiment and not by luck?

The answers are **stratified sampling** (question 1) and **seeds** (question 2).

## 2. Why not just pick randomly?

Photo height controls almost everything: which zoom level is used (3 to 7), how many metres one pixel covers, how many landmarks fit in the picture. Suppose 500 photos contain 200 low, 150 medium and 150 high ones. A purely random subset of 60 might by chance contain 35 low photos and only 10 high ones. A system that is good at low photos and poor at high ones would look better than it is, and the next random subset would give a different answer.

**Stratified sampling** removes this luck: split the photos into groups (strata) by height and take the **same number from every group**.

| group | heights |
|---|---|
| `100-150 m` | 100 ≤ h < 150 |
| `150-250 m` | 150 ≤ h < 250 |
| `250-350 m` | 250 ≤ h < 351 |

(The upper edge 351 only exists so that a photo at exactly 350 m is included.)

### Finding the group of a height

```python
np.searchsorted(BAND_EDGES, h, side="right") - 1        # BAND_EDGES = [100, 150, 250, 351]
```

`searchsorted(edges, h, side="right")` counts **how many edges are ≤ h**. Then `− 1` turns "number of edges passed" into "index of the gap I am in". Try it:

| h | edges ≤ h | −1 | group |
|---|---|---|---|
| 100 | 1 (100) | 0 | `100-150 m` |
| 149.9 | 1 | 0 | `100-150 m` |
| 150 | 2 (100, 150) | 1 | `150-250 m` |
| 250 | 3 | 2 | `250-350 m` |
| 350 | 3 | 2 | `250-350 m` |
| 351 | 4 | 3 | outside: the assert stops the run |

With `side="left"` a photo at exactly 150 m would count only the edge below it and wrongly fall into the first group, so `"right"` is the right choice for "lower edge included".

### Taking the same number from each group

```python
per_group = max(1, n_photos // len(settings.FLIGHT_NAMES))       # e.g. 60 // 3 = 20
chosen = rng.choice(members, size=min(per_group, len(members)), replace=False)
```

* `replace=False` means no photo is picked twice.
* `min(per_group, len(members))` handles a group with fewer photos than requested (you then simply get all of them). Note this also happens with the default `N_PHOTOS = 500`: `500 // 3 = 166` per group is more than some groups hold, so the "sample" is then (nearly) the whole set.
* The final `sorted(...)` puts the ids in photo order. The order does not matter for correctness, because each photo is processed independently (`result_thresholding/concepts/one_photo_at_a_time.md`).

### How precise is a result from N photos?

If a fraction `p` of photos is solved, then measured on `n` photos the typical error (standard error) is

$$\text{SE}=\sqrt{\frac{p(1-p)}{n}}.$$

| p | n | SE |
|---|---|---|
| 0.7 | 20 (one group of a 60-photo run) | ±10 % |
| 0.7 | 60 (all three groups) | ±6 % |

So a difference of 3 percentage points between two experiments on 60 photos is **noise**. Only large gaps mean something. This is also why the repo's test-time numbers must be taken with care.

## 3. Reproducibility: what a "seed" really is

A computer cannot produce truly random numbers; it produces a long fixed list of numbers that *look* random, produced by a formula from an internal **state**. The **seed** is the starting value of that state. Same seed → exactly the same list, every time, on every machine.

```python
rng = np.random.default_rng(7)
rng.choice(10, 3, replace=False)          # [7 5 6]   (and again [7 5 6] next time you run it)
```

`run.py` seeds every random-number source the pipeline touches:

| call | controls |
|---|---|
| `random.seed(SEED)` | Python's `random` module |
| `np.random.seed(SEED)` | numpy's older global generator |
| `torch.manual_seed(SEED)` | PyTorch, on CPU and GPU |
| `np.random.default_rng(seed)` inside functions | a **separate** generator object that you create, hold and pass around |
| `cv2.setRNGSeed(SEED)` in `localise_photo` | OpenCV, used by RANSAC (`fine_matching/concepts/similarity_transform_and_ransac.md`) |

Why use separate generator objects (`default_rng`) instead of the global one? Two different generators never disturb each other. In this project `TestSet` makes its simulated priors with `default_rng(0)` and `pick_sample_ids` picks photos with `default_rng(7)`. Changing the photo sample does not change the priors, and vice versa. If both used the same global generator, changing one thing would silently change the other.

## 4. From CSV rows to photo records (`data.py`)

### `make_items`

Each CSV row becomes a plain Python dictionary with the keys `name, lat, lon, rel_height, yaw, pitch, roll`. Plain dictionaries (instead of classes) were chosen so that any record can be printed and read at a glance.

### Wrapping angles

Pitch and roll arrive as 0…360 but should be small signed tilts. The fix:

$$\operatorname{wrap}(a)=\big((a+180)\bmod 360\big)-180\ \in[-180,180).$$

Why it works, step by step: adding 180 moves the target interval `[−180, 180)` to `[0, 360)`; `mod 360` forces any angle into that interval; subtracting 180 moves it back.

| input a | a + 180 | mod 360 | − 180 |
|---|---|---|---|
| 350 | 530 | 170 | **−10** |
| 10 | 190 | 190 | 10 |
| 190 | 370 | 10 | −170 |
| 0 | 180 | 180 | 0 |

A drone nose-down by 10° must be −10, not 350, or the tilt formulas in `image_processing/geometry.py` would rotate the picture almost a full turn.

### `Camera`

`Camera.__init__` reads the 9 camera numbers (focal lengths, centre, distortion) and **checks** that they are identical across all rows (max − min < 1e-6). All photos came from the same camera, so the numbers must agree. If a different dataset ever has per-photo calibrations, this check stops the run immediately instead of silently using the first row's numbers for all photos.

## 5. Constraints

* The CSV must contain: `filename, lat_gtruth, lon_gtruth, alt, yaw, pitch, roll, cam_fx, cam_fy, cam_cx, cam_cy, cam_k1, cam_k2, cam_k3, cam_p1, cam_p2`. A missing column stops the run with a message naming it.
* All heights must lie in `[100, 351)`; otherwise `assign_flights` asserts.
* `lat_gtruth` and `lon_gtruth` are read only by `evaluation/` (as the truth and to simulate the prior). On a real flight the system receives a prior position instead. Nothing in `image_processing/`, `coarse_search/`, `fine_matching/` or `result_thresholding/` reads the truth.
* `rel_height` is the CSV's `alt` as given.

## 6. Mistakes to avoid

* Comparing two experiments on *different* random samples. Always keep the same seed.
* Reading too much into a 3-point difference on 60 photos.
* Using the global numpy generator inside new functions; take a seed and make your own `default_rng(seed)`.

## 7. Check yourself

1. Which group is a photo at 250.0 m in? *(`250-350 m`)*
2. Why do we pass `replace=False`? *(So a photo cannot be tested twice.)*
3. Why is `350 // 3`-style rounding harmless here? *(Each group gets the same `per_group`; a few photos may be left out, equally from each group.)*
4. Why is angle 350° converted to −10°? *(Pitch and roll are small signed tilts.)*
