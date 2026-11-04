# Simulating the prior: a random start inside a 5 km disc

**Code:** [`../testset.py`](../testset.py) (`random_prior`, `truth_in_metres`, `TestSet`).
**You need first:** `header/concepts/geographic_coordinates_and_map_reading.md` (the map metre frame) and `header/concepts/stratified_sampling_and_seeds.md` (what a seed is).

---

## 1. Why we have to invent a prior

At contest time every photo comes with a rough position, the **prior**: the drone is somewhere within `DISC_RADIUS = 5000` m of it. Our 500 development photos have exact GPS, so there is no real prior. To test the system honestly we **invent one for each photo**: the true position moved by a random vector. Then we ask: "knowing only this rough position, do we still find the truth?"

What the invented prior must satisfy: seen from the prior, the truth should be **uniformly spread inside the disc**. By symmetry that is the same as saying: the prior is a uniformly random point in a disc of radius 5 km around the truth. Not "uniform in distance", but uniform in *area*. The next section shows why this matters.

## 2. The trap: `R·u` is not uniform in a disc

The tempting recipe is "pick a distance `R·u` with `u` random between 0 and 1, and a random angle". This crowds points near the centre. Picture the disc as a target with rings: the ring at distance `d` has circumference `2πd`, so rings far from the centre have much more room. If you give every distance the same chance, each unit of area near the centre gets far more points than each unit near the edge.

Quantify it with the median (half the points closer, half further):

* Naive `R·u`: median distance **2500 m**, and a quarter of the points are within 1250 m.
* Correct (uniform disc): median distance **R·√0.5 ≈ 3536 m**.

A naive prior would make the test **too easy**, because the typical prior would be close to the truth.

## 3. The right recipe: `R·√u`

The probability that a uniformly placed point lies within distance `d` of the centre is the area ratio:

$$F(d)=\frac{\pi d^2}{\pi R^2}=\frac{d^2}{R^2}.$$

To generate a random `d` with this distribution (inverse-transform sampling): draw `u` uniformly between 0 and 1, then solve `F(d) = u` for `d`:

$$\frac{d^2}{R^2}=u\ \Longrightarrow\ d=R\sqrt u.$$

The angle is uniform on [0, 2π). The code is exactly that:

```python
distance = radius * math.sqrt(rng.random())
angle = 2 * math.pi * rng.random()
return truth + distance * np.array([math.cos(angle), math.sin(angle)])
```

Line by line: `rng.random()` is a number in [0, 1); `math.sqrt` of it times the radius is the distance; the angle is a full circle times another random number; `cos` and `sin` turn "distance and angle" into an east-south step (x east, y south as everywhere in the repo); adding it to the truth gives the prior.

### Check it with numbers

Drawing 20 000 priors around the origin (this is the script from the README) gives a median distance of **3531 m** (theory 3536 m), with the quartiles at 2515 m and 4332 m (theory `R√0.25 = 2500` and `R√0.75 = 4330`). Only 24.7 % fall within 2500 m, in line with the area ratio `(2500/5000)² = 25 %`.

```bash
PYTHONPATH=src python -c "
import numpy as np
from evaluation.testset import random_prior
rng = np.random.default_rng(0)
d = [np.hypot(*random_prior(np.zeros(2), rng)) for _ in range(20000)]
print(np.median(d))"       # about 3531
```

## 4. The ground truth in map metres

```python
row, col = satellite_map.gps_to_map_pixel(item["lat"], item["lon"])
return np.array([col * satellite_map.gsd, row * satellite_map.gsd])
```

The photo's GPS (latitude, longitude) is converted to a map pixel by the map's own transform, then to metres: `x = col × gsd` (east), `y = row × gsd` (south). This is the frame in which the prior, the search window, the candidates and the answer all live.

The error between an answer `(x, y)` and the truth is a plain straight-line distance:

```python
def error_m(self, i, x, y):
    return math.hypot(x - self.truths[i][0], y - self.truths[i][1])
```

A straight line is correct here because the map is a **projected** (metre) grid and the area is a few kilometres wide, so curvature is negligible.

## 5. Reproducibility

```python
prior_rng = np.random.default_rng(0)
```

The generator is seeded with 0, so every run draws the same priors. That allows an honest comparison: two configurations are tested on **identical problems**. If you want to check that a conclusion does not depend on lucky priors, change the seed (not the radius) and repeat.

## 6. What `TestSet` stores

| attribute | shape | meaning |
|---|---|---|
| `items` | list | photo metadata dictionaries |
| `truths` | (N, 2) | true (x, y) in map metres |
| `priors` | (N, 2) | simulated priors |
| `sigmas` | (N,) | σ in metres per photo (`header.scoring.sigma_m`) |
| `query_levels` | (N,) | zoom level used for each photo |
| `city` | (N,) bool | height ≤ 200 m ("low" photos) |
| `sample_ids`, `sample_city`, `sample_sigma` | | the stratified subset actually run |

Constraints (asserted): at least one photo, one flight label per photo, `sample_ids` valid indices. `flight_of_frame` and `sample_ids` come from `header/data.py` and `header/sampling.py`; the satellite map must be the map of the same area.

## 7. Common mistakes

* Using `R·u` (too easy a test).
* Mixing frames: the truth in lat/lon and the prior in metres.
* Changing the prior seed between two configurations you want to compare.
* Forgetting that the actual contest prior is **given**, not generated: this simulation only exists so that we can test.

## 8. Check yourself

1. With `R = 2000` m, what is the median distance for a uniform disc? (`2000·√0.5 ≈ 1414` m.)
2. What fraction of uniform-disc priors fall within `R/2` of the truth? (25 %.)
3. Why does the same prior list across runs matter when comparing two settings?
