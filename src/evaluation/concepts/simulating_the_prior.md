# Simulating the prior: a random start inside a 5 km disc

Code: `random_prior`, `truth_in_metres`, `TestSet` in `src/evaluation/testset.py`.

---

## 1. Why simulate

At contest time each photo comes with a rough position (the "prior"): the drone is somewhere within `DISC_RADIUS = 5000` m of it. Our development photos have exact GPS, so we **invent** a prior for each: the true position shifted by a random vector. The test then asks "starting from only the disc, do we find the truth?".

Requirement: the truth must be uniformly placed *inside* the disc as seen from the prior. Equivalent, by symmetry, to the prior being uniform in a disc around the truth.

## 2. Sampling a uniform point in a disc: why `R·sqrt(u)`

Picking `distance = R·u` with `u ~ U(0,1)` is wrong: it crowds points near the centre, because the ring at distance `d` has circumference `2πd`, which grows with `d`.

For a uniform disc the probability that the distance is at most `d` is the area ratio:

$$F(d)=\frac{\pi d^2}{\pi R^2}=\frac{d^2}{R^2}.$$

Inverse-transform sampling: draw `u ~ U(0,1)`, set `F(d)=u`:

$$d=R\sqrt u.$$

Angle is uniform on `[0, 2π)`. The code:

```python
distance = radius * math.sqrt(rng.random())
angle = 2 * math.pi * rng.random()
return truth + distance * np.array([math.cos(angle), math.sin(angle)])
```

Check: the median distance is `R·sqrt(0.5) = 3536 m` for `R = 5000`, not 2500.

## 3. Ground truth in map metres

```python
row, col = satellite_map.gps_to_map_pixel(item["lat"], item["lon"])
return np.array([col * satellite_map.gsd, row * satellite_map.gsd])
```

Same convention as the whole pipeline (x east = column × gsd, y south = row × gsd). Geographic conversion is in `header/concepts/geographic_coordinates_and_map_reading.md`. Error between an answer and the truth is the plain Euclidean `math.hypot` in metres (`TestSet.error_m`), valid because the map is a metric (projected) grid over a few km.

## 4. Reproducibility

`prior_rng = np.random.default_rng(0)`: the same priors every run, so two configurations are compared on identical problems. Change the seed (not the radius) to check that a conclusion does not depend on lucky priors.

## 5. What `TestSet` stores (and how to use it)

| attribute | shape | meaning |
|---|---|---|
| `items` | list | photo metadata dicts |
| `truths` | (N,2) | true (x,y) in map metres |
| `priors` | (N,2) | simulated priors |
| `sigmas` | (N,) | σ in metres per photo (`header.scoring.sigma_m`) |
| `query_levels` | (N,) | zoom level used for each photo |
| `city` | (N,) bool | height ≤ 200 m |
| `sample_ids`, `sample_city`, `sample_sigma` | | the stratified subset actually run |

Constraint: `flight_of_frame` and `sample_ids` come from `header/data.py` and `header/sampling.py`; `satellite_map` must be the map of the same area as the photos.
