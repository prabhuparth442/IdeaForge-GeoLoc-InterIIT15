# One photo at a time: statelessness, seeding and timing

**Code:** [`../pipeline.py`](../pipeline.py) (`localise_photo`).
**You need first:** the earlier sections, because this function simply calls them in order.

---

## 1. The contract

`localise_photo(item, prior_m, camera, store, matcher=None)` takes **one** photo and returns **one** result dictionary. It uses only:

* the photo's own metadata (`item`: file name, height, yaw, pitch, roll),
* the prior `prior_m` (the rough area),
* the camera model,
* the prebuilt map `store`.

It does **not** read other photos, earlier results or the ground truth. That is the contest's rule: the unit of work is a single image, so nothing about the rest of the test set may leak into the decision.

What follows from this:

* the function can be called in any order, in parallel, or repeated for the same photo with the same output;
* every setting and threshold tuned on a development set must be **fixed before** the test photo arrives.

## 2. The steps, one per line

```python
level = level_for_height(item["rel_height"], camera)                     # 1. which map zoom level
canvas, valid = render_query(item, camera, level)                        # 2. north-up picture of the photo
found, _ = search_frame(canvas, valid, level, prior_m, store, variants=[("zncc", 0.0)])   # 3. coarse search
peaks = found[("zncc", 0.0)]                                             # 4. the 25 candidates
ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)                             # 5. winner / runner-up
stage2 = second_stage(matcher, canvas, valid, level, peaks, store["levels"])   # 6. keypoint check
x, y, inliers, chosen = final_answer(peaks, stage2)                      # 7. choose
```

| line | what happens | where to read about it |
|---|---|---|
| 1 | photo height decides the level | `image_processing/concepts/map_zoom_levels.md` |
| 2 | geometry: undo tilt, lens, heading | `image_processing/concepts/rotations_and_the_north_up_canvas.md` |
| 3–5 | edge fingerprint, FFT correlation, peaks | `coarse_search/concepts/` |
| 6 | SuperPoint + LightGlue + RANSAC for each candidate | `fine_matching/concepts/` |
| 7 | pick the winner | [`choosing_the_answer.md`](choosing_the_answer.md) |

Finally `decide(inliers)` turns the winner's inlier count into the yes/no `answer`.

### Without the matcher

If `matcher=None` the fine stage is skipped: the answer is the top correlation peak, `inliers = 0` and so `answer = False`. This is the "correlation only" mode of `python run.py --no-minima`. It lets you test the first half of the pipeline on a computer without `lightglue`.

### Why one variant

`variants=[("zncc", 0.0)]` asks for a single scoring variant: ZNCC, no DINO nudge. The search function can compute several at once (it shares the expensive FFT), which `evaluation` uses for experiments, but the production path uses just one.

## 3. Determinism: why we reseed per photo

```python
cv2.setRNGSeed(SEED)
```

RANSAC draws random pairs of points. Without a fixed seed, two runs on the same photo could give different inlier counts, and so occasionally different answers for borderline photos. We set OpenCV's random generator to `SEED = 7` at the **start of every photo**. Two reasons it is per photo, not once at program start:

1. the result of photo 17 then does not depend on how many photos ran before it (statelessness);
2. `python run.py --independence` can run photos in forward and reverse order and assert that every answer is the same.

What is *not* controlled: the neural networks (SuperPoint, LightGlue, DINO) have no randomness at inference, but floating-point results on a GPU can differ in the last bits between machines, which can flip a borderline match. Expect equal answers across machines for nearly all photos, not bit-for-bit identical numbers.

## 4. Timing and memory

```python
if DEVICE == "cuda":
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
...  seconds = time.time() - clock
```

GPU calls are **asynchronous**: Python asks the GPU to start and gets control back immediately, long before the GPU is done. If we stopped the clock then, we would measure only the time to *issue* the work. `synchronize()` waits until everything has really finished, so `seconds` is honest. `empty_cache()` hands cached GPU memory back so that a long run does not creep towards an out-of-memory error. The contest has a time budget, so look at the `seconds` column of the report.

## 5. The result dictionary

| key | type | meaning |
|---|---|---|
| `x`, `y` | float, metres | the answer in map coordinates (x east, y south) |
| `answer` | bool | `inliers ≥ ANSWER_MIN_INLIERS` |
| `inliers` | int | inliers of the chosen candidate (0 without matcher) |
| `chosen` | int | index into `peaks` of the chosen candidate (0 = coarse winner) |
| `ratio` | float | `peaks[0,0]/peaks[1,0]`, a coarse confidence signal |
| `peaks`, `stage2` | array, dict | everything needed for diagnostics (the report uses them) |
| `seconds` | float | wall-clock time for this photo |

## 6. Inputs and checks

`item` from `make_items` (with `rel_height > 0`), `prior_m` two finite numbers in the map's metre frame, `camera` a `Camera`, `store` from `build_map_store`, `matcher` from `load_matcher` or `None`. All asserted at the top of the function.

## 7. Running it by hand

`run.py` shows the exact set-up (`make_items`, `build_map_store`, `load_matcher`). In pseudo-code:

```python
items, camera = ...             # as run.py does
store = build_map_store(satellite_map)
matcher = load_matcher()
result = localise_photo(items[0], prior_m, camera, store, matcher)
print(result["x"], result["y"], result["answer"], result["inliers"], result["seconds"])
```

`python run.py --n-photos 5 --independence` calls it per photo in both orders and asserts equal answers.

## 8. Common mistakes

* Reading results from a previous photo inside the function (breaks the contest rule).
* Calling `time.time()` on a GPU run without `synchronize()`.
* Forgetting that the converted latitude/longitude still needs the inverse map transform, which is not written yet.

## 9. Check yourself

1. Why must the RNG seed be set inside `localise_photo` and not once in `run.py`?
2. What is the output when `matcher=None`, and why is `answer` False?
3. Name two things that must be decided before the test photos are seen.
