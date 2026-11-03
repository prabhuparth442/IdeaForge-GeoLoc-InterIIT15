# One photo at a time: statelessness, seeding and timing

Code: `src/result_thresholding/pipeline.py` (`localise_photo`).

---

## 1. The contract

`localise_photo(item, prior_m, camera, store, matcher=None)` takes **one** photo and returns **one** result dictionary. It uses only: the photo's own metadata, the prior (rough area), the camera model and the prebuilt map store. It does not read other photos, previous results or the ground truth. That is the contract of the contest: the unit is a single image, and ranking or thresholding cannot use the rest of the test set.

Practical consequences:

* the function can be run in any order, in parallel, or repeated for one photo with the same output;
* anything tuned on a dataset (settings, threshold) must be fixed *before* the test photo arrives.

## 2. The steps

```python
level = level_for_height(item["rel_height"], camera)                     # which zoom level
canvas, valid = render_query(item, camera, level)                        # north-up picture of the photo
found, _ = search_frame(canvas, valid, level, prior_m, store, variants=[("zncc", 0.0)])
peaks = found[("zncc", 0.0)]                                             # 25 candidates
ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)                             # first / second score
stage2 = second_stage(matcher, canvas, valid, level, peaks, store["levels"])
x, y, inliers, chosen = final_answer(peaks, stage2)
```

Each line points to its concept file: height → level (`image_processing/concepts/map_zoom_levels.md`), canvas (`rotations_and_the_north_up_canvas.md`), search (`coarse_search/concepts/*`), check (`fine_matching/concepts/*`), decision (`choosing_the_answer.md`).

`matcher=None` skips the fine stage: the answer is the top correlation peak and `answer` is `False` (inliers = 0). That is the "correlation only" mode of `python run.py --no-minima`.

`variants=[("zncc", 0.0)]` selects the single scoring variant used in the final system (ZNCC, no DINO hint). The search function can compute several variants at once for experiments (e.g. `("zncc", 0.5)` with the DINO prior); the production call asks for one.

## 3. Determinism

```python
cv2.setRNGSeed(SEED)
```

RANSAC picks random point pairs, so two runs could give different inlier counts and so different answers for borderline photos. Seeding OpenCV's generator at the start of each photo (`SEED = 7`) makes the result reproducible, independent of how many photos were run before. Note the seed is reset per photo, not once, for exactly the statelessness reason above. Torch inference (SuperPoint, LightGlue, DINO) has no randomness in inference mode; GPU float results can differ in the last bits between machines, which can change a match at the margin: expect equal but not bit-identical results across hardware.

## 4. Timing and memory

```python
if DEVICE == "cuda":
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
...  seconds = time.time() - clock
```

CUDA calls return before the GPU has finished. Without `synchronize()` the clock stops too early and the time measured is wrong (too small). `empty_cache()` hands cached GPU memory back so that long runs do not grow. The returned `seconds` is the wall-clock time per photo; the contest has a time budget, so check the report column.

## 5. Output dictionary

| key | type | meaning |
|---|---|---|
| `x`, `y` | float, metres | answer, map coordinates (x east, y south) |
| `answer` | bool | `inliers >= ANSWER_MIN_INLIERS` |
| `inliers` | int | inliers of the chosen candidate (0 when no matcher) |
| `chosen` | int | index into `peaks` of the chosen candidate (0 = coarse winner) |
| `ratio` | float | `peaks[0,0] / peaks[1,0]`, a coarse-stage confidence (higher = more distinct) |
| `peaks`, `stage2` | arrays/dict | everything needed for diagnostics (the report uses them) |
| `seconds` | float | time for this photo |

## 6. Running it by hand

```python
PYTHONPATH=src python - <<'P'
from header.settings import *
# build items, camera, store as run.py does, then:
# result = localise_photo(items[0], prior_m, camera, store, matcher)
P
```
`run.py` shows the exact set-up (`make_items`, `build_map_store`, `load_matcher`); copy those lines.
