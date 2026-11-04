# Keypoint matching: SuperPoint, LightGlue and the MINIMA weights

**Code:** [`../matcher.py`](../matcher.py) (`load_matcher`, `find_keypoints`, `match_keypoints`).
**Status:** this is the one part of the pipeline that depends on external packages (`lightglue`) and a downloaded weights file. It has been written and syntax-checked but **never run in this repo**, because those were not available where it was tested. Read this note for what the code is meant to do, and read the first lines printed by `load_matcher` the first time you run it.

---

## 1. Why we need a second stage at all

The coarse search of the previous section is fast and covers the whole 5 km disc, but it has two limits:

1. **Precision.** It compares 4 × 4-pixel cells, so its answer is only as exact as one cell: 3.4 m at level 3, up to 13.6 m at level 7. The error allowed before points start to drop is about 8 to 10 m.
2. **Look-alikes.** Two places with the same layout (two similar road junctions, two field corners) score almost the same. The wrong one may win.

A proper **image matcher** works differently. It picks out hundreds of individual landmarks in each picture and pairs them up. If the candidate place is right, many pairs agree on one common shift. If it is a look-alike, the pairs scatter randomly. So the *number of agreeing pairs* is both a precise measurement (the shift) and an independent confidence signal. The final answer and the "answer or skip" decision both use it.

## 2. The classical recipe in three words: detect, describe, match

1. **Detect** keypoints: pixels that are easy to find again in another image (corners, junctions, small blobs). A flat patch of grass is a poor keypoint; the corner of a roof is a good one.
2. **Describe** each keypoint by a list of numbers (a **descriptor**) that summarises its neighbourhood, so the same landmark seen in another picture gets a similar list.
3. **Match**: pair keypoints of picture A with keypoints of picture B whose descriptors are close and, ideally, whose preference is mutual (A's point prefers B's point and B's prefers A's).

Old systems used hand-made detectors such as SIFT. Modern ones learn all three steps. We use **SuperPoint** for steps 1 and 2 and **LightGlue** for step 3.

## 3. SuperPoint: finding and describing keypoints

SuperPoint is a single convolutional neural network. For one picture it produces:

* a **keypoint probability map**: for each 8 × 8 block of pixels the network chooses among 65 options (64 pixel positions inside the block, plus one "no keypoint in this block"); the result is turned into a full-resolution map saying "how keypoint-like is each pixel";
* a **256-number descriptor** for every location, computed at one eighth of the resolution, interpolated up to each keypoint's exact position and normalised to length 1.

The settings used in `load_matcher`:

| parameter | value | what it means here |
|---|---|---|
| `max_num_keypoints` | 2048 | keep only the 2048 strongest keypoints |
| `detection_threshold` | 0.0005 | very low: accept weak detections too, because thermal and satellite pictures have low contrast and would otherwise give very few points |
| `nms_radius` | 4 | no two keypoints closer than 4 pixels (non-maximum suppression, the same idea as peak picking in coarse_search) |
| `remove_borders` | 4 | ignore keypoints within 4 pixels of the picture edge |

### Why we enlarge the picture first

```python
big = cv2.resize(picture.astype(np.float32), (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_CUBIC)
tensor = torch.from_numpy(big)[None, None].to(DEVICE)        # shape (batch=1, channel=1, H, W)
return matcher.keypoint_finder.extract(tensor, resize=None)
```

Line by line:

* A canvas is only 224 × 224 pixels. SuperPoint works on 8 × 8 blocks, so that is a mere 28 × 28 blocks, too coarse for many keypoints. We enlarge to `MATCH_SIZE = 2 × CTX = 448` using cubic interpolation. This does not add information, but it gives the network more pixels per landmark to operate on.
* `[None, None]` adds two axes of size 1 in front: a "batch" axis (one picture) and a "channel" axis (one grey channel), because networks expect a 4-D array `(batch, channel, height, width)`.
* `resize=None` tells the library not to resize again.

A guard checks that the picture is square and has values in [0, 1].

## 4. LightGlue: deciding which keypoints belong together

LightGlue is a transformer (the same family as in `coarse_search/concepts/dino_descriptor_hint.md`) that looks at the keypoints of **both** pictures at once. Its input is, for each picture, a list of keypoint positions and descriptors. Over 9 layers it alternates two kinds of attention:

* **self-attention** within one picture: each keypoint learns about its neighbours ("I sit at the corner of a long straight edge");
* **cross-attention** between the pictures: each keypoint in A looks at the keypoints in B and asks "who here resembles me, given what I now know about my surroundings?".

Both use the formula you already met, `softmax(QKᵀ/√d)·V`, plus position information so that geometry matters and not only appearance. At the end two small heads compute:

* a **similarity** for every (A point, B point) pair;
* a **matchability** for every point alone ("does this point even have a partner, or is it hidden/out of view?").

Combined, they form a soft assignment table. A pair is accepted if each point is the other's best choice **and** the score is above `filter_threshold = 0.1`.

**"Light"** means adaptive: if the matcher is already sure after a few layers (`depth_confidence = 0.95`) it stops early; points it considers hopeless (`width_confidence = 0.99`) are dropped from later layers. `flash=False` switches off the FlashAttention kernel, which needs specific hardware.

### Reading the output

```python
scale = picture_size / MATCH_SIZE            # undo the 2x enlargement
points_a = features_a["keypoints"][pairs[:, 0]].cpu().numpy() * scale
points_b = features_b["keypoints"][pairs[:, 1]].cpu().numpy() * scale
```

* `pairs` is an `(n, 2)` table of integers; row `k` says "keypoint `pairs[k,0]` of A goes with keypoint `pairs[k,1]` of B".
* Indexing `keypoints[pairs[:, 0]]` pulls the matching positions out in the same order, so row `k` of `points_a` and row `k` of `points_b` are partners.
* Multiplying by `picture_size / MATCH_SIZE` converts positions from the enlarged 448-pixel picture back to the original 224-pixel canvas coordinates.
* `rbd` ("remove batch dimension") strips the leading size-1 axis from the dictionaries.

## 5. MINIMA: why the weights matter

Normal LightGlue weights are trained on pairs of photos of the same kind (two daytime photos of a building). Our two pictures come from different worlds: thermal infrared and visible-light satellite. Hot roofs are bright in one and may be dark in the other; shadows exist in one and not the other.

**MINIMA** (Modality-INvariant Image MAtching, 2024) retrained the matcher on a very large amount of generated data in which one picture of each pair was converted into another modality (infrared, depth, event, sketch). It learned that "same place" can look completely different, which is exactly our situation. The file `minima_lightglue.pth` (about 50 MB) holds only the LightGlue part; SuperPoint's weights come with the `lightglue` package.

### The layer-name patch

The released file uses older parameter names than the installed library, so `load_matcher` renames them before loading:

```python
for layer in range(9):
    weights = {k.replace("self_attn.%d" % layer, "transformers.%d.self_attn" % layer): v for k, v in weights.items()}
    weights = {k.replace("cross_attn.%d" % layer, "transformers.%d.cross_attn" % layer): v for k, v in weights.items()}
report = glue.load_state_dict(weights, strict=False)
```

`weights` is a dictionary: parameter name to numbers. The loop rewrites, for each of the 9 layers, a name such as `self_attn.3.…` into `transformers.3.self_attn.…`. Then `strict=False` loads whatever matches and **reports** the rest instead of crashing.

The printed line is the safety check:

```
MINIMA weights loaded. File layers not in model: <a> | model layers not in file: <b>
```

`b` must be 0 (or at most a few harmless buffers listed on the next line). If `b` is large, the renaming failed for some layers and those layers silently stay at random values: the matcher would run but be nonsense. Always read this line. The code only prints; if you want a hard stop, add `assert len(report.missing_keys) == 0`.

## 6. Inputs, outputs, requirements

| function | input | output |
|---|---|---|
| `load_matcher()` | nothing; reads `settings.MINIMA_FILE`, downloads from `settings.MINIMA_URL` if missing | `Matcher` holding SuperPoint and LightGlue |
| `find_keypoints(matcher, picture)` | square 2-D float array, values in [0, 1] | dict with `keypoints`, `descriptors`, `keypoint_scores` (tensors, batch size 1) |
| `match_keypoints(matcher, fa, fb, picture_size)` | two feature dicts, side of the original pictures in pixels | `(points_a (n,2), points_b (n,2), scores (n,))`, float32 |

Requirements: PyTorch, the `lightglue` package (`pip install git+https://github.com/cvg/LightGlue.git`), the weights file on disk (**no network at the final run**, so ship it). Time: one SuperPoint pass per picture and one LightGlue pass per candidate. With 25 candidates per photo, this stage dominates the run time; a GPU is strongly recommended.

## 7. Common mistakes

* Ignoring the "model layers not in file" line.
* Feeding values outside [0, 1] (for example 0 to 255); the assert catches it.
* Feeding the canvas and a tile of different pixel sizes (different map levels): the matched points then differ by a scale and the scale gate rejects them.
* Forgetting that point coordinates are in **canvas pixels**, not metres.

## 8. Check yourself

1. Why does the number of agreeing pairs contain more information than the best correlation value?
2. Why are the descriptors normalised to length 1 and why does that make similarity a dot product?
3. What would happen if half of LightGlue's layers kept random weights, and how would you notice?
