# Keypoint matching: SuperPoint, LightGlue and the MINIMA weights

Code: `src/fine_matching/matcher.py` (`load_matcher`, `find_keypoints`, `match_keypoints`).

---

## 1. Why a second stage

The coarse correlation compares edge layouts cell by cell. Two limits:

1. **Precision:** it is only as exact as one cell (3.4 to 13.6 m) while the allowed error σ is 8 to 10 m.
2. **Look-alikes:** two places with the same layout score nearly the same; the wrong one can win.

A proper image matcher looks at hundreds of individual landmarks. If the candidate place is right, many landmark pairs agree on one common shift; if it is a look-alike, the pairs scatter. The *count of agreeing pairs* is also an independent confidence signal (the ranking and the answer/skip decision use it).

## 2. The classical recipe: detect, describe, match

1. **Detect** keypoints: pixels that are easy to find again (corners, junctions, blobs).
2. **Describe** each keypoint by a vector `d` (a **descriptor**) summarising the neighbourhood so that the same landmark in another image gets a similar vector.
3. **Match**: pair keypoints of image A with keypoints of image B whose descriptors are close and (ideally) mutually best.

Old pipelines used SIFT; modern ones use learned detectors and matchers.

## 3. SuperPoint (detector + descriptor)

SuperPoint is one convolutional network that outputs, for every picture:

* a **keypoint heat map**: the network predicts for every 8 × 8 block of pixels a 65-way classification (64 positions + one "no keypoint here" dustbin), which is turned into a full-resolution probability map;
* a **256-number descriptor** per location, computed at 1/8 resolution and interpolated (bicubic) to each keypoint, then L2-normalised.

Settings in `load_matcher`:

| parameter | value | meaning |
|---|---|---|
| `max_num_keypoints` | 2048 | keep the strongest 2048 |
| `detection_threshold` | 0.0005 | very low: accept weak detections (thermal vs optical images are low-contrast) |
| `nms_radius` | 4 | non-maximum suppression: no two keypoints within 4 px |
| `remove_borders` | 4 | ignore keypoints within 4 px of the edge |

The picture is upscaled to `MATCH_SIZE = 2 × CTX` (448 × 448) with bicubic interpolation before detection:

```python
big = cv2.resize(picture.astype(np.float32), (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_CUBIC)
tensor = torch.from_numpy(big)[None, None].to(DEVICE)        # (batch=1, channel=1, H, W)
return matcher.keypoint_finder.extract(tensor, resize=None)
```

A 224-pixel canvas gives few keypoints; enlarging it by two gives the network more pixels to work with (`resize=None` stops the library from rescaling again).

## 4. LightGlue (the matcher)

LightGlue is a transformer that looks at **both** sets of keypoints at once. Inputs: for each image the keypoints' positions and descriptors. For `n_layers = 9` layers it alternates

* **self-attention** within an image (a keypoint learns about its neighbours: "I am the corner of a rectangle"), and
* **cross-attention** between images (a keypoint in A compares itself to the keypoints in B),

using the same attention formula as in `coarse_search/concepts/dino_descriptor_hint.md`: `softmax(QKᵀ/√d)V`. Position information is injected so geometry matters, not only appearance. After the layers, a small head computes:

* a **similarity** between every pair of keypoints (A,i) and (B,j);
* a **matchability** score per keypoint ("will this one have a partner at all?").

Together they define a soft assignment; a pair is output if it is mutually the best choice and its score exceeds `filter_threshold = 0.1`.

"Light" refers to adaptivity: if after some layer the matcher is already confident (`depth_confidence = 0.95`) it stops early; keypoints judged unmatchable (`width_confidence = 0.99`) are pruned from later layers. `flash=False` disables the FlashAttention kernel (needs special hardware).

`match_keypoints` returns pair coordinates in original canvas pixels:

```python
scale = picture_size / MATCH_SIZE            # undo the 2x enlargement
points_a = features_a["keypoints"][pairs[:, 0]].cpu().numpy() * scale
points_b = features_b["keypoints"][pairs[:, 1]].cpu().numpy() * scale
```

`pairs` has shape `(n, 2)`: index into A's keypoints and into B's. `rbd` ("remove batch dimension") strips the leading size-1 batch axis from the dictionaries.

## 5. MINIMA: the weights that bridge thermal and visible

Stock LightGlue weights are trained on pairs of photos of the same kind. **MINIMA** (Modality Invariant Image Matching, 2024) retrained the matcher on a huge amount of *generated* data in which one image of each pair was converted into a different modality (infrared, event, depth, sketch). The network learned that "the same place" can look completely different, exactly our thermal vs satellite situation. The file `minima_lightglue.pth` (~50 MB) holds only the LightGlue part; the SuperPoint weights come with the `lightglue` package.

**Layer-name patch.** The released file uses older parameter names than the installed library, so `load_matcher` renames them:

```python
for layer in range(9):
    weights = {k.replace("self_attn.%d" % layer, "transformers.%d.self_attn" % layer): v for k, v in weights.items()}
    weights = {k.replace("cross_attn.%d" % layer, "transformers.%d.cross_attn" % layer): v for k, v in weights.items()}
report = glue.load_state_dict(weights, strict=False)
```

`strict=False` loads whatever matches and *reports* the rest instead of crashing. The printed line shows how many file layers were not found and how many model layers had no weights. If the second number is not (almost) zero, the renaming failed and the matcher would run with random weights: always read this line. The code only prints it; extend `load_matcher` with an `assert` if you want a hard stop.

## 6. Inputs, outputs, constraints

| function | input | output |
|---|---|---|
| `load_matcher()` | nothing (reads `settings.MINIMA_FILE`, downloads from `MINIMA_URL` if missing) | `Matcher` holding SuperPoint and LightGlue |
| `find_keypoints(matcher, picture)` | 2-D float array in `[0,1]` | dict with `keypoints`, `descriptors`, `keypoint_scores` (tensors, batch dim 1) |
| `match_keypoints(matcher, fa, fb, size)` | two feature dicts, original picture side in pixels | `(points_a (n,2), points_b (n,2), scores (n,))` as float32 numpy |

Requirements: the `lightglue` package (installed from GitHub), PyTorch, the weights file. No network at evaluation time: the weights file must be on disk. Runtime: one SuperPoint pass per picture (~ tens of ms on a GPU) and one LightGlue pass per candidate; with 25 candidates this stage dominates the per-photo time.
