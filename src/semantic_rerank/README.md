# semantic_rerank: a second, semantic look between coarse search and fine matching

## What this section does, in plain words

`coarse_search/` scores edge patterns and returns places. Edge patterns are blind to *what* a place is. This layer takes the coarse score map, keeps its **best 50 places**, and asks DINO (a network that knows roofs, roads and fields) how well a map window cut exactly at each place matches the photo. The 50 are re-ordered and the **best 25** go on to `fine_matching/` in the same layout as before.

```
coarse score map ──pick_peaks (50)──► shortlist (50, 5)
                                          │  for each peak: map window at the peak + the photo's valid mask
                                          ▼
                        DINO tokens of photo and window  ──►  dense score (token-to-token cosine)
                                          ▼
              order by  z(coarse score) + weight * z(DINO score)      (weight 100: DINO decides)
                                          ▼
                          peaks (25, 5)  ──► fine_matching.second_stage
```

## Why it is a layer and not another map

DINO as its own map over the whole disc fails (its tokens sit on a 14 px grid, so the score is not shift-invariant). Here the window is cut at the peak, so photo and window have the same grid.

## Measured (tests/Test_Iteration_2, 379 official photos, coarse stage only)

| coarse score | top-1 within 4 sigma before | after | photos fixed / broken |
|---|---|---|---|
| fused (6 maps) | 0.897 | 0.966 | 26 / 0 |
| edge ZNCC | 0.802 | 0.913 | 42 / 0 |

The right place anywhere in the 25 (ceiling) does not change. What it buys after the fine stage is **not yet measured**.

## Use

```python
from semantic_rerank.dino_rerank import load_encoder, semantic_layer

encoder = load_encoder()
peaks, details = semantic_layer(encoder, score_map, geometry, canvas, valid, level, store["levels"])
stage2 = second_stage(matcher, canvas, valid, level, peaks, store["levels"])
```

`result_thresholding/pipeline.py` does not call this layer yet; it is opt-in until the fine-stage result is measured.

## Files

| file | job |
|---|---|
| `dino_rerank.py` | `shortlist_peaks`, `dino_candidate_scores`, `rerank_peaks`, `semantic_layer` |
