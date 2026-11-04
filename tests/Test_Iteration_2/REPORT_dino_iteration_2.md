# Iteration 2 report: adding DINO (patch v0.5.5) to the coarse search

Written for: the team members who need to know what was tried with DINO, what worked, and what is still open.
Date of the runs: 2026-11-04. Nothing in this report is committed to git.

## 1. In one paragraph

DINO as its own search map over the whole 5 km disc does not work (right place in the first 25 peaks for only 2 of 60 photos). DINO as a **judge of a shortlist** does work: the coarse method gives its best 50 places, DINO re-orders them, the best 25 go on. On all 379 official photos this lifts the first-peak accuracy of the fused score from 0.897 to 0.966 and of edge ZNCC from 0.802 to 0.913, with no photo made worse. The layer is now a module in `src/semantic_rerank/`. What it gains after the fine stage (the real contest score) is **not measured yet**.

## 2. What the patch asked for and what was built

| patch idea | what it means | what was built | where |
|---|---|---|---|
| F3, dense DINO | DINO patch tokens, reduced by PCA to 8 channels, put on the 4 x 4 px cell grid, correlated with the same masked ZNCC | `DenseDinoFamily` (a feature family like edge / MIND / learned), DINO token and PCA blocks | `train/features/dino_tokens.py`, `dino_dense.py` |
| pooled DINO similarity (the old "hint") | one vector per 224 px tile, cosine to the query, as one more score map | `PooledDinoMap`, `add_pooled_map` | `train/maps/dino_similarity.py` |
| C2, fusion | z-score each map, learned weights instead of hand-set beta 0.5 | the existing fusion trainer takes DINO maps through one new hook (`extra_maps`, `extra_names` in `dump_split`) | `train/datasets/candidates.py` |
| X6, test | each map alone and fused vs. edges + 0.5 x pooled | method blocks with one contract, `build_methods` / `run_methods` | `train/coarse_methods/` |
| **new, from this iteration** | DINO as a layer between coarse and fine | `semantic_layer` | `src/semantic_rerank/dino_rerank.py` |

Rules followed (`DESIGN.md`): the notebook is the face, blocks do the work, one copy of `src`, no `changed_src` needed (the only change to existing code is the added hook in a `train/` block).

## 3. Experiment A: DINO as its own score map (`notebooks/Dino_Coarse.ipynb`)

60 official photos (d1 on small_sat_area), coarse stage only. Top-1 = first peak within 4 sigma of the truth. Ceiling = the right place is anywhere in the 25 peaks.

| method | top-1 | ceiling |
|---|---|---|
| edge_zncc | 0.833 | 0.917 |
| mind_zncc | 0.850 | 0.950 |
| learned_zncc | 0.883 | 0.967 |
| fused (6 maps, iteration 1) | 0.917 | 0.983 |
| edge_zncc + 0.5 x pooled DINO (old hand-set blend) | 0.850 | 0.933 |
| dense DINO alone | 0.000 | 0.033 |
| pooled DINO alone | 0.000 | 0.033 |

**Why dense DINO fails as a map (the shift test, Part 5 of the notebook).** DINO tokens sit on a fixed 14 px grid, so the dense map is not shift-invariant. Using a crop of the map itself as the query:

| query | edge map finds the place | dense DINO finds the place |
|---|---|---|
| crop aligned to the token grid | 8 of 8 | 8 of 8 (score 1.000) |
| same crop shifted by (77, 51) px | 8 of 8 | **0 of 8** |

A real photo is never aligned to the map's token grid. Averaging over shifted copies and margins did not fix it in our tests; one map per grid phase would, at about 49 times the cost. The pooled map works per tile and cannot place a photo inside one.

The hand-set blend adds 0.017 over edge alone (one photo in 60): not distinguishable from noise.

## 4. Experiment B: DINO as a re-ranker of the shortlist (`notebooks/Dino_Rerank.ipynb`)

Idea (from the user): first take 50 peaks by a non-DINO score, then pass those through DINO.

**Method.** For each of the 50 peaks, cut a 224 px map window exactly at the peak, put the photo's valid mask on it, run photo and window through DINO (layer 8). Two scores:
- *pooled*: cosine of the mean tokens (what kind of place);
- *dense*: mean over valid tokens of the token-to-token cosine (what lies where).

Re-order by z(coarse score) + weight x z(DINO score); weight 100 means DINO alone decides the order inside the shortlist. Photo and window both start their token grid at their own corner, which removes the grid problem of Experiment A.

**Data.** All 379 official photos the loader provides (the earlier "500" was a request cap), coarse stage only, same fused model as iteration 1.

| first stage | re-rank | top-1 | ceiling@25 |
|---|---|---|---|
| fused | none | 0.897 | 0.968 |
| fused | pooled, w 0.25 / 0.5 / 1 / 2 / 4 / 100 | 0.916 / 0.918 / 0.913 / 0.892 / 0.784 / 0.227 | 0.968 (0.931 at w 100) |
| fused | dense, w 0.25 / 0.5 / 1 / 2 / 4 / 100 | 0.926 / 0.950 / 0.958 / 0.963 / 0.966 / 0.966 | 0.968 |
| edge_zncc | none | 0.802 | 0.897 |
| edge_zncc | pooled, w 0.25 / 0.5 / 1 / 2 / 4 / 100 | 0.813 / 0.815 / 0.828 / 0.839 / 0.828 / 0.346 | 0.908 to 0.916 |
| edge_zncc | dense, w 0.25 / 0.5 / 1 / 2 / 4 / 100 | 0.823 / 0.844 / 0.873 / 0.897 / 0.913 / 0.913 | 0.908 to 0.916 |

Shortlist ceiling (right place anywhere in the 50): fused 0.968, edge_zncc 0.916.

**Paired bootstrap** (5000 resamples), dense DINO alone inside the 50 vs. no re-rank, top-1:

| first stage | before | after | difference | 95 % interval | photos fixed / broken |
|---|---|---|---|---|---|
| fused | 0.897 | 0.966 | +0.069 | +0.045 to +0.095 | 26 / 0 |
| edge_zncc | 0.802 | 0.913 | +0.111 | +0.079 to +0.142 | 42 / 0 |

A first run on 60 photos (before the 379-photo run) gave fused 0.917 to 0.983 and edge 0.833 to 0.900, the same direction.

## 5. What this means

1. DINO is a **strong judge of a candidate place** and a **useless searcher over a whole region**. The two roles need different handling; the window cut at the peak is what makes the judge work.
2. The gain is in the **order**, not the **reach**. For fused the ceiling@25 does not move (the shortlist of 50 holds nothing the first 25 do not). For edge ZNCC the ceiling@25 rises from 0.897 to 0.916 because the 50-shortlist reaches places the first 25 miss.
3. "DINO alone decides inside the 50" (weight 100) is as good as the best tuned weight (4) for the dense score, so the headline number needs no tuning. For the pooled score a large weight is harmful (0.227 for fused): the pooled score cannot separate places inside one tile.
4. The weights were swept on the same photos that are reported, but the headline (weight 100) has no tuned parameter.

## 6. Limits and what is not known

- **Fine stage not run.** The fine stage tries all 25 peaks already, so a better rank 1 helps only where the decision uses the order or the peak score, or where the 25 differ. The contest score effect is unmeasured.
- **One data set.** 379 official photos on small_sat_area. NUDT test photos (Changsha, the region the models were trained on) not yet run.
- **Dense DINO as a map in the fusion is untested** with real training data (3000 NUDT photos). Experiment A predicts it is weak; it was not proven inside a trained fusion.
- **Cost not measured.** 50 map windows per photo go through DINO (batches of 25). The notebooks ran 379 photos in about 6 minutes in total, including everything else.
- GPU tests: `test_coarse_methods_share_one_contract` runs out of GPU memory while other notebook kernels hold the GPU; it passes on CPU. All 8 DINO block tests pass on CPU.
- Known failure not caused by this work: `tests/pipeline/test_end_to_end.py::test_report_runs_on_synthetic_results` (empty `min()` in `src/evaluation/report.py`), also failing before the reorganisation.

## 7. Next steps, in order

1. A flow notebook: coarse, then `semantic_layer`, then `second_stage` and the decision rule, scored with the contest points, on the official set and the NUDT test photos (paired bootstrap against no re-rank).
2. If it helps: wire the layer into `result_thresholding/pipeline.py` behind a flag (this is a change of an existing `src` file, so it goes in `changed_src/` of the iteration).
3. Feed the DINO score to the decision rule too (it is independent of the edge score).
4. Real fusion training with DINO maps (only if step 1 shows the layer is not enough).

## 8. Files

| what | where |
|---|---|
| module in the pipeline | `src/semantic_rerank/dino_rerank.py`, `README.md` |
| notebook, DINO as map | `notebooks/Dino_Coarse.ipynb` (executed, 60 photos) |
| notebook, DINO as re-ranker | `notebooks/Dino_Rerank.ipynb` (executed, 379 photos) |
| raw per-photo results of the 379 run | `results/rerank_official_379.pkl` |
| DINO and coarse-method blocks | `train/features/dino_*.py`, `train/maps/dino_similarity.py`, `train/coarse_methods/` |
| tests | `tests/train/test_dino_blocks.py` |
| short summary | `README.md` in this folder |
