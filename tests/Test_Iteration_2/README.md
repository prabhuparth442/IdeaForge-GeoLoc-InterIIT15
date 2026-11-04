# Test_Iteration_2: adding DINO (patch v0.5.5)

Iteration 1 fused six maps that all compare *where edges run*. DINO describes *what is there*. This iteration asks whether adding it
helps the coarse search, and builds the pieces to put it into the fusion. Rules: `../../DESIGN.md` (notebook is the face, atomic blocks do
the work). The pipeline is used from the repo's `src/`; no existing `src` file is changed, so there is no `changed_src/`.

**Full report: [REPORT_dino_iteration_2.md](REPORT_dino_iteration_2.md)**

## What the patch asks for (Documents/Our_Architecture/Patch_v0.5.5_Breakthroughs.tex)

| id | idea | block here |
|---|---|---|
| F3 | dense DINO tokens (layer 8) reduced by a PCA, resampled to the 4 x 4 pixel cell grid, correlated with the same masked ZNCC: a second score map | `train/features/dino_tokens.py`, `dino_dense.py` (`DenseDinoFamily`) |
| old hint | pooled DINO similarity (one vector per 224 px tile) as one more map | `train/maps/dino_similarity.py` (`PooledDinoMap`, `add_pooled_map`) |
| C2 | normalise each map per photo (robust z), a trained model weighs them instead of a hand-set beta = 0.5 | existing `coarse_search/fusion.py` and `train/models/fusion_ablation.py`; the new maps enter through the hook below |
| X6 | each map alone and fused, on the photos. Pass: fused beats edges + pooled prior (beta 0.5) by 0.03 in the low-altitude ceiling | `notebooks/Dino_Coarse.ipynb`, Part 7 |

Also new, from `DESIGN.md`: `train/coarse_methods/` (one small function per coarse method, same inputs, one score map out;
`build_methods`, `run_methods`, `make_inputs`) and `train/datasets/photo_sets.py` (official or NUDT photos as a `TestSet`).
The candidate dump `train/datasets/candidates.dump_split` got one optional hook, `extra_maps` / `extra_names`, for maps that
`compute_maps` cannot make (the pooled DINO map). Without the hook nothing changes.

## What is in this folder

| path | what |
|---|---|
| `notebooks/Dino_Coarse.ipynb` | the main face: settings, photos and map, DINO encoder + PCA + the two DINO maps, what DINO sees, the shift test, the methods on one photo, coarse statistics, how the maps go into fusion data |
| `results/` | `cache/` (dense DINO map features, rebuilt on demand, not in git), small demo outputs |

Tests of the blocks: `python tests/train/test_dino_blocks.py` (needs the DINOv2-S weights in the local cache).

## A finding made while building it: dense DINO is not shift-invariant

DINO tokens sit on a 14 px grid. On the official map, with a crop of the map itself as the query:
- a query identical to one window of the token grid is found exactly (score 1.0, error 0 m);
- the same query shifted by (77, 51) px is not: the score at the true place is about -0.1, while places elsewhere in the 5 km circle reach about 0.85 (the correlation latches onto grid-phase patterns). The edge map finds both.
- Window margins, and averaging over copies shifted by half a token (`phases=` in `DenseDinoFamily`), did not cure it in our tests.
- A map per token-grid phase (49 maps) would, at about 49 times the build and correlation cost.

The patch does not mention this. The dense map therefore has to be judged on real photos before more work goes into it; the pooled map (tile based, no grid) does not have the problem.

## First results (60 official photos, coarse stage only, from the executed notebook)

Top-1 = the first of the 25 peaks is within the given sigma of the truth. Ceiling = the right place is anywhere in the 25.

| method | top-1 <= 4 sigma | ceiling <= 4 sigma | ceiling, low photos | ceiling, high photos |
|---|---|---|---|---|
| edge_zncc | 0.833 | 0.917 | 0.867 | 0.967 |
| mind_zncc | 0.850 | 0.950 | 0.900 | 1.000 |
| learned_zncc | 0.883 | 0.967 | 0.933 | 1.000 |
| fused (6 maps, iteration 1) | 0.917 | 0.983 | 0.967 | 1.000 |
| edge_zncc + 0.5 x pooled DINO (old hand-set blend) | 0.850 | 0.933 | 0.900 | 0.967 |
| **dino_zncc** (dense, F3) | **0.000** | **0.033** | 0.067 | 0.000 |
| **dino_pool** (pooled tile similarity) | **0.000** | **0.033** | 0.067 | 0.000 |

- **Dense and pooled DINO are useless as methods on their own** here: the first peak is never right, and the right place is in the 25 for 2 of 60 photos. This is more extreme than "weak". It fits the shift finding above for the dense map; the pooled map is tile-level, so it cannot place a photo inside a tile.
- **The hand-set blend adds a little** to edges (+0.017 top-1, +0.017 ceiling), which is one photo of 60 each, so it could be noise.
- **Not tested yet:** whether DINO helps *inside the fusion* (C2). A map that is useless alone can still add information next to others, and a map that looks fine alone can add nothing. That needs the real fusion training of plan step 3.
- Shift test, 8 crops of the map itself: edge_zncc finds all 8 aligned and all 8 shifted (score 0.98 / 0.75); dino_zncc finds all 8 aligned (1.00) and 0 of 8 shifted (-0.07).
- Plumbing check: candidate sets with 8 maps have shape (photos, 79 candidates, 8 maps) and the existing `run_ablation` trains on them.

## Plan

1. **Done:** blocks, tests, notebook (this folder). The notebook shows each step on real photos.
2. **Judge on more photos:** Part 7 of the notebook on the 500 official photos and on the NUDT test photos (the first run above used 60 official photos).
3. **If anything adds information:** dump real training candidate sets with the DINO maps (3000 NUDT train + valid photos, about 20 minutes with the hook), retrain the fusion on 7-8 maps and compare with the 6-map fusion on real photos (paired bootstrap, official set as the check). This needs a runner like iteration 1's; not written yet.
4. **If dense DINO stays weak:** try the phase-exact map once, or keep only the pooled map.
5. **Related:** pooled DINO similarity as a feature for the ranker (patch R1); whether pseudo-thermal photos (iteration 1) help the DINO fusion; DINO weights on the Cartosat imagery of the real contest.

## Settings worth knowing
- 8 PCA channels, not the patch's 16: the pipeline (`src/coarse_search/search.py`) is built for 8 channels. The patch recommends 8 for small GPUs.
- Two DINOv2-S encoders are in memory (the dense family and `src`'s `DinoHint`); both are small (about 22 M parameters).


## Second try: DINO as a re-ranker of the shortlist (block `train/coarse_methods/dino_rerank.py`)

The first try gave DINO its own map over the whole disc and it failed (see First results). The better question: take the **50 best peaks of a non-DINO method**, look at only those places with DINO, re-order, and give the best 25 to the fine stage.
Each shortlisted place gets a 224 px map window cut exactly at the peak; query and window both go through DINO with the query's valid mask on both. Their token grids start at their own corners, so the grid-phase problem of the dense map does not arise. Two scores: `pooled` (cosine of mean tokens) and `dense` (mean token-to-token cosine over valid tokens). Re-order by z(first score) + weight x z(DINO score).

Notebook: `notebooks/Dino_Rerank.ipynb` (all 379 official photos, about 5 min). Earlier raw numbers: `results/rerank_official_379.pkl`.

Coarse stage only, 379 official photos (top-1 = first peak within 4 sigma; ceiling@25 = right place anywhere in the first 25):

| first stage | re-rank | top-1 | ceiling@25 |
|---|---|---|---|
| fused (6 maps) | none | 0.897 | 0.968 |
| fused | pooled, weight 0.5 | 0.918 | 0.968 |
| fused | dense, weight 1 | 0.958 | 0.968 |
| fused | dense, weight 4 | 0.966 | 0.968 |
| fused | dense alone inside the 50 | 0.966 | 0.968 |
| edge_zncc | none | 0.802 | 0.897 |
| edge_zncc | dense alone inside the 50 | 0.913 | 0.916 |

Paired bootstrap (5000 resamples) for "dense alone inside the 50" vs. the first stage: fused +0.069 (95 % interval +0.045 to +0.095; fixed 26 photos, broke 0); edge_zncc +0.111 (+0.079 to +0.142; fixed 42, broke 0).

**Reading it.**
- The dense DINO score is a strong judge of a shortlisted place even though it is useless as a search map. It fixes the ordering inside the shortlist and breaks nothing.
- It does not add reach: ceiling@25 is unchanged for fused (shortlist 50 and 25 have the same ceiling, 0.968). The gain is in rank 1. The fine stage already tries all 25 peaks, so the benefit shows only where the decision uses the order or the peak score; this has to be measured with the fine stage (next step), not assumed.
- Weights were swept on the same photos, but "dense alone" (weight 100) needs no tuning and is as good as the best weight.
- Pooled DINO helps a little at small weights and hurts at large ones: it cannot tell places inside a tile apart.
- Only the 379 official photos on small_sat_area (a different region from the map the models were trained on). Not yet run on NUDT test photos or with the fine stage.

**Next.** (1) Put this block between `pick_peaks` and `second_stage` in a flow notebook and score the fine-stage result. (2) Run on NUDT test photos. (3) Try feeding the DINO score to the decision rule as well, since it is independent of the edge score.

**Now in `src`.** The re-rank is a new section `src/semantic_rerank/` (`dino_rerank.py`: `shortlist_peaks`, `dino_candidate_scores`, `rerank_peaks`, `semantic_layer`), sitting between `coarse_search` and `fine_matching`. `train/coarse_methods/dino_rerank.py` only re-exports it, so the notebook and tests use the one copy. It is not wired into `result_thresholding/pipeline.py` (opt-in until the fine-stage result is measured).
