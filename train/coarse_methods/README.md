# train/coarse_methods

Coarse methods as lego pieces: each is a function `score_map(inputs) -> (rows, cols)` over the map (see `inputs.py`). Swap one for another in a run.

| file | what |
|---|---|
| `dino_rerank.py` | DINO as a second look at the shortlist of the other maps. The code lives in src/semantic_rerank/dino_rerank.py (one copy); |
| `fused.py` | a coarse method that is a trained fusion model over some of the maps (patch C2). |
| `hint.py` | the original way of using DINO: a base map plus beta times the pooled DINO similarity (src search_frame, beta = 0.5). |
| `inputs.py` | everything a coarse method may look at, for one photo, computed once. |
| `single_map.py` | a coarse method that is just one of the score maps (edge_zncc, mind_zncc, learned_zncc, dino_zncc, dino_pool ...). |

Related: tests/Test_Iteration_1 and 2. Repo map: `REPO_MAP.md` at the repo root.
