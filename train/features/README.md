# train/features

Feature families: a picture becomes a grid of cells (one cell = 4x4 pixels) for the map and for the query. The edge family is the one the submission pipeline uses; the others (MIND, DINO, learned) are experiments.

| file | what |
|---|---|
| `base.py` | A feature family turns a picture into a grid of 8-number cells (one cell = 4x4 pixels), for the map and for the query. |
| `dino_dense.py` | F3 of patch v0.5.5: dense DINO tokens as one more feature family. |
| `dino_tokens.py` | DINOv2 patch tokens: one 384-number vector per 14 x 14 pixel patch of a picture (layer 8, as src/coarse_search/dino_hint.py). |
| `edge.py` | The existing edge-direction cells (built by image_processing.map_store; nothing new here). |
| `learned_family.py` | The learned features (F4) used at search time: the trained nets turn the map and the query into cells like the edge family does. |
| `learned_net.py` | F4: learned correlation features. |
| `mind.py` | F1: MIND, local self-similarity. Each pixel is described by how much its patch resembles the patches 8 steps away. |
| `whitening.py` | C1: pre-whitening. A city grid repeats, so its power sits at a few spatial frequencies; dividing by the map's average |

Related: tests/Test_Iteration_1 and 2. Repo map: `REPO_MAP.md` at the repo root.
