# train/maps

Score maps: many score maps from one photo (`map_bank.py`), and the pooled DINO similarity as a map.

| file | what |
|---|---|
| `dino_similarity.py` | The pooled DINO similarity as a score map (the "DINO hint" of src/coarse_search/dino_hint.py, kept as a map of its own). |
| `map_bank.py` | Many score maps from one photo. |

Related: tests/Test_Iteration_1 and 2. Repo map: `REPO_MAP.md` at the repo root.
