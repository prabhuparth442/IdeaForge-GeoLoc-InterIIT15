# tests/train

Unit tests of the `train/` building blocks on a small synthetic world (no dataset needed, except `test_dino_blocks.py` which needs the DINOv2 weights).

| file | what |
|---|---|
| `test_blocks.py` | Tests for the iteration-1 building blocks (train/ package) on the synthetic world, no dataset needed. |
| `test_dino_blocks.py` | Tests for the DINO blocks (train/features/dino_*.py, train/maps/dino_similarity.py, train/coarse_methods) on the synthetic world. |
| `test_fusion.py` | Fusion network tests. Run: python tests/train/test_fusion.py |
| `test_simulation.py` | Tests for the drone-view simulator (train/simulation) on a small synthetic map written as a GeoTIFF, no dataset needed. |

Related: `python tests/train/test_blocks.py`, `python tests/train/test_fusion.py`. Repo map: `REPO_MAP.md` at the repo root.
