# train/models

Learned fusion of score maps and a candidate ranker (Test_Iteration_1). Not used by the submission pipeline.

| file | what |
|---|---|
| `fusion_ablation.py` | C2: learn how to fuse the score maps. Trains the three model kinds on several subsets of maps and reports which pays. |
| `ranker.py` | R1: learn to rank the strong candidates (not only the fused score). |

Related: tests/Test_Iteration_1. Repo map: `REPO_MAP.md` at the repo root.
