# tests

| folder | what | how to run |
|---|---|---|
| `pipeline/` | unit tests of the pipeline in `src/` (header, image processing, coarse search, fine matching, result thresholding, evaluation, end to end) on a small synthetic world, no dataset needed. Also how to swap one function for your own. Details: `pipeline/README.md` | `python tests/pipeline/run_all.py` |
| `train/` | tests of the `train/` building blocks: `test_fusion.py` (fusion models, training loop, `run_fusion.py`) `test_blocks.py` (feature families, map bank, candidate sets, learned features) and `test_dino_blocks.py` (DINO tokens, dense family, coarse-method contract; needs the DINOv2 weights) | `python tests/train/test_fusion.py`, `python tests/train/test_blocks.py` |
| `Parameter_Test/` | `Parameter_Test.ipynb`: how much every hyperparameter (K, M, DINO weight, bar, gamma, rescue, LoFTR confidence, RANSAC, border, margin) moves the score, with the strategies of the analysis. It grew out of `Stage_Evaluation.ipynb`; its data (per-photo tables, sweeps, `full_runs/`) is in the same folder | open the notebook |
| `Heading_Queries/` | `Heading_Queries.ipynb`: every photo drawn at 9 heading guesses (-10 to +10 degrees) and answered from all candidates, three random seeds side by side. A record test, not a tuning run (Patch v0.5.6). Blocks: `src/coarse_search/heading_queries.py`, `heading_query_pipeline.py`, `train/checks/heading_results.py` | open the notebook |
| `Test_Iteration_2/` | adding DINO (patch v0.5.5): blocks, notebook and a finding about the dense map. See its README | see its README |
| `Test_Iteration_1/` | the first round of experiments on real data (seven coarse methods, fine matching, pseudo-thermal training): runners, stages, notebooks, results, report. Needs the datasets and a GPU. See its README | see its README |

Everything at once (pipeline and train tests, a few minutes): `python tests/run_all.py`.
