# tests

| folder | what | how to run |
|---|---|---|
| `pipeline/` | unit tests of the pipeline in `src/` (header, image processing, coarse search, fine matching, result thresholding, evaluation, end to end) on a small synthetic world, no dataset needed. Also how to swap one function for your own. Details: `pipeline/README.md` | `python tests/pipeline/run_all.py` |
| `train/` | tests of the `train/` building blocks: `test_fusion.py` (fusion models, training loop, `run_fusion.py`) `test_blocks.py` (feature families, map bank, candidate sets, learned features) and `test_dino_blocks.py` (DINO tokens, dense family, coarse-method contract; needs the DINOv2 weights) | `python tests/train/test_fusion.py`, `python tests/train/test_blocks.py` |
| `Test_Iteration_2/` | adding DINO (patch v0.5.5): blocks, notebook and a finding about the dense map. See its README | see its README |
| `Test_Iteration_1/` | the first round of experiments on real data (seven coarse methods, fine matching, pseudo-thermal training): runners, stages, notebooks, results, report. Needs the datasets and a GPU. See its README | see its README |

Everything at once (pipeline and train tests, a few minutes): `python tests/run_all.py`.
