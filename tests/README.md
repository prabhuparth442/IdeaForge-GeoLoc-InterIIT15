# tests

| folder | what | how to run |
|---|---|---|
| `pipeline/` | unit tests of the pipeline in `src/` (header, image processing, coarse search, fine matching, result thresholding, evaluation, end to end) on a small synthetic world, no dataset needed. Also how to swap one function for your own. Details: `pipeline/README.md` | `python tests/pipeline/run_all.py` |
| `train/` | tests of the `train/` building blocks: `test_fusion.py` (fusion models, training loop, `run_fusion.py`) `test_blocks.py` (feature families, map bank, candidate sets, learned features) and `test_dino_blocks.py` (DINO tokens, dense family, coarse-method contract; needs the DINOv2 weights) | `python tests/train/test_fusion.py`, `python tests/train/test_blocks.py` |
| `Parameter_Test/` | `Parameter_Test.ipynb`: how much every hyperparameter (K, M, DINO weight, bar, gamma, rescue, LoFTR confidence, RANSAC, border, margin) moves the score, with the strategies of the analysis. It grew out of `Stage_Evaluation.ipynb`; its data (per-photo tables, sweeps, `full_runs/`) is in the same folder | open the notebook |
| `Heading_Queries/` | `Heading_Queries.ipynb`: every photo drawn at 9 heading guesses (-10 to +10 degrees) and answered from all candidates, three random seeds side by side. A record test, not a tuning run (Patch v0.5.6). Blocks: `src/coarse_search/heading_queries.py`, `heading_query_pipeline.py`, `train/checks/heading_results.py` | open the notebook |
| `Test_Iteration_2/` | adding DINO (patch v0.5.5): blocks, notebook and a finding about the dense map. See its README | see its README |
| `Test_Iteration_1/` | the first round of experiments on real data (seven coarse methods, fine matching, pseudo-thermal training): runners, stages, notebooks, results, report. Needs the datasets and a GPU. See its README | see its README |

| `Single_Image_Run/`, `Stage_Evaluation/`, `Batch_Speed_Check/`, `Fine_Fit_Methods/` | the saved tables of the root notebooks of the same names (per-photo results, stage tables, decision-step sweeps, batch-size timings, fit-method comparison) | open the notebook at the repo root |
| `Test_Iteration_3/` | the decision after the queries: agreement and escalation (`Decision_Report.md`), dumps and notebooks | see the report |
| `Test_Iteration_4/` | XoFTR against LoFTR as the fine matcher (`XoFTR_vs_LoFTR*.ipynb`, results) | open the notebooks |
| `Test_Iteration_5/` | drone-view simulator: map in, simulated thermal photo out (`README.md`) | see its README |
| `Test_Iteration_6/` | the prompt of the round that made the rich dump (every candidate of every query with its matches); the dump code is `train/checks/rich_dump.py` | `PROMPT.md` |
| `Test_Iteration_7/` | the decision after fine matching: Palash's judge checked on our dumps, early exit, swapped training, fresh draws, the inflation guard, and the ideas-planner tests (ideas 0 to 9, incl. altered copies 5b and the aligned-count rescue 1b-1e). Start with `OURS_CHECK.md`; Palash's write-up is `REPORT.md`; launchers are in `scripts/`, logs in `results/logs/` | `OURS_CHECK.md` |
| `Test_Iteration_8/`, `Test_Iteration_9/`, `Test_Iteration_10/`, `Test_Iteration_11/` | Palash's later rounds (imported 2026-11-15): honest hold-outs (T8), red team with altered photos (T9), evaluation fixes and inlier-scale robustness (T10), a judge trained on satellite imagery only (T11). Each has `README.md` and `REPORT.md` | see the folders |
| `Test_Iteration_12/` | the high-altitude set (Rajasthan, 100-2017 m): three-set Stage_Evaluation reports, mega dump and replay, dataset similarity, Fable's truth diagnostic, vignette cleaning, GPU-server chains. `README.md` inside; state in `docs/HANDOFF_2026-11-17.md` | see its README |
| `*_codex/` (`Review_20261113_codex`, `Decision_Ideas_20261114_codex`, `Satellite_Only_20261114_codex`, `Review_20261115_codex`, `Speed_20261115_codex`) | isolated reviews and experiments made by another tool; each folder is self-contained (`README_codex.md`, `PROTOCOL_codex.md`, `REPORT_codex.md`) and can be deleted alone without affecting anything else. `Satellite_Only_20261114_codex` is 612 MB | see each README |

Everything at once (pipeline and train tests, a few minutes): `python tests/run_all.py`.
