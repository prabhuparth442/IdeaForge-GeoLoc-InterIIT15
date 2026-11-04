# DESIGN: notebooks are the face, atomic blocks do the work

## The idea

- **The notebook is the main face of every piece of work.** You open it, run a few cells and see pictures and numbers. A reader should follow the whole pipeline from the notebook alone.
- **The work itself is done by small atomic blocks** (plain `.py` files, "lego pieces"). A notebook cell calls a block; it does not contain the logic.
- **Any block can be swapped later.** Because every block has the same inputs and outputs as the one it replaces, a new idea means a new block and a one-line change in the notebook, not a rewrite.
- **A new method is inserted, not built into the pipeline.** The seven coarse methods of iteration 1 should be seven small block files. The main pipeline notebook takes them from a list, and adding an eighth means adding one file and one name.

## Rules for a block

1. One job, one file, short functions. Plain loops and full names (see `HUMAN_CODE_STYLE.md` in the base folder).
2. Explicit inputs and outputs, written in the docstring. No hidden global state, no reading of notebook variables.
3. No plotting, no printing of results and no hard-coded paths inside a block. Pictures and tables belong to the notebook; paths come from `train/config.py`.
4. A block can be run and tested alone (`tests/`), on the synthetic world, without the real data.
5. A block never edits `src/`. The pipeline in `src/` is used where it is, and there is only one copy of it. If an experiment has to change an existing `src` file, only the changed files go into `changed_src/` inside that experiment's folder.

## Rules for a notebook

1. A few cells per step; each cell calls a block and shows what came out (a picture, a table, a number that can be checked).
2. The list of what is used (which methods, which model, which data) sits in one settings cell at the top, so swapping a part means changing that cell.
3. Notebooks keep their outputs when they are the record of a result.
4. A notebook finds the repo by walking up from its own folder and adds `src/` and the repo root to the path; it does not depend on where Jupyter was started.

## The swap points of the pipeline

Each row is one place where a block can be replaced by another with the same signature.

| step | block today | gets | gives |
|---|---|---|---|
| photo to canvas | `src.image_processing.geometry.render_query` | photo item, camera, level | `canvas` (224 x 224), `valid` mask |
| feature family | `train.features.*` (`EdgeFamily`, `MindFamily`, `LearnedFamily`) | canvas, valid; map level | cells (8 x 56 x 56) + valid cells; map cells per level |
| score maps | `train.maps.map_bank.compute_maps`, `z_stack` | canvas, valid, level, prior, families | one score map per name, shared grid, z-scores, context |
| **coarse method** | the seven methods (see below) | the shared maps | **one score map** in (-1, 1), -1 where unusable |
| peaks | `src.coarse_search.search.pick_peaks` | score map, geometry | 25 candidates `[score, x, y, sharpness, rank]` |
| fine matching | `src.fine_matching.second_stage.second_stage` | matcher, canvas, valid, level, peaks | inliers and shifts per candidate |
| decision | `src.result_thresholding.decision.final_answer`, `decide` | peaks, fine result | position, inliers, answer or skip |
| evaluation | `train.checks`, `src.header.scoring` | answers, truths | accuracy, points, score, paired bootstrap |

## Where iteration 1 stands against this idea

Already like this:
- The experiment code is split into blocks under `train/` and one file per stage under `tests/Test_Iteration_1/stages/`.
- Fine matching, the decision rule and scoring are `src/` functions that are called, not copied.
- The two flow notebooks call these pieces and show the result.

Started in iteration 2:
- `train/coarse_methods/` holds coarse methods as separate small functions with one contract (`single_map`, `fused`, `hint`, and `inputs.make_inputs` for the shared inputs), with `build_methods` / `run_methods`. The new DINO notebook uses it.
- `train/datasets/photo_sets.py` loads the official or the NUDT photos as one block.

Not yet like this (to do):
- The seven coarse methods of iteration 1 still live inside a notebook cell (`make_one_score_map`, `make_score_maps` in section 3.1 of the flow notebooks); they can now be replaced by `train/coarse_methods`.
- The results table, the paired bootstrap and the charts are written in notebook cells, not as blocks.
- There are two flow notebooks (NUDT and official data) that repeat most of their cells. One main pipeline notebook with a data setting would do.

## Next steps to make it so

1. Switch the iteration-1 flow notebooks to `train/coarse_methods` (the package exists; `build_methods` takes the list of map names and fused models).
2. Move the results table, the bootstrap and the standard charts into `train/checks/`.
3. Build one main pipeline notebook: settings cell (data set, list of methods, fusion model), then a few cells per step in the order of the table above. The NUDT and the official versions differ only in the settings cell.
4. For each later iteration: a folder under `tests/` with its notebook, its `stages/` or blocks, its results and, only if needed, `changed_src/`.
