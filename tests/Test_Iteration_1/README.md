# Test_Iteration_1: seven coarse methods, fine matching, pseudo-thermal training

The first round of experiments on top of the pipeline in `src/`. The full story with all numbers and caveats is in
[`REPORT_seven_methods_and_synthetic_training.md`](REPORT_seven_methods_and_synthetic_training.md).

## What is where

| folder / file | what |
|---|---|
| `run_overnight.py` | the experiment on real NUDT photos: check, setup, geometry, f4, dump, fusion, ranker, evaluate, official, report |
| `run_synthetic.py` | train the fusion on pseudo-thermal photos cut from the Changsha map, test on real photos: setup, dump, fusion, evaluate, official, report |
| `stages/` | one small file per stage (`setup_nudt.py`, `dump.py`, `fusion.py`, ...). Each has one function, `run_stage(run)`. Shared pieces: `run_context.py` (log, saved state), `driver.py` (runs the stages), `helpers.py`, `nudt_check.py` and `official_check.py` (the evaluation blocks used by both experiments) |
| `notebooks/` | `Seven_Methods_Flow.ipynb` (NUDT), `Seven_Methods_Flow_D1.ipynb` (official set), `Overnight_Run.ipynb` (the overnight stages as cells). Executed, with outputs |
| `results/overnight/` | output of the real-photo run: `report.md`, `state.json`, `geometry.json`, `models/` (`f4.pt`, `fusion/`, `ranker.pt`), the per-photo CSVs of the notebooks |
| `results/synthetic/` | output of the synthetic run: `report.md`, `state.json`, `pseudo_samples.png`, `models/{synth,mixed}/` |

The building blocks themselves live in `train/` (see `train/README.md`). The pipeline is used straight from the repo's `src/`: nothing is copied and no existing `src` file is changed (`coarse_search/fusion.py` is a new module in `src/`). A later iteration that has to change an existing `src` file keeps only the changed files in its own `changed_src/` folder.

## Run

From this folder:

```bash
python run_overnight.py --smoke                      # a few minutes, tiny synthetic world: checks that every stage runs
python run_overnight.py --stages evaluate official   # only some stages (finished work is kept)
python run_synthetic.py                              # about 45 minutes, needs results/overnight/ (models, geometry, real candidate sets)
python ../train/test_blocks.py                      # tests of the building blocks (they live in tests/train/)
```

Data: NUDT is found automatically in `dataset/NUDT` of this repo or in `../Code_Testing/dataset/NUDT`; the official set (`d1`, `small_sat_area`) the same way.
Set `NUDT_DIR` or `DEV_DATA_DIR` to point somewhere else. The MINIMA matcher weights are `dataset/NUDT/checkpoint/MINIMA/minima_lightglue.pth`.
Needs a GPU for the real runs.

## Results kept in git, and what is not

Kept: code, notebooks with outputs, `report.md`, `state.json`, CSVs, `run.log` excluded by `*.log`.
Not kept (see `.gitignore`): model files (`*.pt`), candidate shards (`shards_*/`), smoke-run folders. The notebooks and `run_synthetic.py` need the
models and shards of `results/overnight/`; without them run `run_overnight.py` first (hours) or copy them from the shared drive.
