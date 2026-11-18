# IdeaForge-GeoLoc-InterIIT15

UAV visual geo-localisation for the ideaForge problem statement (Inter-IIT Tech Meet 15.0, team 7114): a thermal photo (320 x 256, with height, yaw, pitch, roll and a GPS hint up to 5 km off) is placed on a satellite map. This is the **research and testing repo**. The submission code is the separate `../Main_Pipeline_Headless`.

**Start here:** `REPO_MAP.md` (what is where, for any AI or person), then `docs/HANDOFF_2026-11-17.md` (current state, results, server layout, open tasks), `CHANGELOG.md` (newest work) and `tests/README.md` (what every test folder is). `AGENTS.md` holds the working rules.

## Run

```
python run.py                      # whole run on the development photos
python run.py --n-photos 60        # smaller sample
python run.py --no-minima          # correlation only
python run.py --dino --tests       # more switches: --sweep --sensitivity --independence
```

Settings are in `src/header/settings.py`. Big data files (maps, model weights): `dataset/README.md`.

## Layout

| path | what |
|---|---|
| `src/<section>/` | the pipeline as atomic blocks (header, image processing, coarse search, semantic rerank, fine matching, result thresholding, evaluation), each with its own README. One copy only; experiments put changed files in their own `changed_src/` |
| `train/` | building blocks and checks for experiments (`train/checks/` has the rich dump, replay, perturbations) |
| `Stage_Evaluation.ipynb` | **the report notebook**: six stage tables, recall at K, funnel, heading queries, time, sweeps, sample figures. Driven by `SE_*` environment variables (data, photo count, tag, map files, levels, height bands); executed copies per data set are in `tests/Test_Iteration_12/results/stageeval/` |
| `Single_Image_Run.ipynb` | one photo through the whole pipeline, step by step |
| `Stage_Evaluation_Colab.ipynb`, `Pipeline_Visual_Check.ipynb`, `Fine_Fit_Methods.ipynb`, `Batch_Speed_Check.ipynb` | older or side notebooks, kept with their outputs |
| `single_photo_pipeline.py`, `agreement_query_pipeline.py`, `heading_query_pipeline.py`, `inflation_guard_pipeline.py` | pipeline classes the notebooks import (single query, agreement rule, heading queries, inflation guard) |
| `tests/` | unit tests and every experiment round (iterations 1-12), see `tests/README.md` |
| `docs/` | handoff, prompts to Fable, overnight and tidy logs, see `docs/README.md` |
| `tools/` | `run_capped.sh` (heavy jobs with a 6 GB / 8 core cap), `pack_dataset.py` |
| `third_party/` | vendored matcher code |
| `dataset/` | official photos and small maps (large files gitignored); `dataset_temp/` is never uploaded |
| `DESIGN.md` | the notebook-is-the-face, atomic-block design rules |

## Where things stand (2026-11-17)

Old official set 52.69 / 60 base (53.60 with altered copies); NUDT 54.24 / 60 (saturated); new high-altitude set 36.35 / 60 base, losses mainly above 800 m (matcher failure, not search). Altered copies is the only repeatable gain; lower bar, agreement rule and aligned rescue do not transfer across data sets. Next experiments (Fable's coarse-answer rule above 800 m, pose-from-matches for precision) are listed in the handoff. Iterations 24-28 (coarse-stage analysis of 2026-11-17) and the 2026-11-18 tidy are not committed.
