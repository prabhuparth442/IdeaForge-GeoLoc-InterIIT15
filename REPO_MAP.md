# REPO_MAP: what is where (read this first, then open only what the task needs)

Written 2026-11-18. Meant for any AI or person who enters this repo cold. Paths are relative to this folder (`IdeaForge-GeoLoc-InterIIT15/`) unless they start with `../`. The folder above this one (`InterIIT Tech 15/`) has its own `index.md` (workspace map) and `MAIN_PIPELINE.md` (the submitted pipeline); read those for anything about the submission.

## 1. What this repo is

The **research and testing repo** for the ideaForge problem (Inter-IIT Tech Meet 15, team 7114): a thermal drone photo (320 x 256, plus height, yaw, pitch, roll and a GPS hint up to 5 km off) must be placed on a satellite map. Score per photo from the error in sigma units: within 1 sigma +10, 2 sigma +6, 4 sigma +3, 8 sigma 0, beyond 8 sigma -20, no answer 0 (`src/header/scoring.py`). The **submission code is not here**: it is `../Main_Pipeline_Headless/` (own git repo). This repo holds the experiments, notebooks and reports that decided what went into it.

Pipeline in one line: photo -> north-up 224 px canvas at a map level -> edge-direction cells -> FFT ZNCC over a 5.5 km disc -> 50 peaks -> DINOv2 re-order -> 5 places -> LoFTR (MINIMA weights) + RANSAC inliers -> answer if at least 8 inliers, else skip; 6 queries (headings -5/0/+5 x scales 1.0/1.05), second pass, altered copies.

## 2. Find it fast

| I want to... | open |
|---|---|
| know the current state, results, decisions, open tasks | `docs/HANDOFF_2026-11-17.md` (addenda 1-5 at the bottom are the newest), then `CHANGELOG.md` |
| know what every test folder is | `tests/README.md`, then the folder's own `README.md` |
| see the latest experiments (2026-11-16 to 18) | `tests/Test_Iteration_12/README.md` and `tests/Test_Iteration_12/results/README.md` |
| read an evaluation report in the standard layout | root `Stage_Evaluation.ipynb`; executed copies in `tests/Test_Iteration_12/results/stageeval/` |
| run the pipeline on photos | `python run.py` (see `README.md`); one photo step by step: `Single_Image_Run.ipynb` |
| change a setting | `src/header/settings.py` (research code); `../Main_Pipeline_Headless/settings.py` (submission code, the one that counts) |
| find a function of the pipeline | `src/README.md` (section map), each `src/<section>/README.md` |
| find a training or experiment block | `train/README.md`, then `train/<part>/README.md` |
| know where the data is and what each run needs | `dataset/README.md`, section 4 below |
| know the design rules (notebook is the face, atomic blocks) | `DESIGN.md`; code style `../HUMAN_CODE_STYLE.md` |
| reach the GPU server and its rules | section 6 below, `docs/HANDOFF_2026-11-17.md` section 1 |
| know what was deleted or moved in earlier tidies | `docs/TIDY_2026-11-15.md` |
| see Fable's prompts and answers | `docs/FABLE_PROMPT_*.md`; answers are on Notion (section 7) |
| read the whole story of 16 to 18 November in one document (figures, tables, what failed) | `../Documents/Outputs/Progress_Report_v0.6_High_Altitude_Set_Matchers_and_Precision.pdf` |
| read the two Codex reviews of 2026-11-18 and how they were checked | `docs/CODEX_REVIEWS_2026-11-18.md` (full reports in `tests/Review_Highalt_20261118_codex/` and `tests/LoFTR_Inputs_20261118_codex/`) |

## 3. Layout

| path | what |
|---|---|
| `src/` | the research pipeline as atomic blocks, ONE copy only. Sections: `header/` (settings, scoring, readers), `image_processing/` (canvas, map levels, edge cells), `coarse_search/` (FFT ZNCC, peaks), `semantic_rerank/` (DINO), `fine_matching/` (matcher, RANSAC), `result_thresholding/` (answer or skip), `evaluation/` (simulated priors, scoring). Each has a `README.md` and `concepts/` teaching notes. Older text in `src/` still mentions SuperPoint/LightGlue; the submission uses LoFTR with MINIMA weights |
| `train/` | building blocks for experiments: `checks/` (rich dump, replay, perturbations, judges), `coarse_methods/`, `datasets/`, `features/`, `maps/`, `models/`, `simulation/` (drone-view simulator), `config.py`, `train_fusion.py`, `simulate.py` |
| `tests/` | unit tests (`pipeline/`, `train/`, `run_all.py`) and every experiment round, `Test_Iteration_1` to `12`, the root-notebook tables (`Single_Image_Run/`, `Stage_Evaluation/`, `Batch_Speed_Check/`, `Fine_Fit_Methods/`, `Parameter_Test/`, `Heading_Queries/`) and isolated reviews by another tool (`*_codex/`, deletable alone) |
| `docs/` | handoff, prompts to Fable, overnight and tidy logs (`docs/README.md`) |
| `tools/` | `run_capped.sh` (heavy jobs with a 6 GB / 8 core cap), `pack_dataset.py` |
| `third_party/` | vendored XoFTR code |
| `dataset/` | data, mostly NOT in git (section 4) |
| `dataset_temp/` | never uploaded |
| root `*.ipynb` | `Stage_Evaluation.ipynb` (the report notebook, driven by `SE_*` environment variables), `Single_Image_Run.ipynb`, `Pipeline_Visual_Check.ipynb`, `Fine_Fit_Methods.ipynb`, `Batch_Speed_Check.ipynb`, `Stage_Evaluation_Colab.ipynb` |
| root `*.py` | `run.py` (whole run), `run_fusion.py` (fusion demo), `single_photo_pipeline.py`, `agreement_query_pipeline.py`, `heading_query_pipeline.py`, `inflation_guard_pipeline.py` (pipeline classes the notebooks import) |

## 4. Data

| data | where | notes |
|---|---|---|
| old official set ("official"): 500 thermal photos, Mumbai area, 96-351 m | `dataset/d1/` (in git); a packed copy is `dataset_temp/d1.npz` | maps for it: `dataset/Dataset_Official_2/` (two scenes, 11.7 GB each, the "new base map") or `dataset/small_sat_area/` (old map) |
| NUDT Thermal-UAV (Changsha), valid split 1425 frames, 197-320 m | `../Code_Testing/dataset/NUDT/`; here only `dataset/NUDT/checkpoint/` (model weights) | needs `NUDT_DIR` or the search order in `dataset/README.md` |
| new high-altitude set ("high-altitude", "new"): 500 photos, Rajasthan, 100-2017 m, 332 working + 168 LOCKED test | `dataset/Dataset_Official_3/` = hard links (no extra disk space) to `../Code_Testing/dataset/Dataset_Official_3/` (`d1.zip`, `d1/images`, `sat_imagery/*.zip`, 4.3-4.7 GB each) | not in git. The extracted working copy, the map store (levels 3-11) and the dev / locked-test photo folders are on the GPU server only (section 6). The split is `tests/Test_Iteration_12/results/highalt_split.csv` |
| model weights (MINIMA LoFTR, DINOv2) | `dataset/NUDT/checkpoint/` here; `../Main_Pipeline_Headless/models/` for the submission | |

Words that cause confusion: **"official"** always means the OLD set (Mumbai) in the results pages; the new set is called **"high-altitude"** or "new". **"Dataset_Official_2"** is the satellite imagery of the old set's area, not a photo set. Many big files in this workspace are hard links of one copy on disk; never edit one in place unless you mean to change all its names (see `../index.md`).

## 5. Rules that apply to every change (from the user)

- One `src/` copy. An experiment keeps only its changed files in its own `changed_src/`.
- Experiments live in executed notebooks, not in `run_*.py`. The notebook is the face; blocks do the work and can be swapped.
- Do not commit, push or overwrite unasked. Commit `3ecbbe0` (2026-11-17 14:10) holds the work up to the three Stage_Evaluation reports and Fable's prompts; iterations 24-28, the new notebooks/results of 2026-11-17 afternoon and the 2026-11-18 tidy are uncommitted.
- Check any gain on more than one data set (old official, NUDT, new) and give a logical reason; no one-off in-sample claims. Report wrong answers (beyond 8 sigma) next to every score. Write the kill rule before the run.
- The locked test split (168 photos of the new set) is not for tuning; it was used once by Fable and once by us (2026-11-18, frozen XoFTR run, report only), so it must not be used to choose anything any more.
- Heavy jobs on the laptop go through `tools/run_capped.sh`. Keep the Notion pages updated after every iteration (hub = current state, dated subpages, never overwrite old results).
- Detailed evaluations follow the layout of `Stage_Evaluation.ipynb`.

## 6. GPU server (shared)

`ssh workstation@100.105.160.107` (key login; append `2>&1 | grep -v "setlocale\|sign_and"`), WSL2, RTX 4000 Ada 20 GB. Our files live only on the E: drive, under `/mnt/e/IdeaForge/` (`work/` code and outputs, `datasets/highalt_v2/` the new set). Venv `~/training/ideaforge/venv-geoloc`. At most two GPU jobs of ours at once. Do not touch other people's files or processes on it. Layout and pitfalls: `docs/HANDOFF_2026-11-17.md` sections 1 and 7.

## 7. Notion

Workspace page `InterIIT_UAV_GeoLoc` (hub with the current state). Results page "Second-list ideas checked in the submission pipeline" holds every iteration (id `41a0a439-e483-81ff-92c5-cb8ad7dd2ffd`; iterations 24-28 are the coarse-stage analysis of 2026-11-17); handoff page id `41b0a439-e483-812a-bd05-f9591ed7eab7`; Fable's answers are subpages of the hub.

## 8. Current state in five lines (2026-11-18)

- Old official 52.69 / 60 base (53.60 with altered copies); NUDT 54.24 / 60 (saturated); new high-altitude 36.35 / 60 base.
- Below 800 m the new set loses points only to imprecision; above 800 m to skipped photos. The true place is in the coarse top 5 for 88-100 % of photos up to 1600 m, so it is a fine-matching failure there (features of about 1 pixel), plus coarse failure above 1600 m.
- A coarse z threshold does not transfer between the three sets; coarse answer plus matcher support of at least 4 inliers gives 0 wrong on all three but rescues only 13 % of skipped photos above 800 m (iteration 26). Averaging the coarse peaks does nothing (27). A parabola plus ZNCC two levels finer improves the coarse position (28). Coarser-scale structural channels are weaker than the edge search at height; their agreement with the edge answer is safe (0 wrong, three sets) but covers only 9 % above 800 m (29). Larger edge scales do not take over at height (sigma 1 px best from 500 m up) and a different blur for photo and map does not help (30, 31). The best lead for LoFTR above 800 m: draw the canvas at the photo's native resolution (two levels finer) and clean photo and tile; at least 8 true-place inliers 40 % to 51 %, first gain above 1400 m (34, diagnostic only). In the headless pipeline it gives 35.64 to 37.79 of 60 on the 332 working photos, 0 wrong (35); code is in `../Main_Pipeline_Headless`, off by default. XoFTR at the same 224 px input and cost finds the true place more often than LoFTR above 800 m (62 % against 46 %, bar per matcher from wrong-place counts), but was worse on the old official set: a swap must be height-gated (36). In the headless pipeline XoFTR from 800 m gives 35.64 to 39.76 of 60 on the 332 working photos with 0 wrong, the best change so far (37); on the LOCKED test split (run once, report only) 38.00 to 40.61 with 0 wrong (38). The locked split is now used up: no selection with it. Precision of accepted answers is mostly limited by noise in the attitude and height metadata (a synthetic audit reproduces the real error with 1 degree of attitude noise; a pose fit from the matches does not recover it): the remaining levers are skipped photos and NUDT look-alike answers (40, 41).
- Open: put the refined coarse fallback into the headless pipeline and run the 332 working photos; extra coarse channels; levels 12-13 for 2000-3000 m; Jetson timing; a new freeze with `DECLARATIONS.md`; one final run on the locked split.
- Full list with numbers: `docs/HANDOFF_2026-11-17.md`.
