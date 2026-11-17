# Test_Iteration_12: the high-altitude data set, the mega evaluation and the GPU server (2026-11-16 to 17)

Everything measured on the new high-altitude photos (Rajasthan, 500 photos, 100-2017 m), the three-set Stage_Evaluation reports, dataset similarity, and the second-list ideas run through the submission pipeline (`../../../Main_Pipeline_Headless`). Full state, decisions and open tasks: `../../docs/HANDOFF_2026-11-17.md`.

## Notebooks (`notebooks/`)

| notebook | what |
|---|---|
| `Mega_Dump.ipynb` | writes the 143-query raw-match dump per photo (env: KIND, TAG, PHOTOS_FILE, MAX_PHOTOS), shuffled order |
| `Mega_Replay_Figures.ipynb`, `Mega_Replay_dev60_executed.ipynb` | offline replay of those dumps: query maps, bar sweep, gamma/M, height bands, time guard (figures 1-9). A complement to Stage_Evaluation, not the report |
| `Dataset_Similarity.ipynb` (+ `_executed`) | old official vs new photos: DINOv2 classifier AUC, vignette, edges, dark spots |
| `HighAlt_Truth_Diagnostic.ipynb` | Fable's one-query truth diagnostic (coarse and LoFTR with the true place known) |

The main report is the repo-root `Stage_Evaluation.ipynb`, driven by `SE_*` environment variables; its three executed copies are in `results/stageeval/` (`StageEval_highalt_dev332`, `_official150`, `_nudt_valid150`, each `.ipynb` + `.html` + `tables_*`).

## Results (`results/`)

| path | what |
|---|---|
| `stageeval/` | the three Stage_Evaluation reports (102 MB) |
| `figures/`, `fig*.csv` | replay figures 1-9 and their tables |
| `similarity/` | dataset similarity figures 1-6 (fig6: dark spots against the map) |
| `highalt_split.csv`, `highalt_dev_bands.csv` | 332 working / 168 locked test split and the per-band scores |
| `ids_official150.csv`, `ids_nudt_valid100.csv` | photo ids used for the 150/100-photo runs |
| `server/` | replays copied back from the GPU server (interim, final, official_final, nudt, official3) |
| `rich_dump/`, `highalt_truth_diag/`, `flat_field_look.png` | local dumps, Fable's diagnostic outputs, the vignette-removal look |

## Scripts (`scripts/`)

- `server_chains/`: the shell chains that ran on the GPU server (download, store build, dumps, Stage_Evaluation, dino, refine). They assume the server layout in the handoff doc (`/mnt/e/IdeaForge/...`).
- `analysis/`: small analysis scripts (`band_cmp.py`, `avg_test.py`, `spots_look.py`, `ff_look.py`).
- `highalt_truth_diag/`: Fable's map/store builders and runners (own README).
- `tick.sh`, `overnight_supervisor.sh`, `overnight_wrapper.sh`: the 15-minute status signal and the overnight supervisor (no longer running).

## Rules that came out of this round

- At most two GPU jobs of ours at once on the shared server; nothing next to a 143-query dump.
- The locked test split (168 photos) was used once by Fable; do not tune on it.
- Report wrong answers next to every score; every gain is checked on old, NUDT and new sets.
