# Training the fusion network

Files: `src/coarse_search/fusion.py` (models, z-scores, fused search), `train/train_fusion.py` (loss, training, bootstrap),
`train/simulate.py` (data with a known rule), `run_fusion.py` (main run file), `tests/train/test_fusion.py`.

Quick check, no dataset: `python run_fusion.py demo`.

## What one training example is
One photo, with candidate positions: the true one (index 0) plus hard negatives (the strongest wrong peaks of every map) and random
disc positions. For each candidate: the robust z-score under each map (`z`), and per-photo `context` (peak height of each map, gap to
its runner-up, ground pixel size). Loss: the truth must outscore the other candidates (softmax over the list).
Only these small candidate sets are stored, never the 6.8 M-position maps.

## Is the SCC/NUDT data enough?
Independent examples = photos (~8k train, 1.4k valid). The models have 2 to a few hundred parameters, so yes for the fusion itself.
What the numbers do not cover:
- Domain shift: NUDT is Changsha thermal against a 0.3 m map; the contest is Cartosat panchromatic. Linear and monotone models transfer
  best; the gated one must be checked on the official 500 before trusting it.
- Flights are correlated: always split by flight (done in `split_by_group`).
- Use the NUDT valid split to pick the model, the official 500 only to report.

## Still to build (needs your data)
A NUDT reader and camera adapter (1280x1024, focal ~1563 px, pitch -90/roll 180 conventions) so the existing pipeline can produce
`cands.npz` (z at candidates via `correlation_maps_multi` + `z_maps_and_context`). Then `python run_fusion.py train --data cands.npz`.

## Building blocks for the experiments (iteration 1)

Small single-purpose modules, used by the stages in `tests/Test_Iteration_1/stages/`. Each can be imported and used alone.

| package | module | what it does |
|---|---|---|
| `train/config.py` | | every number and path of the experiments (result folders, photo counts, data finders) |
| `train/features/` | `base.py`, `edge.py`, `mind.py`, `whitening.py` | feature families: edge cells, MIND self-similarity, pre-whitening |
| | `learned_net.py`, `learned_family.py` | learned features: the two small CNNs and their training, and the family used at search time |
| `train/maps/` | `map_bank.py` | many score maps from one photo on one grid (`compute_maps`), z-scores (`z_stack`), peaks (`nms_peaks`) |
| `train/datasets/` | `nudt_data.py` | NUDT photos, camera and map as pipeline items |
| | `candidates.py` | one small training example per photo (truth + hard negatives), written as shards |
| | `pseudo_thermal.py`, `place_picker.py`, `synthetic_candidates.py` | fake thermal photos from satellite crops, where to cut them, and their candidate sets |
| | `smoke_world.py` | a tiny synthetic world for quick checks |
| `train/models/` | `fusion_ablation.py`, `ranker.py` | train linear / monotone / gated fusion on map subsets; a small listwise ranker |
| `train/checks/` | `full_eval.py`, `overnight_report.py`, `synthetic_report.py` | full-map evaluation with paired bootstrap, and the report writers |
| `train/train_fusion.py`, `simulate.py` | | the fusion training loop and a simulated data set with a known rule |
