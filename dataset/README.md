# Where the data goes

Git holds only the small files. The big ones (maps, model weights) are not in the repo: get them from the team Drive and put them in the folders below, **inside the `dataset/` folder next to this file**. After that nothing needs configuring: the code looks for these exact paths.

```
dataset/
├── d1/images/                          IN GIT   500 official thermal photos + images/metadata.csv
├── Dataset_Official_2/                 NOT IN GIT, ~24 GB   the new base map (two satellite scenes)
│   ├── 268234931/BAND.tif                       scene A, covers all the d1 photos (11.7 GB)
│   └── 268234921/BAND.tif                       scene B, reaches only the south of them (11.7 GB)
├── small_sat_area/                     NOT IN GIT, 2.5 GB   the old base map
│   └── PNEO3_acq20260114_dela3b31007.tif        (any single .tif in this folder is used)
└── NUDT/                               NOT IN GIT
    ├── checkpoint/                              model weights, 2.6 GB (needed for every run)
    │   ├── MINIMA/   minima_lightglue.pth, minima_loftr.ckpt, minima_roma.pth, minima_eloftr.ckpt, minima_xoftr.ckpt
    │   ├── RoMa/     roma_outdoor.pth, dinov2_vitl14_pretrain.pth
    │   ├── DINO/     dinov2_vits14_timm.safetensors      (the semantic layer; without it the code tries to download it)
    │   └── CAMP/     weights_0.9446_for_U1652.pth
    ├── Reference_map/changsha/ref.tif           only for the Changsha (DATA = "nudt") test photos and for training
    └── Thermal-UAV/                             only for the Changsha test photos and for training
```

## What each run needs

| You run | Needs |
|---|---|
| `Single_Image_Run.ipynb`, `Stage_Evaluation.ipynb`, `tests/Parameter_Test`, `tests/Heading_Queries`, `DATA = "official"`, `MAP = "new"` | `d1/` (in git), `Dataset_Official_2/` (both scenes), `NUDT/checkpoint/MINIMA/minima_loftr.ckpt`, `NUDT/checkpoint/DINO/` |
| same with `MAP = "old"` | `small_sat_area/` instead of `Dataset_Official_2/` |
| `MATCHER = "lightglue"` | `MINIMA/minima_lightglue.pth` |
| `MATCHER = "roma_minima"` | `MINIMA/minima_roma.pth` and `RoMa/dinov2_vitl14_pretrain.pth` |
| `DATA = "nudt"` and the training code in `train/` | the whole `NUDT/` folder including `Reference_map/` and `Thermal-UAV/` |

The code also needs the `NUDT/` folder to contain `Reference_map` and `Thermal-UAV` before it accepts it as "the NUDT folder" (`train/config.py: find_nudt_dir`). If those two are not on your disk, keep the full NUDT folder somewhere else and point to it:
`export NUDT_DIR=/path/to/NUDT` (the MINIMA / RoMa weights are then read from `$NUDT_DIR/checkpoint`; the DINO weights are always read from `dataset/NUDT/checkpoint/DINO/`). Searched in this order: `$NUDT_DIR`, `<repo>/dataset/NUDT`, `<repo>/../Code_Testing/dataset/NUDT`, `<repo>/../dataset/NUDT`.

## Choosing the map and the photos

In the notebooks: `DATA` ("official" or "nudt"), `MAP` ("new" = both scenes, "new_931", "new_921", "old"), `N_PHOTOS`. To use another map file, edit the one file `src/header/map_source.py` (add an entry to `MAPS`). The first run on the new map cuts a window around the photos out of the 11.7 GB scenes, which takes a few minutes. `MAP_MARGIN_M = 2500` makes that window smaller and faster.

## Files that are NOT in git but are needed for `COARSE = "fused"`

The learned fusion models (`*.pt`) are covered by `.gitignore`, so a fresh clone does **not** have them. They live in `tests/Test_Iteration_1/results/overnight/models/` (about 340 KB in total: `f4.pt`, `ranker.pt`, `fusion/*.pt`). Copy that `models/` folder from a teammate or the Drive. Without it `COARSE = "fused"` stops with "the fusion model needs the learned features ... does not exist"; `COARSE = "edge_zncc"` works without it (and was about as good in our tests).

## Setting up the Python environment

See `requirements.txt` at the top (Python 3.10, torch 2.8.0 from the PyTorch index first, then `pip install -r requirements.txt`). Check the install with `python run.py --tests`. Two notebook kernels on one small GPU can run out of memory: shut the old kernel down first.

## Checking that the data is in the right place

From the repo folder:

```
python - <<'E'
import sys; sys.path.insert(0, "src")
from pathlib import Path
here = Path("dataset")
need = ["d1/images/metadata.csv",
        "Dataset_Official_2/268234931/BAND.tif", "Dataset_Official_2/268234921/BAND.tif",
        "NUDT/checkpoint/MINIMA/minima_loftr.ckpt", "NUDT/checkpoint/DINO/dinov2_vits14_timm.safetensors"]
for name in need:
    print("ok     " if (here / name).exists() else "MISSING", name)
E
```
