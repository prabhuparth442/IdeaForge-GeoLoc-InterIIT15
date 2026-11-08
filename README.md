# Main_Pipeline_Headless

UAV visual geo-localisation for the ideaForge problem (Inter-IIT Tech Meet 15.0): a thermal photo from a drone with jammed GPS, a hint point
within 5 km of the truth, and a satellite map. The pipeline finds the drone's position on the map, or says `no_match`.

This folder is the **submittable code**: no notebooks, no experiments, only the parts that gave a measured gain. It can be mounted as `/data/code`
of the evaluation (`run` and `preprocess` are in its top level, as `interface/SPEC.md` asks).

## What the pipeline does

```
photo + metadata
  1. draw the photo north-up on a 224x224 canvas (camera model, pose, zoom level from the flight height)   geoloc/photo.py
     repeat steps 2-4 for each heading / scale query (default 3 headings x 2 scales = 6 queries)          geoloc/queries.py
  2. coarse search: edge-direction cells, FFT correlation over the whole 5.5 km disc, 50 best places      geoloc/coarse_search.py
  3. DINO layer: DINOv2 ViT-S/14 re-orders the 50 places, the best 10 go on                              geoloc/semantic_rerank.py
  4. fine matching: LoFTR (MINIMA weights) + RANSAC on each of the 10 places -> inliers and a shift       geoloc/fine_matching.py
  5. pool the candidates of all queries, pick the one with most inliers, answer if inliers >= 8           geoloc/decision.py
  -> RESULT,<filename>,match,<lat>,<lon>,<confidence>   or   RESULT,<filename>,no_match,<best guess lat>,<lon>,
```

Why each part is here (gain measured on the 500 official development photos, score out of 60, details in "Where the numbers come from"):

| part | what it adds |
|---|---|
| whole-disc edge ZNCC correlation | the right place is first for 76 % of photos, in the first 50 for 90 % |
| DINO layer | the right place first for 90 % instead of 76 % |
| LoFTR + MINIMA weights | far more inliers on the right place than the older SuperPoint+LightGlue matcher (median 30 vs 0-4 on wrong places) |
| heading + scale queries | 41.3 -> 49.5 / 60 (6 queries); the right place is among the candidates for 99.6 % of photos instead of 87 % |

Left out on purpose: the learned fusion models and MIND/learned feature maps (trained on NUDT, no gain after the DINO layer), SuperPoint + LightGlue
(research-only licence), RoMa (never used), the heading penalty and the 0-inlier rescue rule (no measured gain). There is no switch to bring
any of them back; only publicly usable code and weights are in this folder.

## Layout

```
run                 ./run serve | batch | preprocess      the entry point of the evaluation
preprocess          same as ./run preprocess              the optional preprocessing step of the evaluation
settings.py         every tunable value, with range and options (read this to tune)
geoloc/             the pipeline, one small file per step
models/             minima_loftr.ckpt, dinov2_vits14_timm.safetensors (not in git, see models/README.md)
requirements.txt    Python packages for a PC / dev environment
docker/             Dockerfile of the team image for the Jetson (see docker/README.md)
```

## Install

Python 3.10, CUDA 12.6 builds of torch 2.8.0 and torchvision 0.23.0, then:

```
pip install -r requirements.txt
```

On the Jetson use the starter kit's Dockerfile for torch (see `Submission_Files/TeamKit/starter/`) and add `requirements.txt` to its team image.
Put the two weight files into `models/` (see `models/README.md`).

## How to use it

Three commands. Folders default to `/data/map`, `/data/models`, `/data/preprocess`, `/data/dataset` (the evaluation container). On a PC pass them as
options, or set `GEOLOC_MAP`, `GEOLOC_MODELS`, `GEOLOC_PREPROCESS`, `GEOLOC_DATASET`.

### 1. Build the map store (once per map)

```
./preprocess --map /path/to/map --preprocess /path/to/store
```

Reads every GeoTIFF under the map folder, writes the map at 5 zoom levels and its edge cells to the store folder, and a log
`preprocess.log` next to them. Takes minutes to an hour for two 11 GB scenes and needs a few GB of RAM and about 5 GB of disk.
For a quick test on a small part of the map: `--region west,south,east,north` (degrees).
Where the scenes overlap, the first file in alphabetical order wins.

### 2a. Answer a folder of photos (batch)

```
./run batch --dataset dev_data/dataset --preprocess /path/to/store --models models --out results
```

Writes `results/results.csv` with the columns `filename,status,lat,lon,confidence` (the format of the evaluation). Useful options:
`--limit 50`, `--first 100`, `--mode full`, `--set NAME=VALUE`, and for the development set `--hint-jitter-m 5000`
(the development metadata gives the true position as the hint; the real evaluation hint is up to 5 km away, this option imitates that).

### 2b. The evaluation protocol (serve)

```
./run serve            # the evaluation starts it as /data/code/run serve
```

Loads the map store and both networks, warms the GPU, prints `READY`, then reads the CSV header and one metadata row per photo from stdin and
prints one `RESULT,...` line per photo. Logs go to stderr. A failing photo is answered `no_match`; the process does not stop.
Try it with the starter kit's harness: `python3 harness/evaluate.py --image <image> --code Main_Pipeline_Headless --models Main_Pipeline_Headless/models --preprocess <store> --map <map> --dataset <dataset>`.

### The two modes

| | `answer` (default) | `full` |
|---|---|---|
| how | `MODE = "answer"` in `settings.py` or `--mode answer` | `MODE = "full"` or `--mode full` |
| output | only the answer for each photo | the same answers, plus the extras below |
| `serve` | `RESULT` lines only | `RESULT` lines on stdout + one `INFO` line per photo on stderr |
| `batch` | `results.csv` | `results.csv` + `details.csv` + a summary on stderr |
| extras | none, and no timing code runs | seconds per stage (photo, render, coarse, semantic, fine, decision), total seconds, zoom level, inliers, winning heading and scale, DINO z-score, peak GPU memory, and when the metadata has the true position: error in metres, sigma, points and the score out of 60 |

`answer` is the mode to submit. `full` synchronises the GPU before every timing, so it is a little slower than `answer`.

## Tuning

Everything tunable is in `settings.py`. Each value has a comment with what it does, the allowed range or options, and what we measured.
To try a value without editing the file: `--set K_PEAKS=100`, `--set HEADINGS=[-5,0,5]` (any Python literal). Values outside their range stop the run
with a clear message (`geoloc/check_settings.py`). The groups:

| group | values | effect |
|---|---|---|
| run mode and folders | `MODE`, `MAP_DIR` ... `DATASET_DIR`, weight file names | where things are, answer or full |
| map store | `MAP_GSD`, `LEVELS`, `STRIP_ROWS`, `MAP_REGION` | needs `./preprocess` again when changed |
| projection | `GEOMETRY` (attitude, signs, yaw offset, scale, height offset) | found on the development set; change only with a new map or camera |
| queries | `HEADINGS`, `SCALE_FACTORS` | biggest lever: more queries = more recall = more time |
| coarse search | `DISC_RADIUS`, `MIN_ENERGY`, `MIN_VARIANCE`, `K_PEAKS`, `M_FINE` | recall of the right place, and fine-stage time |
| DINO layer | `USE_DINO`, `DINO_WEIGHT`, `DINO_LAYER`, `DINO_BATCH` | order of the places |
| fine matching | `LOFTR_CONFIDENCE`, `LOFTR_COARSE_THRESHOLD`, `RANSAC_*`, `MAX_SHIFT_CELLS` | how many inliers a place gets |
| decision | `GAMMA`, `MIN_INLIERS` | answer or no_match; a wrong answer costs -20, a skipped photo 0 |

`MIN_INLIERS` and `LOFTR_CONFIDENCE` belong together: the inlier counts change with the confidence, so change the bar when you change the confidence.

## Time and memory

On the development laptop GPU (RTX 3050 6 GB) one query takes about 0.7 s, so 6 queries are about 4.2 s per photo. **The Jetson Orin NX time has not been
measured.** The limit is 30 s per photo (a later answer counts as `no_match`). If it is too slow there: fewer headings or scales (`HEADINGS`, `SCALE_FACTORS`),
smaller `M_FINE` (5 was as good as 50 in a sweep) or `K_PEAKS`. The map store is opened memory-mapped, only the windows in use are read from disk.

## Rules of the evaluation that the code follows

- Each photo is answered on its own: no state is kept between photos except the loaded map store and networks. The random seed of RANSAC is set before every query.
- No network access: all weights are local files, nothing is downloaded.
- Only `/tmp` would be written during serve (nothing is written at all in the default mode).

## Where the numbers come from

All from the development runs of 6-7 Nov 2026 on the 500 official photos (Cartosat-3 map, Mumbai area), in `../IdeaForge-GeoLoc-InterIIT15/tests/`:
`Heading_Queries/Analysis_2026-11-07_Heading_Queries_and_Parameter_Tests.md` (queries, parameter sweeps) and
`Parameter_Test/Analysis_2026-11-06_Stage_Evaluation_and_Strategies.md` (DINO layer, funnel). The default settings are the ones of the best run
(3 headings x scales 1.0 and 1.05, K 50, M 10, confidence 0.21, bar 8). Caveats stay as in those reports: the settings were looked at on the same photos,
there is one map and one matcher, and the Jetson speed is not measured.
That code lives in `../IdeaForge-GeoLoc-InterIIT15/` (the research repo, with notebooks and all experiments). This folder is a clean copy of the winning path only;
changes to the winning path need to be made here too.
