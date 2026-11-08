"""All the tunable values of the pipeline, in one place.

Every value has: what it does, the range / options that make sense, and what we use now (the value written here).
"Measured" notes come from our development runs on the official 500-photo set (see README.md, "Where the numbers come from").
Change a value here, or for one run only:  ./run batch ... --set K_PEAKS=100 --set HEADINGS=[-5,0,5]
The code reads these as settings.NAME at run time, so an override really changes the run.
check_settings() in geoloc/check_settings.py rejects values outside the ranges written below.
"""
import os

# ---------------------------------------------------------------------------
# 1. Run mode and folders
# ---------------------------------------------------------------------------

# "answer": only the answer for each photo (the RESULT line / results.csv). Nothing else is computed or written.
# "full":   the same answers plus time per stage, inliers, zoom level, chosen heading and scale, GPU memory
#           (and error in metres + points when the metadata has the true position). Goes to details.csv and stderr.
# Options: "answer" | "full".  Override: --mode full
MODE = "answer"

# Where things are. In the evaluation container these are /data/..., on a PC use the environment variables
# (GEOLOC_MAP, GEOLOC_MODELS, GEOLOC_PREPROCESS, GEOLOC_DATASET) or the --map / --models ... options of ./run.
HERE = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.environ.get("GEOLOC_MAP", "/data/map")                       # the GeoTIFF scenes (read by preprocess only)
MODELS_DIR = os.environ.get("GEOLOC_MODELS", "/data/models")              # minima_loftr.ckpt and dinov2_vits14_timm.safetensors
PREPROCESS_DIR = os.environ.get("GEOLOC_PREPROCESS", "/data/preprocess")  # the map store that preprocess writes and serve reads
DATASET_DIR = os.environ.get("GEOLOC_DATASET", "/data/dataset")           # images/ and metadata.csv

LOFTR_FILE = "minima_loftr.ckpt"                 # LoFTR with the MINIMA weights (Apache-2.0), inside MODELS_DIR
DINO_FILE = "dinov2_vits14_timm.safetensors"     # DINOv2 ViT-S/14 (Apache-2.0), inside MODELS_DIR

# ---------------------------------------------------------------------------
# 2. Map store (built once by ./preprocess, so serve starts fast)
# ---------------------------------------------------------------------------

MAP_GSD = 0.3            # metres per pixel of the working map. The scenes (about 0.28 m) are resampled to it. 0.25 to 0.4 sensible; the zoom levels below assume 0.3
LEVELS = [3, 4, 5, 6, 7]  # zoom levels kept. Level L has a pixel of MAP_GSD * 2^(L/2) metres (3: 0.85 m ... 7: 3.4 m).
#                          A photo uses the level whose map window matches its footprint; heights 100 to 350 m need 3..7.
#                          Dropping a level saves memory but photos that need it are lost. Allowed values: whole numbers 0 to 10.
STRIP_ROWS = 2048        # map rows read at a time while building the store. Lower = less RAM while building (512 to 4096)
MAP_REGION = None        # None = the whole map. Or (west, south, east, north) in degrees to build a small store for a test
#                          (override: ./preprocess --region 73.0,19.0,73.2,19.2)

# ---------------------------------------------------------------------------
# 3. Photo to map: the projection (how the thermal photo is drawn north-up on the canvas)
# ---------------------------------------------------------------------------

TILE = 224               # canvas side in pixels. Must be a multiple of 56 (cell 4 and DINO patch 14). Do not change without re-checking everything
GEOMETRY = {
    "attitude": True,        # True: use pitch and roll from the metadata to tilt the drawing. False: draw as if looking straight down
    "pitch_sign": -1,        # -1 or 1: sign convention of the pitch column (found on the development set)
    "roll_sign": -1,         # -1 or 1: sign convention of the roll column
    "yaw_offset": 180.0,     # degrees added to the yaw column. -180 to 180. Found on the development set (the camera looks backwards)
    "scale": 1.04,           # multiplies the flight height before drawing. 0.8 to 1.3. 1.04 fits the development set best
    "height_offset": 0.0,    # metres added to the flight height. -30 to 30
}

# ---------------------------------------------------------------------------
# 4. Heading and scale queries (the photo is drawn several times, the best candidate over all drawings wins)
# ---------------------------------------------------------------------------

# Which set of drawings to use. Options: "standard" | "all" | "custom".  Override: --set QUERY_SET='"all"'
#   standard: 3 headings (-5, 0, 5) x scales (1.0, 1.05) = 6 queries, about 0.7 s each on the dev GPU
#   all:      9 headings (-10 to 10 step 2.5) x scales (1.0, 1.05) = 18 queries, about 3x the time of standard
#   custom:   HEADINGS x SCALE_FACTORS below
QUERY_SET = "standard"

# Early stop: after each query the pooled best candidate is looked at; when it already has at least this many inliers the remaining queries are skipped.
# It depends only on the photo, not on the clock. None = never stop early (run every query). 15 to 60 sensible; lower = faster but more risk of a changed answer.
EARLY_STOP_INLIERS = None

# Second pass: when the first pass ends in no_match, the photo is tried again with more drawings and all candidates (first and second pass) are pooled.
# The second pass uses the 12 queries of the "all" set (9 headings x 1.0 / 1.05) that the first pass did not use, so a photo that needs it ends up with the same
# 18 queries as QUERY_SET = "all". Photos that were answered in the first pass cost nothing extra.
# Measured on 120 photos: standard 46.6 / 60 at 4.0 s per photo, all 50.6 at 11.7 s, two passes see README for the result.
# Options: True | False.  Ignored when QUERY_SET is "all" (nothing left to add) or "custom".
SECOND_PASS = True

# Headings: degrees added to the yaw, one drawing each. The metadata yaw is off by up to 10 degrees for some photos and the coarse search
# only tolerates about 3, so we try several. Must contain 0.0, values between -15 and 15.
# Measured (one scale): 1 heading 41.3 / 60, 3 headings (-5, 0, 5) 46.5, 9 headings (-10 to 10 step 2.5) 49.7.
HEADINGS = [-5.0, 0.0, 5.0]               # used when QUERY_SET = "custom"

# Scale factors: the projection scale is multiplied by each, one drawing per combination. Must contain 1.0.
# Measured: only scales above 1 help (1.05 yes, 0.95 no). 3 headings x (1.0, 1.05) scored 49.5 / 60.
SCALE_FACTORS = [1.0, 1.05]               # used when QUERY_SET = "custom"

# ---------------------------------------------------------------------------
# 5. Coarse search (whole 5.5 km disc, edge-direction cells, FFT correlation)
# ---------------------------------------------------------------------------

DISC_RADIUS = 5500       # metres searched around the hint point. The hint is within 5000 m of the truth; the extra 500 m covers the disc edge. 5000 to 7000
CELL = 4                 # pixels per edge-direction cell. Fixed (the stored map features use it)
N_BINS = 8               # edge directions per cell. Fixed (the stored map features use it)
MIN_ENERGY = 0.05        # a position is ignored when the map there has less edge energy than this (per cell). 0.01 to 0.2. Higher drops more empty areas (water, fields)
MIN_VARIANCE = 0.005     # a position is ignored when the map there is too flat for a correlation. 0.001 to 0.02
K_PEAKS = 50             # K: places the coarse search keeps for the DINO layer. 10 to 200. Recall rises slowly with K (0.84 at 10, 0.89 at 50); cost is DINO only
M_FINE = 10              # M: places that go to the fine matcher, per query. 1 to K_PEAKS. 5 was as good as 50 in a sweep; fine matching is the slow part (about 0.05 s per place)

# ---------------------------------------------------------------------------
# 6. DINO layer (DINOv2 ViT-S/14 re-orders the K places before fine matching)
# ---------------------------------------------------------------------------

USE_DINO = True          # True / False. Measured: the right place becomes first for 90 % of photos instead of 76 %
DINO_WEIGHT = 100.0      # order inside the K places = z(coarse score) + DINO_WEIGHT * z(DINO score). 0 = coarse order only, 100 = DINO decides. 0 to 100; 3 to 100 measured equal
DINO_LAYER = 8           # transformer block whose tokens are compared. 4 to 12. Fixed after testing
DINO_BATCH = 25          # places sent through DINO at once. Lower if the GPU runs out of memory (5 to 50)

# ---------------------------------------------------------------------------
# 7. Fine matching (LoFTR with the MINIMA weights, then RANSAC)
# ---------------------------------------------------------------------------

LOFTR_CONFIDENCE = 0.21       # matches below this confidence are dropped before RANSAC. 0.2 to 0.5 (LoFTR itself drops below its coarse threshold).
#                               It changes how many inliers a place gets, so MIN_INLIERS must be chosen together with it. A higher value gives fewer inliers
LOFTR_COARSE_THRESHOLD = 0.21  # LoFTR's own cut on its coarse matching probability. 0.1 to 0.5. Lower = more, weaker matches (0.1 produced false answers)
LOFTR_BATCH = 1               # candidate places sent through LoFTR together. 1 = one at a time (the tested way). 2 to 10: faster on a GPU, numbers can differ slightly
RANSAC_PIXELS = 3.0           # a match is an inlier when it is within this many canvas pixels of the fitted similarity. 1 to 4 (4 and above let a wrong answer in)
MAX_SHIFT_CELLS = 3           # the position correction from RANSAC may move the answer at most this many cells (12 pixels) from the candidate. 1 to 6.
#                               A larger shift is treated as a failed match (0 inliers)
RANSAC_MAX_ITERS = 3000       # RANSAC tries per candidate. 500 to 10000
RANSAC_MAX_SCALE_CHANGE = 0.15   # the fitted transform must keep the scale within 1 +- this. 0.05 to 0.3
RANSAC_MAX_ROTATION = 12.0       # ... and the rotation within +- this many degrees. 5 to 30

# ---------------------------------------------------------------------------
# 8. Decision (which candidate wins, answer or no_match)
# ---------------------------------------------------------------------------

GAMMA = 1.0              # the winner has the highest  inliers + GAMMA * (DINO z-score). 0 = inliers only. 0 to 4 (no effect measured on one query; used across queries)
MIN_INLIERS = 8          # answer only when the winner has at least this many inliers, otherwise no_match. 4 to 16
#                          Measured with LOFTR_CONFIDENCE 0.21: 8 is the lowest bar with no wrong answer on one query. With several queries a bar-level
#                          answer (8 inliers) was wrong once in 273 photos. A wrong answer costs -20, a skipped photo 0, a right one +3 to +10.

# ---------------------------------------------------------------------------
# 9. Repeatability
# ---------------------------------------------------------------------------

SEED = 7                 # seed of OpenCV's RANSAC, set before every query so the same photo always gives the same answer
CONFIDENCE_FULL_AT = 30  # only for the optional confidence column: inliers / this, cut at 1. Not scored. 30 inliers is the median of correct places
