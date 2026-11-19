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

# Optional changes of the second pass. None for all three = the frozen behaviour above (9 headings x scales 1.0 / 1.05, bar MIN_INLIERS).
#   SECOND_PASS_HEADINGS / SECOND_PASS_SCALES: the grid of the second pass (queries already run in the first pass are not repeated). Headings -15 to 15, scales 0.5 to 1.5.
#     Research repo: escalating to headings -10..+10 x scales 0.95 / 1.0 / 1.05 gave 49.0 -> 53.5 on 379 official photos (with USE_AGREEMENT); escalating the 6 standard queries
#     to headings -5/0/5 x scales 0.80 / 0.85 / 0.90 / 1.0 / 1.05 kept the clean score (54.8) and recovered about 70 % of the loss when the height metadata was 20 % too high
#     (18.8 -> 43.6, 60 photos, 0 wrong); not checked on a second photo set, not run on the Jetson.
#   SECOND_PASS_MIN_INLIERS: the bar for a photo that needed the second pass (more queries give more chance hits). None = MIN_INLIERS. The research repo used 11.
SECOND_PASS_HEADINGS = None       # None or a list, for example [-5.0, 0.0, 5.0]
SECOND_PASS_SCALES = None         # None or a list, for example [0.8, 0.85, 0.9, 1.0, 1.05]
SECOND_PASS_MIN_INLIERS = None    # None or a number of at least MIN_INLIERS (up to 30)

# Time budget per photo, in seconds. The evaluation counts an answer later than 30 s as no_match, so a photo that is still working at that point earns nothing.
# Before each query (not the first one) the elapsed time of the photo plus the slowest query so far is compared with this number; when it does not fit, the
# remaining queries are skipped and the photo is answered with the candidates found so far (an answer only if it has enough inliers, otherwise no_match).
# The second pass is cut the same way. None = no limit. 5 to 29. 22 leaves about 8 s for the slowest single query, the decision and the I/O.
TIME_BUDGET_S = 22.0

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
M_FINE = 5               # M: places that go to the fine matcher, per query. 1 to K_PEAKS. 5 was as good as 50 in a sweep; fine matching is the slow part (about 0.05 s per place)
FFT_BACKEND = "auto"   # "auto", "torch" or "matmul". auto = use cuFFT, and switch to our matrix-multiply FFT if cuFFT fails (it fails on some Jetsons). Same results either way, matmul is slower

# ---------------------------------------------------------------------------
# 6. DINO layer (DINOv2 ViT-S/14 re-orders the K places before fine matching)
# ---------------------------------------------------------------------------

USE_DINO = True          # True / False. Measured: the right place becomes first for 90 % of photos instead of 76 %
DINO_WEIGHT = 100.0      # order inside the K places = z(coarse score) + DINO_WEIGHT * z(DINO score). 0 = coarse order only, 100 = DINO decides. 0 to 100; 3 to 100 measured equal
DINO_ONLY_BELOW_M = None   # None = DINO re-orders the K places of every photo. A number = photos taken at this height or higher skip DINO (the first M places in coarse order go on).
#                            Why: above about 800 m the disc holds fewer than about 70 independent places, the coarse top place is already right for about 0.9 of the photos and DINO adds
#                            nothing (332 high-altitude photos: score 34.52 without against 34.57 with DINO, DINO alone as judge lowers the first place from 0.92 to 0.88) while it costs about a quarter of each query.
DINO_LAYER = 8           # transformer block whose tokens are compared. 4 to 12. Fixed after testing
DINO_BATCH = 25          # places sent through DINO at once. Lower if the GPU runs out of memory (5 to 50)

# ---------------------------------------------------------------------------
# 7. Fine matching (LoFTR with the MINIMA weights, then RANSAC)
# ---------------------------------------------------------------------------

LOFTR_CONFIDENCE = 0.21       # matches below this confidence are dropped before RANSAC. 0.2 to 0.5 (LoFTR itself drops below its coarse threshold).
#                               It changes how many inliers a place gets, so MIN_INLIERS must be chosen together with it. A higher value gives fewer inliers
LOFTR_COARSE_THRESHOLD = 0.21  # LoFTR's own cut on its coarse matching probability. 0.1 to 0.5. Lower = more, weaker matches (0.1 produced false answers)
LOFTR_REUSE_QUERY = True      # True | False. ON for the Jetson test (2026-11-15). True = the photo canvas goes through LoFTR's CNN once per query instead of once per candidate place (M_FINE times).
#                               Same matches in a CPU test (tests/test_query_reuse.py, exact) and 1.6x faster fine stage in the research repo's CPU test; not yet run on the Jetson.
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

# Query agreement (geoloc/agreement.py). Off = the decision above only (what was frozen on 2026-11-10). On = the first pass (the standard queries) also answers
# when at least AGREE_MIN_QUERIES of its queries each found the same place, within AGREE_RADIUS_SIGMA sigma of the winner, each with at least
# AGREE_QUERY_INLIERS inliers on its own best candidate. The second pass is still decided by the bar only (queries 2.5 degrees apart see the same picture).
# Measured offline on stored candidates (research repo): +2 to +4.5 marks of 60 with 0 wrong on three sets; not yet run on the Jetson; ON in this version for the Jetson test.
USE_AGREEMENT = True         # True | False. ON for the Jetson test (2026-11-15); set False for the behaviour frozen on 2026-11-10
AGREE_MIN_QUERIES = 2        # 2 to 4 (a query always agrees with itself)
AGREE_RADIUS_SIGMA = 1.0     # 0.5 to 2
AGREE_QUERY_INLIERS = 3      # 2 to 6

# Bar by map level. At the high levels (a photo from 700 m or more) the matcher finds few matches per photo, so a right place has few inliers while a wrong one
# has almost none (on the 500 high-altitude photos: wrong best places at levels 10 and 11 never had more than 3 inliers). None = one bar (MIN_INLIERS) for every level.
# A dict {level: bar} replaces the bar for the photos of that level, for example {9: 4, 10: 4, 11: 4}. Checked offline only on the high-altitude set (in-sample), see tests.
BAR_BY_LEVEL = None       # None | dict of level -> bar (1 to 30)

# Photo cleaning before the stretch (geoloc/flat_field.py). All off by default = the photo goes in as it is.
# FLAT_FIELD_FROM_HEIGHT_M: None = never. A number = photos taken at this height or higher get the steps below (0 = every photo). The reason (Fable, 2026-11-17, measured on the
#   500 high-altitude photos): above about 800 m half of the stretched range is lens vignetting that the map does not have, and edges are 70 times rarer than at low height.
# FLAT_FIELD_DEGREE: degree of the radial polynomial that is subtracted (0 = no vignette removal, 2 to 4 sensible).
# LOCAL_NORM_SIGMA: None, or a Gaussian size in photo pixels: grey minus local mean divided by local spread (5 to 30).
# PHOTO_MASK_RECTS: None, or a list of (x0, y0, x1, y1) frame rectangles filled with the median grey (a fixed artefact of one camera, for example the dark corner hook of the
#   high-altitude camera at x 303-312, y 225-255). Applies to every photo, whatever the height.
FLAT_FIELD_FROM_HEIGHT_M = None   # None | height in metres
FLAT_FIELD_DEGREE = 3             # 0 to 6
LOCAL_NORM_SIGMA = None           # None | pixels
PHOTO_MASK_RECTS = None           # None | [(x0, y0, x1, y1), ...]

# Native-resolution fine matching (geoloc/native_fine.py, iteration 34). All off by default = LoFTR gets the 224 px canvas of the coarse level as before.
# NATIVE_FINE_FROM_HEIGHT_M: None = never. A number = photos taken at this height or higher get a LoFTR canvas drawn NATIVE_FINE_LEVELS levels finer (448 x 448 px for 2), with the map tile at that level,
#   photo cleaned (vignette removal of degree FLAT_FIELD_DEGREE + local normalisation) and tile locally normalised. The coarse search is not changed. Needs that finer level in the map store.
#   Measured only as true-place inliers on the high-altitude development photos (at least 8 inliers above 800 m: 40 % to 51 %); LoFTR costs about 4 times as much per place on the 448 px canvas.
# NATIVE_FINE_LEVELS: 2 (the only tested value; the canvas side doubles). NATIVE_FINE_LOCAL_NORM_PX: local normalisation size in PHOTO pixels (the map tile gets the same size in its own pixels, at least 3).
NATIVE_FINE_FROM_HEIGHT_M = None   # None | height in metres
NATIVE_FINE_LEVELS = 2             # 2
NATIVE_FINE_LOCAL_NORM_PX = 15.0   # 5 to 30

# XoFTR in place of LoFTR from a height (geoloc/xoftr_matcher.py, iteration 36). Off by default.
# PRESET (off by default = None): a named configuration from presets/, applied before any --set. "pv_x" = the frozen photo view + pose fit + XoFTR configuration (iterations 43 to 47);
#   "pv_x_r2" = the same with two render-match-fit rounds and wider priors, robust to worse metadata (red team, 2026-11-19; see tests/Test_Iteration_12/results/photo_view/red_team/ROBUSTNESS.md in the research repo).
#   Needs minima_xoftr.ckpt in MODELS_DIR and, to match the measurements, a map store with levels 3 to 11 (./run preprocess --set "LEVELS=[3,4,5,6,7,8,9,10,11]"). Run time is about twice the base (2 to 4 s per photo on the server).
PRESET = None                      # None | "pv_x" | "pv_x_r2"
# XOFTR_FROM_HEIGHT_M: None = LoFTR for every photo. A number = photos taken at this height or higher use XoFTR (MINIMA weights, file XOFTR_FILE in MODELS_DIR) for the fine stage; the rest use LoFTR.
#   Measured only as true-place recall on the high-altitude development photos (62 % against 46 % above 800 m with a bar per matcher); worse than LoFTR on the low-height official set, so it needs a gate.
# XOFTR_COARSE_THRESHOLD / XOFTR_FINE_THRESHOLD: XoFTR's own cuts (MINIMA defaults 0.3 and 0.1). XOFTR_MIN_CONFIDENCE: matches below it are dropped before RANSAC (0 = keep all XoFTR kept).
XOFTR_FROM_HEIGHT_M = None         # None | height in metres
XOFTR_FILE = "minima_xoftr.ckpt"    # inside MODELS_DIR
XOFTR_COARSE_THRESHOLD = 0.3       # 0.1 to 0.5
XOFTR_FINE_THRESHOLD = 0.1         # 0.05 to 0.5
XOFTR_MIN_CONFIDENCE = 0.0         # 0 to 0.5

# Final fit one or more levels finer (geoloc/pipeline.py, refine_finer). 0 = off (the position of the winning candidate is the answer). 1 or 2 = after a photo is answered, the photo is drawn
# again at this many levels finer (a pixel of 0.3 * 2^(L/2) m shrinks by 1.4 per level) around the answer with the heading and scale of the winning query, matched once more, and the answer moves
# by the new shift when the new fit has at least REFINE_MIN_INLIERS inliers and moves the answer by at most REFINE_MAX_SHIFT_SIGMA sigma; otherwise the answer stays. Why: from about 800 m the
# error of an answer is 2-3 canvas pixels, as large as 1 sigma (about 2 canvas pixels), so half of the answers lie in the 1-2 sigma band (6 points instead of 10). Costs one more matcher call.
REFINE_FINER_LEVELS = 0      # 0 | 1 | 2
REFINE_MIN_INLIERS = 8       # 4 to 30
REFINE_MAX_SHIFT_SIGMA = 2.0  # 0.5 to 4
# REFINE_NATIVE (iteration 40, off by default): True = the final fit uses the matcher that answered the photo (XoFTR from XOFTR_FROM_HEIGHT_M, otherwise LoFTR) and, instead of a 224 px canvas
#   that covers less ground at the finer level, a canvas of the same ground footprint (448 px for 2 levels) with the photo cleaned (vignette + local normalisation) and the tile locally normalised,
#   as in the native-resolution path. Needs REFINE_FINER_LEVELS = 2. False = the old final fit (224 px canvas, LoFTR or the answering matcher).
# TILE_CENTRE_FIX (iteration 40, off by default): True = a shift found by the fine stage is added to the true centre of the map tile (its position is rounded to whole map pixels), not to the
#   unrounded candidate place. Removes an error of up to half a map pixel (up to 6.8 m at level 11) that no fit can see.
REFINE_NATIVE = False         # True | False
TILE_CENTRE_FIX = False       # True | False

# Photo view (iteration 43, off by default): the fine stage for high photos with the photo left as it is (320 x 256, cleaned) and the MAP drawn into the photo's own camera view (geoloc/photo_view.py).
# PHOTO_VIEW_FROM_HEIGHT_M: None = the canvas path for every photo. A number = photos at this height or higher use the photo view in the fine stage (their altered copies, native canvas, fallback and final refit are off).
# PHOTO_VIEW_MIN_INLIERS: the bar for these photos. It is higher than MIN_INLIERS because the photo view gives wrong places more inliers (XoFTR wrong-place counts reached 20 on the development photos; the true place has a median of 77). 8 to 60.
# PHOTO_VIEW_LOCAL_NORM_PX: local normalisation size in photo pixels, for the photo and the drawn map. PHOTO_VIEW_MIN_MAP_SHARE: a candidate is skipped when less than this share of the drawn view has map.
PHOTO_VIEW_FROM_HEIGHT_M = None   # None | height in metres
PHOTO_VIEW_MIN_INLIERS = 20       # 8 to 60
PHOTO_VIEW_LOCAL_NORM_PX = 15.0   # 5 to 30
PHOTO_VIEW_MIN_MAP_SHARE = 0.5    # 0.2 to 1
# PHOTO_VIEW_FIT_POSE (iteration 44, off by default): True = the answer's position comes from a fit of the camera pose to the winner's matches instead of the median of the match votes (geoloc/photo_view.py fit_pose).
#   Every match says which ground point a photo pixel sees; the fit moves position, pitch, roll, yaw and height so that all pixels land on them, with the metadata's noise as a prior (the noise the synthetic audit found
#   to reproduce the real error): PHOTO_VIEW_PRIOR_TILT_DEG for pitch and roll, PHOTO_VIEW_PRIOR_YAW_DEG for yaw, PHOTO_VIEW_PRIOR_HEIGHT as a share of the height. The fit is dropped (the median vote stays) when it moves the
#   answer by more than PHOTO_VIEW_FIT_MAX_MOVE_SIGMA sigma from the median vote. Needs PHOTO_VIEW_FROM_HEIGHT_M.
PHOTO_VIEW_FIT_POSE = False
PHOTO_VIEW_PRIOR_TILT_DEG = 1.0     # 0.3 to 4
PHOTO_VIEW_PRIOR_YAW_DEG = 2.0      # 0.5 to 8
PHOTO_VIEW_PRIOR_HEIGHT = 0.02      # 0.005 to 0.1
PHOTO_VIEW_FIT_MAX_MOVE_SIGMA = 3.0 # 0.5 to 8
# PHOTO_VIEW_FIT_ROUNDS (iteration 45, off by default = 1): rounds of render - match - fit. 2 = after the first fit the map is drawn again with the corrected pose at the fitted position, matched again and the pose fitted again
#   (the second round starts from a better-aligned picture); the second round is dropped when it has fewer than PHOTO_VIEW_MIN_INLIERS inliers or moves the answer more than 1.5 sigma. 1 to 3.
PHOTO_VIEW_FIT_ROUNDS = 1
# PHOTO_VIEW_VERIFY_BELOW (iteration 47, off by default = None): a photo whose winner has fewer inliers than this and uses the photo view gets a second opinion: the OTHER matcher (LoFTR when XoFTR answered, XoFTR otherwise)
#   matches the same place in the photo view; its inlier count and how far its position is from the answer are logged (verify_inliers, verify_distance_m). PHOTO_VIEW_VERIFY_ACT = True: the photo is also skipped (no_match) when the
#   other matcher has fewer than PHOTO_VIEW_VERIFY_MIN inliers or lands more than 2 sigma from the answer. False = log only.
PHOTO_VIEW_VERIFY_BELOW = None     # None | inliers
PHOTO_VIEW_VERIFY_ACT = False      # True | False
PHOTO_VIEW_VERIFY_MIN = 8          # 1 to 60
PHOTO_VIEW_VERIFY_MAX_HEIGHT_M = None   # None | metres: only photos at or below this height get the second opinion (LoFTR is weak at height, where it often finds nothing at a right place)

# Second-matcher fallback (iteration 42, off by default). FALLBACK_MATCHER = True: when a query's fine stage does not reach MIN_INLIERS with the matcher of the photo (XoFTR from XOFTR_FROM_HEIGHT_M, LoFTR otherwise),
#   the OTHER matcher tries the same candidates once, and every candidate keeps the higher inlier count (and its shift). Costs one more fine call only for queries that fail.
#   Needs the XoFTR weights. FALLBACK_NATIVE_FROM_HEIGHT_M: None, or a height from which the fallback matcher gets the native-resolution canvas (NATIVE_FINE_LEVELS finer, photo and tile cleaned).
#   Kill rule written before the first run (2026-11-18): useful only if the score rises on at least two of the three sets (new working photos, old official, NUDT headless), the wrong answers (beyond 8 sigma)
#   rise by at most 1 per 300 photos on any set, and the mean seconds per photo stay under twice the base.
FALLBACK_MATCHER = False             # True | False
FALLBACK_NATIVE_FROM_HEIGHT_M = None # None | height in metres

# Inflation guard (geoloc/agreement.py). Off = nothing changes. On = the bar and the agreement floor are multiplied by how much stronger the wrong places
# have become on this run than on the development photos (running mean over the photos of the process); inside the dead zone the factor is 1 (no change on clean data).
USE_INFLATION_GUARD = False  # True | False
GUARD_PRIOR = 0.169          # mean inliers of a wrong place on the development photos, from the standard queries. 0.1 to 0.3
GUARD_PRIOR_WEIGHT = 30      # the prior counts like this many photos. 10 to 100
GUARD_DEAD_ZONE = 1.5        # act only when the running level is more than this many times the prior. 1.2 to 2

# Altered copies (geoloc/altered_copies.py). Off = nothing changes. On = a photo that is still not answered is run again on copies of itself (the polarity flipped,
# a sharpened one), each with the standard queries and the same decision as the first pass; the copy with the most inliers that answers is used.
# Research repo (Idea5b, 60 official photos, first pass only): inverted photos 39.2 -> 53.8 of 60, blurred 25.2 -> 38.2, clean 54.8 -> 56.4, no wrong answer from a copy.
# Costs 12 extra queries on a photo that is not answered, nothing on the others. Not yet run on the Jetson.
USE_ALTERED_COPIES = False       # True | False
ALTERED_COPY_KINDS = ["invert", "sharpen"]   # any of "invert", "sharpen"
ALTERED_COPIES_ORDER = "before_second_pass"  # "before_second_pass" (copies first, wide queries only if they fail) or "after_second_pass"
ALTERED_COPY_BAR_ONLY = True     # True | False. True = a copy answers only with MIN_INLIERS inliers (no agreement rule). Each copy is one more try, so one more chance for a
#                                  wrong place to look right (the same reason the second pass has its own bar). Research repo (Idea5b): with the agreement rule a NUDT photo that
#                                  no map place fits got two wrong copy answers (4 and 6 inliers, -20 each); on official photos 10 of 28 copy answers had under 8 inliers.

# Aligned-count rescue (geoloc/aligned_count.py). Off = nothing changes. On = a photo that would be no_match is answered at the candidate place whose weak LoFTR
# matches (down to ALIGNED_CONFIDENCE) sit at the same pixel in the photo and the map window far more often than chance, if that place is clearly better than every place
# more than ALIGNED_FAR_SIGMA sigma away. Needs LOFTR_COARSE_THRESHOLD <= ALIGNED_CONFIDENCE so that weak matches exist (the bar still reads matches at LOFTR_CONFIDENCE, same inliers).
# Research repo (Idea1c, 4 photo sets, stored candidates): official no_match photos +3.7 to +5.3 of 60, NUDT sets 0 to +0.5; the strict values below gave 0 extra wrong answers
# on all four, but they were chosen after seeing the held-out sets (the looser values gave 2 extra wrong answers on two NUDT sets). Not yet run on the Jetson.
USE_ALIGNED_RESCUE = False   # True | False
ALIGNED_CONFIDENCE = 0.05    # weakest match that is counted. 0.05 to 0.2
ALIGNED_TOLERANCE_M = 20.0   # two points count as the same pixel when they are this many metres apart or less. 4 to 20
ALIGNED_MIN_Z = 10.0         # the best place needs at least this much excess over chance. 2 to 45
ALIGNED_MARGIN = 8.0         # and every place farther away than ALIGNED_FAR_SIGMA sigma must be this much below it. 0 to 16
ALIGNED_FAR_SIGMA = 2.0      # 1 to 4

# ---------------------------------------------------------------------------
# 9. Repeatability
# ---------------------------------------------------------------------------

SEED = 7                 # seed of OpenCV's RANSAC, set before every query so the same photo always gives the same answer
CONFIDENCE_FULL_AT = 30  # only for the optional confidence column: inliers / this, cut at 1. Not scored. 30 inliers is the median of correct places
