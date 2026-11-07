"""All the knobs of the iteration-1 experiments in one place. Change numbers here (or with the run_*.py flags), nothing else."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # the repo folder (IdeaForge-GeoLoc-InterIIT15)
ITERATION_DIR = ROOT / "tests" / "Test_Iteration_3_Thresholding"
OUT_DIR = ITERATION_DIR / "results" / "overnight"   # the run on real photos: logs, shards, models, reports
SYNTH_OUT_DIR = ITERATION_DIR / "results" / "synthetic"   # the run with pseudo-thermal photos

# ---- data ----
NUDT_LEVELS = [3, 4]                                # NUDT frames are 160-330 m high: only map levels 3 and 4 are ever used
N_TRAIN_PHOTOS = 3000                               # photos used to learn (spread evenly over the train flights)
N_VALID_PHOTOS = 800                                # photos used to choose between models
N_TEST_PHOTOS = 400                                 # photos for the final full-map report (never used for choosing)
GEOMETRY_PHOTOS = 24                                # photos for the yaw/scale check
SEED = 7

# ---- score maps ----
WHITEN_ALPHA = 0.5                                  # C1: strength of pre-whitening (0 = off, 1 = full)
MAP_NAMES_BASE = ["edge_zncc", "edge_cosine", "edge_mag", "edge_zncc_w", "mind_zncc"]   # learned_zncc is added once F4 is trained

# ---- candidate sets (what one training example looks like) ----
N_HARD_PER_MAP = 3                                  # strongest wrong peaks taken from every map
N_HARD_FUSED = 6                                    # and from the plain average of all maps
N_RANDOM = 48                                       # random wrong positions inside the disc
EXCLUDE_M = 45.0                                    # a candidate closer than this to the truth is not "wrong"
SHARD_SIZE = 50                                     # photos per saved file (a restart skips finished shards)

# ---- F4: learned features ----
F4_PAIRS = 3000                                     # training pairs (photo canvas + map piece around the truth)
F4_MAP_MARGIN_CELLS = 40                            # the learner must find the truth within +-40 cells (+-136 m at level 3)
F4_MINUTES = 90                                     # training time budget
F4_BATCH = 16
F4_LR = 2e-3
F4_TEMPERATURE = 0.05

# ---- fusion training ----
FUSION_EPOCHS = 400
FUSION_KINDS = ["linear", "monotone", "gated"]
BOOTSTRAP = 2000


def data_roots():
    """Folders that may hold the big data sets: this repo's dataset/, then the Code_Testing/dataset next to the repo."""
    return [ROOT / "dataset", ROOT.parent / "Code_Testing" / "dataset", ROOT.parent / "dataset"]


def find_nudt_dir():
    """The folder that holds Reference_map/ and Thermal-UAV/. Order: $NUDT_DIR, then NUDT inside each data root."""
    candidates = [Path(os.environ.get("NUDT_DIR", "x/none"))] if os.environ.get("NUDT_DIR") else []
    for root in data_roots():
        candidates.append(root / "NUDT")
    for path in candidates:
        if (path / "Reference_map").exists() and (path / "Thermal-UAV").exists():
            return path
    raise FileNotFoundError("NUDT folder not found. Set the environment variable NUDT_DIR to the folder that contains "
                            "Reference_map/ and Thermal-UAV/. Looked in: %s" % [str(c) for c in candidates])


def find_dev_dir():
    """The folder that holds d1/images/metadata.csv and small_sat_area/*.tif (the official development set). Order: $DEV_DATA_DIR, then each data root."""
    candidates = [Path(os.environ["DEV_DATA_DIR"])] if os.environ.get("DEV_DATA_DIR") else []
    candidates += data_roots()
    for path in candidates:
        if (path / "d1" / "images" / "metadata.csv").exists() and any((path / "small_sat_area").glob("*.tif")):
            return path
    raise FileNotFoundError("development set not found (needs d1/images/metadata.csv and small_sat_area/*.tif). Set DEV_DATA_DIR. Looked in: %s" % [str(c) for c in candidates])
