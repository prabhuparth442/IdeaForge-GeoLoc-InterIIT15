from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
WORK_DIR = DATASET_DIR / "d1"
IMAGES_DIR = WORK_DIR / "images"
MAP_DIR = DATASET_DIR / "small_sat_area"
MINIMA_FILE = DATASET_DIR / "NUDT" / "checkpoint" / "MINIMA" / "minima_lightglue.pth"
MINIMA_URL = "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_PHOTOS = 500
USE_DINO = False
USE_MINIMA = True
SEED = 7

BAND_EDGES = [100, 150, 250, 351]
FLIGHT_NAMES = ["100-150 m", "150-250 m", "250-350 m"]

MAP_BANDS = (1, 2, 3)
MAP_GSD = 0.3
TILE = 224
LEVELS = [3, 4, 5, 6, 7]
STRIP_ROWS = 2048

GEOMETRY = {
    "attitude": True,
    "pitch_sign": -1,
    "roll_sign": -1,
    "yaw_offset": 180.0,
    "scale": 1.04,
    "height_offset": 0.0,
}

CELL = 4
N_BINS = 8

DISC_RADIUS = 5000
MIN_ENERGY = 0.05
MIN_VARIANCE = 0.005
NUM_PEAKS = 25
TILE_STEP = TILE // 2
DINO_BETA = 0.5

CONTEXT_MARGIN = 0
USE_BORDER_MASK = False
CTX = TILE + 2 * CONTEXT_MARGIN
MATCH_SIZE = 2 * CTX
RANSAC_PIXELS = 3.0
MAX_SHIFT_CELLS = 3

MIN_INLIERS = 12
ANSWER_MIN_INLIERS = 12
