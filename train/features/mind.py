"""F1: MIND, local self-similarity. Each pixel is described by how much its patch resembles the patches 8 steps away.
Works the same on thermal and optical pictures because it never compares brightness between the two modalities, only inside one picture."""
import cv2
import numpy as np

from header.settings import CELL, N_BINS
from image_processing.edge_features import no_data_cells
from train.features.base import Family

MIND_OFFSETS = [(0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]
assert len(MIND_OFFSETS) == N_BINS, "one MIND channel per cell channel"


# ------------------------------------------------------------------ F1: MIND

def mind_pixels(image):
    """image: 2-D float in [0, 1]  ->  (8, H, W) values in (0, 1]: exp(-patch distance to the neighbour / local variance)."""
    assert image.ndim == 2 and min(image.shape) >= 8, "image must be 2-D and at least 8x8, got %s" % (image.shape,)
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    distances = []
    for dy, dx in MIND_OFFSETS:
        shifted = np.roll(blurred, (dy, dx), axis=(0, 1))
        distances.append(cv2.boxFilter((blurred - shifted) ** 2, -1, (5, 5)))
    distances = np.stack(distances)
    variance = np.maximum(distances.mean(axis=0), 1e-3)
    return np.exp(-distances / variance).astype(np.float32)


def pool_cells(pixels):
    """(C, H, W) -> (C, H//CELL, W//CELL) by averaging each CELL x CELL block."""
    assert pixels.ndim == 3, "pixels must be (channels, H, W)"
    channels, height, width = pixels.shape
    rows, cols = height // CELL, width // CELL
    assert rows > 0 and cols > 0, "picture smaller than one cell"
    cut = pixels[:, :rows * CELL, :cols * CELL]
    return cut.reshape(channels, rows, CELL, cols, CELL).mean(axis=(2, 4)).astype(np.float32)


def mind_query_features(canvas, valid):
    assert canvas.ndim == 2 and canvas.shape[0] == canvas.shape[1] and canvas.shape[0] % CELL == 0, "canvas must be square, side a multiple of CELL"
    assert valid.shape == canvas.shape and valid.dtype == bool, "valid must be a bool array shaped like canvas"
    features = pool_cells(mind_pixels(canvas))
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8))
    side = canvas.shape[0] // CELL
    cell_valid = safe.reshape(side, CELL, side, CELL).mean(axis=(1, 3)) >= 0.8
    return features * cell_valid, cell_valid


def mind_map_features(level_image, strip=1024, pad=8):
    """Same strip-by-strip walk as image_processing.edge_features.map_edge_features, so memory stays small."""
    assert level_image.ndim == 2 and level_image.dtype == np.uint8, "level image must be 2-D uint8"
    pieces = []
    for top in range(0, level_image.shape[0], strip):
        first = max(top - pad, 0)
        last = min(top + strip + pad, level_image.shape[0])
        part = pool_cells(mind_pixels(level_image[first:last] / 255.0))
        skip = (top - first) // CELL
        pieces.append(part[:, skip:skip + strip // CELL])
    features = np.concatenate(pieces, axis=1)
    border = no_data_cells(level_image)[:features.shape[1], :features.shape[2]]
    features[:, border] = 0
    return features


class MindFamily(Family):
    name = "mind"

    def __init__(self, store, levels):
        super().__init__()
        self.features = {level: mind_map_features(store["levels"][level]) for level in levels}

    def map_features(self, level):
        return self.features[level]

    def query(self, canvas, valid):
        return mind_query_features(canvas, valid)
