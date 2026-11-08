"""Edge-direction cells: the picture is cut into 4x4 pixel cells and each cell holds 8 numbers, the edge strength in 8 directions.
The photo and the map are turned into the same kind of cells, so they can be compared by correlation."""
import cv2
import numpy as np

import settings


def edge_features(image):
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    gx = np.zeros_like(blurred)
    gy = np.zeros_like(blurred)
    gx[:, 1:-1] = blurred[:, 2:] - blurred[:, :-2]
    gy[1:-1, :] = blurred[2:, :] - blurred[:-2, :]
    strength = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) % np.pi
    bin_id = np.minimum((angle / np.pi * settings.N_BINS).astype(int), settings.N_BINS - 1)

    cell = settings.CELL
    rows = image.shape[0] // cell
    cols = image.shape[1] // cell
    features = np.zeros((settings.N_BINS, rows, cols), dtype=np.float32)
    for b in range(settings.N_BINS):
        weight = (strength * (bin_id == b))[:rows * cell, :cols * cell]
        features[b] = weight.reshape(rows, cell, cols, cell).sum(axis=(1, 3))
    return features


def normalise_cells(features, cells_to_use=None):
    length = np.sqrt((features ** 2).sum(axis=0))
    if cells_to_use is None:
        cells_to_use = length > 0
    if cells_to_use.any():
        typical = np.percentile(length[cells_to_use], 30)
    else:
        typical = 1.0
    return features / (length + typical + 1e-3)


def query_edge_features(canvas, valid):
    """Cells of the photo canvas and the mask of cells that are fully inside the photographed area."""
    cell = settings.CELL
    features = edge_features(canvas)
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8))
    side = canvas.shape[0] // cell
    cell_valid = safe.reshape(side, cell, side, cell).mean(axis=(1, 3)) >= 0.8
    features = normalise_cells(features, cell_valid) * cell_valid
    return features, cell_valid


def no_data_cells(image):
    cell = settings.CELL
    rows = image.shape[0] // cell
    cols = image.shape[1] // cell
    empty = (image[:rows * cell, :cols * cell] == 0).reshape(rows, cell, cols, cell).any(axis=(1, 3))
    grown = cv2.dilate(empty.astype(np.uint8), np.ones((3, 3), np.uint8))
    return grown > 0


def map_edge_features(image):
    """Cells of one whole map level picture (uint8, 0 = no map). Done in strips so it needs little memory."""
    cell = settings.CELL
    rows = image.shape[0]
    strip = 1024
    pad = 8
    pieces = []
    for top in range(0, rows, strip):
        first = max(top - pad, 0)
        last = min(top + strip + pad, rows)
        part = edge_features(image[first:last] / 255)
        skip = (top - first) // cell
        pieces.append(part[:, skip:skip + strip // cell])
    features = np.concatenate(pieces, axis=1)

    border = no_data_cells(image)[:features.shape[1], :features.shape[2]]
    features[:, border] = 0

    length = np.sqrt((features ** 2).sum(axis=0))
    typical = np.percentile(length[length > 0][::50], 30)
    return features / (length + typical + 1e-3)
