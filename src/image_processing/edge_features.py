import cv2
import numpy as np

from header.settings import CELL, N_BINS


def gradient_parts(image):
    assert image.ndim == 2 and min(image.shape) >= 3, "image must be 2-D and at least 3 x 3, got shape %s" % (image.shape,)
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    gx = np.zeros_like(blurred)
    gy = np.zeros_like(blurred)
    gx[:, 1:-1] = blurred[:, 2:] - blurred[:, :-2]
    gy[1:-1, :] = blurred[2:, :] - blurred[:-2, :]
    strength = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) % np.pi
    bin_id = np.minimum((angle / np.pi * N_BINS).astype(int), N_BINS - 1)
    return blurred, gx, gy, strength, angle, bin_id


def edge_features(image):
    assert image.ndim == 2 and min(image.shape) >= CELL, "image must be 2-D and at least CELL (%d) pixels each way, got %s" % (CELL, image.shape)
    blurred, gx, gy, strength, angle, bin_id = gradient_parts(image)
    rows = image.shape[0] // CELL
    cols = image.shape[1] // CELL
    features = np.zeros((N_BINS, rows, cols), dtype=np.float32)
    for b in range(N_BINS):
        weight = (strength * (bin_id == b))[:rows * CELL, :cols * CELL]
        features[b] = weight.reshape(rows, CELL, cols, CELL).sum(axis=(1, 3))
    return features


def normalise_cells(features, cells_to_use=None):
    assert features.ndim == 3 and features.shape[0] == N_BINS, "features must be (N_BINS, rows, cols), got %s" % (features.shape,)
    assert cells_to_use is None or cells_to_use.shape == features.shape[1:], "cells_to_use must be (rows, cols)"
    length = np.sqrt((features ** 2).sum(axis=0))
    if cells_to_use is None:
        cells_to_use = length > 0
    if cells_to_use.any():
        typical = np.percentile(length[cells_to_use], 30)
    else:
        typical = 1.0
    return features / (length + typical + 1e-3)


def query_edge_features(canvas, valid):
    assert canvas.ndim == 2 and canvas.shape[0] == canvas.shape[1], "canvas must be square, got %s" % (canvas.shape,)
    assert canvas.shape[0] % CELL == 0, "canvas side must be a multiple of CELL (%d)" % CELL
    assert valid.shape == canvas.shape and valid.dtype == bool, "valid must be a bool array shaped like canvas"
    features = edge_features(canvas)
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8))
    side = canvas.shape[0] // CELL
    cell_valid = safe.reshape(side, CELL, side, CELL).mean(axis=(1, 3)) >= 0.8
    features = normalise_cells(features, cell_valid) * cell_valid
    return features, cell_valid


def no_data_cells(image):
    assert image.ndim == 2, "image must be 2-D"
    rows = image.shape[0] // CELL
    cols = image.shape[1] // CELL
    empty = (image[:rows * CELL, :cols * CELL] == 0).reshape(rows, CELL, cols, CELL).any(axis=(1, 3))
    grown = cv2.dilate(empty.astype(np.uint8), np.ones((3, 3), np.uint8))
    return grown > 0


def map_edge_features(image):
    assert image.ndim == 2 and image.dtype == np.uint8, "level image must be 2-D uint8, got %s %s" % (image.shape, image.dtype)
    assert (image > 0).any(), "level image has no map pixels at all"
    rows = image.shape[0]
    strip = 1024
    pad = 8
    pieces = []
    for top in range(0, rows, strip):
        first = max(top - pad, 0)
        last = min(top + strip + pad, rows)
        part = edge_features(image[first:last] / 255)
        skip = (top - first) // CELL
        pieces.append(part[:, skip:skip + strip // CELL])
    features = np.concatenate(pieces, axis=1)

    border = no_data_cells(image)[:features.shape[1], :features.shape[2]]
    features[:, border] = 0

    length = np.sqrt((features ** 2).sum(axis=0))
    typical = np.percentile(length[length > 0][::50], 30)
    return features / (length + typical + 1e-3)


def build_map_features(level_images):
    assert len(level_images) > 0, "no level images given"
    map_features = {}
    for level, image in level_images.items():
        map_features[level] = map_edge_features(image)
    return map_features
