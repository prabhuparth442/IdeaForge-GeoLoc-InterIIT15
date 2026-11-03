import math
import time

import cv2
import numpy as np
import torch

from coarse_search.correlation import correlate_everywhere
from coarse_search.search import correlation_maps, pick_peaks
from header.settings import CELL, DEVICE, DISC_RADIUS, N_BINS
from image_processing.map_levels import level_gsd


def slide_slowly(window, kernel, cells):
    n_bins, height, width = window.shape
    kernel_h, kernel_w = kernel.shape[1], kernel.shape[2]
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    n_valid = cells.sum()
    cosine = np.zeros((out_h, out_w))
    zncc = np.zeros((out_h, out_w))

    kernel_mean = kernel.sum(axis=(1, 2), keepdims=True) / n_valid
    kernel_zero = (kernel - kernel_mean) * cells
    kernel_length = np.sqrt((kernel ** 2).sum())
    kernel_zero_length = np.sqrt((kernel_zero ** 2).sum())

    for row in range(out_h):
        for col in range(out_w):
            piece = window[:, row:row + kernel_h, col:col + kernel_w] * cells
            cosine[row, col] = (piece * kernel).sum() / (kernel_length * np.sqrt((piece ** 2).sum()) + 1e-9)
            piece_zero = (piece - piece.sum(axis=(1, 2), keepdims=True) / n_valid) * cells
            zncc[row, col] = (piece_zero * kernel_zero).sum() / (kernel_zero_length * np.sqrt((piece_zero ** 2).sum()) + 1e-9)
    return cosine, zncc


def fast_equals_slow():
    rng = np.random.default_rng(1)
    window = np.zeros((N_BINS, 48, 56), dtype=np.float32)
    for b in range(N_BINS):
        noise = rng.random((48, 56)).astype(np.float32)
        window[b] = cv2.GaussianBlur(noise, (0, 0), 1.2) ** 2
    window = window / window.max()

    row, col, size = 20, 31, 14
    yy, xx = np.mgrid[0:size, 0:size]
    cells = (((yy - size / 2 + 0.5) ** 2 + (xx - size / 2 + 0.5) ** 2) <= (size / 2) ** 2).astype(np.float32)
    kernel = window[:, row:row + size, col:col + size].copy()
    kernel = kernel + 0.15 * rng.random(kernel.shape).astype(np.float32)
    kernel = kernel * cells

    start = time.time()
    slow_cosine, slow_zncc = slide_slowly(window, kernel, cells)
    print("slow sliding took %.2f s (%d positions)" % (time.time() - start, slow_cosine.size))
    fast = correlate_everywhere(torch.from_numpy(window).to(DEVICE), torch.from_numpy(kernel).to(DEVICE), torch.from_numpy(cells).to(DEVICE))
    difference_cosine = np.abs(fast["cosine"].cpu().numpy() - slow_cosine).max()
    difference_zncc = np.abs(fast["zncc"].cpu().numpy() - slow_zncc).max()
    print("largest difference fast vs slow: cosine %.1e, zncc %.1e" % (difference_cosine, difference_zncc))
    assert difference_cosine < 1e-4 and difference_zncc < 1e-4, "the fast and the slow version disagree!"
    for name, heat in [("cosine", slow_cosine), ("zncc", slow_zncc)]:
        best = np.unravel_index(np.argmax(heat), heat.shape)
        assert best == (row, col), name + " did not find the hidden position"
    print("ok: identical, and both find the hidden position")


def map_tests(store, centre_m):
    map_features = store["features"]
    test_level = 4
    gsd = level_gsd(test_level)
    piece_rows = int(centre_m[1] / gsd / CELL) - 28
    piece_cols = int(centre_m[0] / gsd / CELL) - 28
    piece = map_features[test_level][:, piece_rows:piece_rows + 56, piece_cols:piece_cols + 56].copy()
    piece_valid = np.ones((56, 56), dtype=bool)
    true_x = (piece_cols * CELL + 56 * CELL / 2) * gsd
    true_y = (piece_rows * CELL + 56 * CELL / 2) * gsd

    maps, geometry = correlation_maps(map_features, test_level, piece, piece_valid, np.array([true_x + 900.0, true_y - 700.0]))
    for mode in ["cosine", "zncc"]:
        found = pick_peaks(maps[mode], geometry)
        error = math.hypot(found[0, 1] - true_x, found[0, 2] - true_y)
        print("test 1, %-6s: best score %.3f, error %.2f m" % (mode, found[0, 0], error))
        assert found[0, 0] > 0.99 and error < 1.0
    print("   ok")

    maps, geometry = correlation_maps(map_features, test_level, piece, piece_valid, np.array([-50000.0, -50000.0]))
    found = pick_peaks(maps["cosine"], geometry)
    print("test 2: peaks found far outside the map:", int((found[:, 1] < 1e8).sum()))
    assert (found[:, 1] > 1e8).all()
    print("   ok")

    maps, geometry = correlation_maps(map_features, test_level, piece, piece_valid, np.array([true_x, true_y]))
    inside_count = int(geometry["inside"].sum())
    expected = math.pi * (DISC_RADIUS / (CELL * gsd)) ** 2
    print("test 3: positions inside the circle: %d, area formula says %.0f" % (inside_count, expected))
    assert abs(inside_count - expected) / expected < 0.01
    print("   ok")
    del maps, geometry
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
