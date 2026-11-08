"""Coarse search: slide the photo's edge cells over the whole search disc with an FFT correlation (ZNCC), then pick the best places.

score_map_for_query()  one query canvas in, a score map over the disc out (values -1..1, -1 = position not usable)
pick_peaks()           the best `count` separated places of a score map, as rows  [score, x_m, y_m, sharpness, rank]
"""
import math

import numpy as np
import torch

import settings
from geoloc.edge_features import query_edge_features
from geoloc.levels import level_gsd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fast_length(n):
    # the next size that the FFT handles quickly (only factors 2, 3 and 5)
    while True:
        rest = n
        for factor in (2, 3, 5):
            while rest % factor == 0:
                rest = rest // factor
        if rest == 1:
            return n
        n = n + 1


def zncc_everywhere(window, kernel, cells):
    """Masked zero-mean normalised correlation of the kernel at every position of the window (all (bins, rows, cols))."""
    n_bins, height, width = window.shape
    kernel_h, kernel_w = kernel.shape[1], kernel.shape[2]
    size = (fast_length(height), fast_length(width))
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    n_cells = cells.sum()

    window_fft = torch.fft.rfft2(window, s=size)
    cells_fft = torch.fft.rfft2(cells, s=size)

    strength_fft = torch.fft.rfft2((window ** 2).sum(dim=0), s=size)
    energy = torch.fft.irfft2(strength_fft * cells_fft.conj(), s=size)[:out_h, :out_w].clamp_min(0)

    kernel_mean = kernel.sum(dim=(1, 2), keepdim=True) / n_cells
    kernel_zero = (kernel - kernel_mean) * cells
    zero_fft = torch.fft.rfft2(kernel_zero, s=size)
    dot_zero = torch.fft.irfft2((window_fft * zero_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]

    squared_sums = torch.zeros_like(energy)
    for b in range(n_bins):
        local_sum = torch.fft.irfft2(window_fft[b] * cells_fft.conj(), s=size)[:out_h, :out_w]
        squared_sums += local_sum ** 2 / n_cells
    variance = (energy - squared_sums).clamp_min(0)
    zncc = dot_zero / (kernel_zero.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)

    no_map = energy < settings.MIN_ENERGY * n_cells
    too_flat = variance < settings.MIN_VARIANCE * n_cells
    zncc[no_map | too_flat] = -1.0
    return zncc


def cut_map_features(features, top, left, rows, cols):
    """Window of the map cells, zeros outside the map."""
    piece = np.zeros((settings.N_BINS, rows, cols), dtype=np.float32)
    row0 = max(top, 0)
    row1 = min(top + rows, features.shape[1])
    col0 = max(left, 0)
    col1 = min(left + cols, features.shape[2])
    if row1 > row0 and col1 > col0:
        piece[:, row0 - top:row1 - top, col0 - left:col1 - left] = features[:, row0:row1, col0:col1]
    return piece


def score_map_for_query(canvas, valid, level, prior_m, map_features):
    """Returns the score map and a geometry dict (level, side, x_m, y_m, inside) that says where each map cell is."""
    cell = settings.CELL
    gsd = level_gsd(level)
    query_features, query_cells = query_edge_features(canvas, valid)
    if query_cells.sum() == 0:
        raise ValueError("the photo covers no usable cell of the canvas")
    side = query_features.shape[1]
    reach = settings.DISC_RADIUS / gsd + side * cell / 2
    cells_wide = int(math.ceil(2 * reach / cell))
    top = int(round((prior_m[1] / gsd - reach) / cell))
    left = int(round((prior_m[0] / gsd - reach) / cell))

    window = torch.from_numpy(cut_map_features(map_features[level], top, left, cells_wide, cells_wide)).to(DEVICE)
    kernel = torch.from_numpy(query_features).to(DEVICE)
    cells = torch.from_numpy(query_cells.astype(np.float32)).to(DEVICE)
    zncc = zncc_everywhere(window, kernel, cells)

    row_ids = torch.arange(zncc.shape[0], device=DEVICE)
    col_ids = torch.arange(zncc.shape[1], device=DEVICE)
    y_m = ((top + row_ids) * cell + side * cell / 2) * gsd
    x_m = ((left + col_ids) * cell + side * cell / 2) * gsd
    dx = x_m[None, :] - float(prior_m[0])
    dy = y_m[:, None] - float(prior_m[1])
    inside = dx ** 2 + dy ** 2 <= settings.DISC_RADIUS ** 2
    geometry = dict(level=level, side=side, x_m=x_m, y_m=y_m, inside=inside)

    usable = (zncc > -1) & inside
    score_map = torch.where(usable, zncc, torch.full_like(zncc, -1.0))
    return score_map, geometry


def pick_peaks(score_map, geometry, count):
    """The best `count` places, each blanked out around before the next is taken. Missing places are filled with x = y = 1e9."""
    usable = (score_map > -1) & geometry["inside"]
    ranking = score_map.clone()
    ranking[~usable] = -1e9

    gap = max(settings.TILE // settings.CELL, geometry["side"] // 2)
    last_row = score_map.shape[0] - 1
    last_col = score_map.shape[1] - 1
    peaks = []
    for _ in range(count):
        best_value, best_flat = ranking.reshape(-1).max(0)
        if float(best_value) < -1e8:
            break
        row = int(best_flat) // score_map.shape[1]
        col = int(best_flat) % score_map.shape[1]
        score = float(score_map[row, col])
        around = (float(score_map[max(row - 1, 0), col]) + float(score_map[min(row + 1, last_row), col])
                  + float(score_map[row, max(col - 1, 0)]) + float(score_map[row, min(col + 1, last_col)])) / 4
        peaks.append((score, float(geometry["x_m"][col]), float(geometry["y_m"][row]), score - around, float(best_value)))
        ranking[max(row - gap, 0):row + gap + 1, max(col - gap, 0):col + gap + 1] = -1e9

    while len(peaks) < count:
        peaks.append((0.0, 1e9, 1e9, 0.0, -1.0))
    return np.array(peaks)
