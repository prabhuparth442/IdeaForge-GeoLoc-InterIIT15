import math

import numpy as np
import torch

from coarse_search.correlation import correlate_everywhere
from header import settings
from header.settings import CELL, DEVICE, DISC_RADIUS, NUM_PEAKS, TILE, TILE_STEP
from image_processing.edge_features import query_edge_features
from image_processing.map_levels import level_gsd


def cut_map_features(features, top, left, rows, cols):
    piece = np.zeros((settings.N_BINS, rows, cols), dtype=np.float32)
    r0 = max(top, 0)
    r1 = min(top + rows, features.shape[1])
    c0 = max(left, 0)
    c1 = min(left + cols, features.shape[2])
    if r1 > r0 and c1 > c0:
        piece[:, r0 - top:r1 - top, c0 - left:c1 - left] = features[:, r0:r1, c0:c1]
    return piece


def correlation_maps(map_features, level, query_features, query_cells, prior_m):
    gsd = level_gsd(level)
    side = query_features.shape[1]
    reach = DISC_RADIUS / gsd + side * CELL / 2
    cells_wide = int(math.ceil(2 * reach / CELL))
    top = int(round((prior_m[1] / gsd - reach) / CELL))
    left = int(round((prior_m[0] / gsd - reach) / CELL))

    window = torch.from_numpy(cut_map_features(map_features[level], top, left, cells_wide, cells_wide)).to(DEVICE)
    kernel = torch.from_numpy(query_features).to(DEVICE)
    cells = torch.from_numpy(query_cells.astype(np.float32)).to(DEVICE)
    maps = correlate_everywhere(window, kernel, cells)

    shape = maps["cosine"].shape
    row_ids = torch.arange(shape[0], device=DEVICE)
    col_ids = torch.arange(shape[1], device=DEVICE)
    y_m = ((top + row_ids) * CELL + side * CELL / 2) * gsd
    x_m = ((left + col_ids) * CELL + side * CELL / 2) * gsd
    dx = x_m[None, :] - float(prior_m[0])
    dy = y_m[:, None] - float(prior_m[1])
    inside = dx ** 2 + dy ** 2 <= DISC_RADIUS ** 2
    geometry = dict(level=level, side=side, x_m=x_m, y_m=y_m, inside=inside)
    return maps, geometry


def pick_peaks(score_map, geometry, prior_map=None, beta=0.0):
    usable = (score_map > -1) & geometry["inside"]
    ranking = score_map.clone()
    if prior_map is not None and beta > 0:
        ranking = ranking + beta * prior_map
    ranking[~usable] = -1e9

    side = geometry["side"]
    gap = max(TILE // CELL, side // 2)
    last_row = score_map.shape[0] - 1
    last_col = score_map.shape[1] - 1
    peaks = []
    for _ in range(NUM_PEAKS):
        best_value, best_flat = ranking.reshape(-1).max(0)
        if float(best_value) < -1e8:
            break
        row = int(best_flat) // score_map.shape[1]
        col = int(best_flat) % score_map.shape[1]
        dense = float(score_map[row, col])
        around = (float(score_map[max(row - 1, 0), col]) + float(score_map[min(row + 1, last_row), col])
                  + float(score_map[row, max(col - 1, 0)]) + float(score_map[row, min(col + 1, last_col)])) / 4
        peaks.append((dense, float(geometry["x_m"][col]), float(geometry["y_m"][row]), dense - around, float(best_value)))
        ranking[max(row - gap, 0):row + gap + 1, max(col - gap, 0):col + gap + 1] = -1e9

    while len(peaks) < NUM_PEAKS:
        peaks.append((0.0, 1e9, 1e9, 0.0, -1.0))
    return np.array(peaks)


def search_frame(canvas, valid, level, prior_m, store, variants=(("cosine", 0.0), ("zncc", 0.0)), dino=None):
    query_features, query_cells = query_edge_features(canvas, valid)
    maps, geometry = correlation_maps(store["features"], level, query_features, query_cells, prior_m)
    raster, cutoff, prior_map = None, 0.0, None
    if dino is not None:
        raster, cutoff = dino.tile_score_raster(level, canvas, valid, prior_m)
        prior_map = dino.prior_on_grid(raster, geometry)
    found = {}
    for mode, beta in variants:
        peaks = pick_peaks(maps[mode], geometry, prior_map, beta)
        dino_column = np.full((NUM_PEAKS, 1), -1.0)
        if dino is not None:
            for k in range(NUM_PEAKS):
                if peaks[k, 1] < 1e8:
                    tile_row = int(round(peaks[k, 2] / level_gsd(level) / TILE_STEP - 1))
                    tile_col = int(round(peaks[k, 1] / level_gsd(level) / TILE_STEP - 1))
                    tile_row = min(max(tile_row, 0), raster.shape[0] - 1)
                    tile_col = min(max(tile_col, 0), raster.shape[1] - 1)
                    dino_column[k, 0] = raster[tile_row, tile_col]
        found[(mode, beta)] = np.concatenate([peaks, dino_column], axis=1)
    return found, cutoff
