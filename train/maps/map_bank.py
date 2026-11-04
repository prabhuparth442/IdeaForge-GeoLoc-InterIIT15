"""Many score maps from one photo.

A map is named  <family>_<kind>[_w]  :  edge_zncc, edge_cosine, edge_mag, edge_zncc_w (whitened), mind_zncc, learned_zncc ...
  kind zncc    the existing masked ZNCC over the 8 cell channels
  kind cosine  the existing masked cosine
  kind mag     ZNCC of the single channel 'total edge strength' (sum of the 8 channels): ignores direction, keeps structure
  suffix _w    the same, after pre-whitening both sides (C1), strength config.WHITEN_ALPHA
Everything shares one window and one geometry, so maps can be added or compared cell by cell.
"""
import math

import numpy as np
import torch

from coarse_search.correlation import correlate_everywhere
from coarse_search.fusion import Z_FLOOR, robust_z
from coarse_search.search import cut_map_features
from header import settings
from header.settings import CELL, DEVICE, DISC_RADIUS
from image_processing.map_levels import level_gsd
from train import config
from train.features.whitening import whiten


def parse_name(name):
    parts = name.split("_")
    assert len(parts) in (2, 3) and parts[1] in ("zncc", "cosine", "mag"), "bad map name %r (use family_kind or family_kind_w)" % name
    return parts[0], parts[1], (len(parts) == 3 and parts[2] == "w")


def compute_maps(canvas, valid, level, prior_m, families, names, alpha=None):
    """Returns (maps dict name -> (R, C) tensor, geometry). families: dict name -> Family."""
    assert len(names) > 0, "need at least one map name"
    alpha = config.WHITEN_ALPHA if alpha is None else alpha
    gsd = level_gsd(level)
    side = canvas.shape[0] // CELL
    reach = DISC_RADIUS / gsd + side * CELL / 2
    cells_wide = int(math.ceil(2 * reach / CELL))
    top = int(round((prior_m[1] / gsd - reach) / CELL))
    left = int(round((prior_m[0] / gsd - reach) / CELL))

    needed = {}
    for name in names:
        family, kind, whitened = parse_name(name)
        assert family in families, "map %s needs the family %s, which is not loaded" % (name, family)
        needed.setdefault(family, []).append((name, kind, whitened))

    maps = {}
    for family_name, entries in needed.items():
        family = families[family_name]
        query_features, query_cells = family.query(canvas, valid)
        window = torch.from_numpy(cut_map_features(family.map_features(level), top, left, cells_wide, cells_wide)).to(DEVICE)
        kernel = torch.from_numpy(query_features).to(DEVICE)
        cells = torch.from_numpy(query_cells.astype(np.float32)).to(DEVICE)
        window_w = kernel_w = None
        for name, kind, whitened in entries:
            if kind == "mag":
                out = correlate_everywhere(window.sum(0, keepdim=True), kernel.sum(0, keepdim=True), cells)["zncc"]
            elif whitened:
                if window_w is None:
                    profile = family.profile(level)
                    window_w = whiten(window, profile, alpha, window.abs().sum(0) > 0)
                    kernel_w = whiten(kernel, profile, alpha, cells > 0)
                out = correlate_everywhere(window_w, kernel_w, cells)[kind]
            else:
                out = correlate_everywhere(window, kernel, cells)[kind]
            maps[name] = out
        del window, window_w, kernel_w

    shape = next(iter(maps.values())).shape
    assert all(m.shape == shape for m in maps.values()), "maps of different families do not share a grid"
    row_ids = torch.arange(shape[0], device=DEVICE)
    col_ids = torch.arange(shape[1], device=DEVICE)
    y_m = ((top + row_ids) * CELL + side * CELL / 2) * gsd
    x_m = ((left + col_ids) * CELL + side * CELL / 2) * gsd
    inside = ((x_m[None, :] - float(prior_m[0])) ** 2 + (y_m[:, None] - float(prior_m[1])) ** 2) <= DISC_RADIUS ** 2
    return maps, dict(level=level, side=side, x_m=x_m, y_m=y_m, inside=inside)


def nms_peaks(score, usable, gap_cells, n):
    """Greedy peaks with a blanked square around each pick: [(row, col, value)]."""
    work = torch.where(usable, score, torch.full_like(score, -1e9))
    peaks, width = [], score.shape[1]
    for _ in range(n):
        value, flat = work.reshape(-1).max(0)
        if float(value) < -1e8:
            break
        row, col = int(flat) // width, int(flat) % width
        peaks.append((row, col, float(value)))
        work[max(row - gap_cells, 0):row + gap_cells + 1, max(col - gap_cells, 0):col + gap_cells + 1] = -1e9
    return peaks


def z_stack(maps, geometry, names):
    """z-scores of all maps, (R, C, K); the usable mask; the context vector (2K + 1,)."""
    usable = (maps[names[0]] > -1) & geometry["inside"]
    assert usable.any(), "no usable position (photo outside the map?)"
    gap = max(settings.TILE // CELL, geometry["side"] // 2)
    zs, context = [], []
    for name in names:
        z = robust_z(maps[name], usable & (maps[name] > -1)) if bool((usable & (maps[name] > -1)).any()) else torch.full_like(maps[name], Z_FLOOR)
        z = torch.where(usable, z, torch.full_like(z, Z_FLOOR))
        zs.append(z)
        peaks = nms_peaks(z, usable, gap, 2)
        first = peaks[0][2] if peaks else Z_FLOOR
        second = peaks[1][2] if len(peaks) > 1 else Z_FLOOR
        context += [first, first - second]
    context.append(math.log(level_gsd(geometry["level"])))
    return torch.stack(zs, dim=-1), usable, np.array(context, dtype=np.float32)


def cell_of(geometry, x_m, y_m):
    """(row, col) of the grid cell nearest to a point in metres, or None if the point is outside the window."""
    cols = geometry["x_m"]
    rows = geometry["y_m"]
    col = int(torch.argmin((cols - x_m).abs()))
    row = int(torch.argmin((rows - y_m).abs()))
    cell_m = CELL * level_gsd(geometry["level"])
    if abs(float(cols[col]) - x_m) > cell_m or abs(float(rows[row]) - y_m) > cell_m:
        return None
    return row, col
