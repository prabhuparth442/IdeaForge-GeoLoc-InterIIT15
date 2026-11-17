"""Block: coarse score maps of the whole photo and of parts of it (ideas planner, second list: ideas 3, 4, 7, 6).

No plotting, no learning. The coarse stage correlates the 224 px edge-direction canvas over the whole search disc. Here the same correlation is run
with a mask on the canvas cells, so a part of the photo (a quadrant, the centre) is correlated on its own; all maps share one grid, so they can be
compared cell by cell.

zncc_map           score map (rows, cols) of the canvas, or of the cells in `mask`, over the disc; None when the mask keeps too few cells
part_masks         the four quadrants and a centre disc, as masks over the canvas cells
peaks_of           the best `count` non-overlapping peaks of a map (the pipeline keeps 25; here more, to find the rank of the true place)
truth_rank         1-based rank of the first peak within `within` sigma of the truth, 0 = not among the peaks
pool_slack         max over a small window, so a part that sits a few cells off still counts
combine            mean or minimum of several maps where all are usable
local_peak         best zncc of the canvas (or a part) in a small window around one place: value and the offset of the peak in cells
"""
import numpy as np
import torch
import torch.nn.functional as F

from coarse_search import search
from coarse_search.correlation import correlate_everywhere
from header.settings import CELL, DEVICE
from image_processing.edge_features import query_edge_features
from image_processing.map_levels import level_gsd

MIN_PART_CELLS = 40


def part_masks(side):
    """Four quadrants and a centre disc (60 % of the canvas width), as bool masks over the (side, side) cells."""
    rows, cols = np.mgrid[0:side, 0:side]
    half = side // 2
    masks = dict(top_left=(rows < half) & (cols < half), top_right=(rows < half) & (cols >= half),
                 bottom_left=(rows >= half) & (cols < half), bottom_right=(rows >= half) & (cols >= half))
    middle = (side - 1) / 2
    centre = (rows - middle) ** 2 + (cols - middle) ** 2 <= (0.3 * side) ** 2
    return masks, centre


def zncc_map(store, canvas, valid, level, prior_m, mask=None):
    """Returns (score_map, geometry) like search.correlation_maps()['zncc'], or (None, None) when the mask keeps fewer than MIN_PART_CELLS valid cells."""
    features, cells = query_edge_features(canvas, valid)
    if mask is not None:
        cells = cells & mask
        if cells.sum() < MIN_PART_CELLS:
            return None, None
        features = features * cells
    maps, geometry = search.correlation_maps(store["features"], level, features, cells, prior_m)
    return maps["zncc"], geometry


def peaks_of(score_map, geometry, count=200):
    keep = search.NUM_PEAKS
    search.NUM_PEAKS = count
    try:
        return search.pick_peaks(score_map, geometry)
    finally:
        search.NUM_PEAKS = keep


def truth_rank(peaks, truth, sigma, within=4.0):
    distance = np.hypot(peaks[:, 1] - truth[0], peaks[:, 2] - truth[1]) / sigma
    hit = np.where(distance <= within)[0]
    return int(hit[0]) + 1 if len(hit) else 0


def pool_slack(score_map, cells=5):
    pooled = F.max_pool2d(score_map[None, None], cells, stride=1, padding=cells // 2)[0, 0]
    pooled[score_map <= -1] = -1.0
    return pooled


def combine(maps, how="mean"):
    stack = torch.stack(maps)
    usable = (stack > -1).all(dim=0)
    out = stack.mean(dim=0) if how == "mean" else stack.min(dim=0).values
    out[~usable] = -1.0
    return out


def local_peak(store, canvas, valid, level, x_m, y_m, radius_cells=3, mask=None):
    """Best zncc of the canvas (or the part in `mask`) when it is slid over +-radius_cells around the place (x_m, y_m).
    Returns (value, row_offset, col_offset) in cells; (-1, 0, 0) when nothing usable."""
    features, cells = query_edge_features(canvas, valid)
    if mask is not None:
        cells = cells & mask
        if cells.sum() < MIN_PART_CELLS:
            return -1.0, 0, 0
        features = features * cells
    side = features.shape[1]
    gsd = level_gsd(level)
    top = int(round(y_m / gsd / CELL - side / 2)) - radius_cells
    left = int(round(x_m / gsd / CELL - side / 2)) - radius_cells
    window = torch.from_numpy(search.cut_map_features(store["features"][level], top, left, side + 2 * radius_cells, side + 2 * radius_cells)).to(DEVICE)
    kernel = torch.from_numpy(features).to(DEVICE)
    cells_t = torch.from_numpy(cells.astype(np.float32)).to(DEVICE)
    zncc = correlate_everywhere(window, kernel, cells_t)["zncc"]
    value, flat = zncc.reshape(-1).max(0)
    row, col = divmod(int(flat), zncc.shape[1])
    middle = (zncc.shape[0] - 1) / 2
    return float(value), row - middle, col - middle
