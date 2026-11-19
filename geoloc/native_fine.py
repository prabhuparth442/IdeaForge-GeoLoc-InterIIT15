"""Native-resolution fine matching: for high photos LoFTR gets a canvas drawn at the photo's own resolution instead of the coarse level's.

Why (iteration 34, 2026-11-18): above about 1000 m the coarse level's pixel (9.6 or 13.6 m) is 1.4 to 1.9 times bigger than the photo's pixel, and the photo is blurred to match,
so clumps of one to three photo pixels are gone before LoFTR sees them. Drawing the canvas two levels finer (448 x 448 pixels, same footprint) with the map tile at the same
finer level keeps them. The photo is cleaned (vignette removal and local normalisation) and so is the map tile (local normalisation), because treating only one side was worse.
The coarse search is not touched: only the LoFTR stage of the queries of the photo itself (not of altered copies) uses this.

fine_level   the finer level to use for this photo, or None when the switch is off, the photo is lower than the gate, or the store has no such level
tile_sigma   local normalisation size for the map tile, in tile pixels, equal to NATIVE_FINE_LOCAL_NORM_PX photo pixels
clean_tile   the map tile as LoFTR should get it: local normalisation, then the same stretch + CLAHE as the photo
"""
import numpy as np

import settings
from geoloc import flat_field, photo


def fine_level(height_m, level, store_levels):
    gate = settings.NATIVE_FINE_FROM_HEIGHT_M
    if gate is None or height_m < gate:
        return None
    finer = level - settings.NATIVE_FINE_LEVELS
    if finer < min(settings.LEVELS) or finer not in store_levels:
        return None
    return finer


def canvas_size(levels_finer):
    # every second level halves the pixel, so the canvas that covers the same footprint has 2 ** (levels_finer / 2) times as many pixels per side
    return int(round(settings.TILE * 2 ** (levels_finer / 2)))


def tile_sigma(photo_gsd, tile_gsd):
    return max(3.0, settings.NATIVE_FINE_LOCAL_NORM_PX * photo_gsd / tile_gsd)


def clean_tile(tile, sigma):
    has_map = tile > 0
    grey = float(tile[has_map].mean()) if has_map.any() else 0.5
    filled = np.where(has_map, tile, grey).astype(np.float32)
    normalised = photo.normalise_frame(flat_field.local_normalise(filled, sigma))
    background = float(normalised[has_map].mean()) if has_map.any() else 0.5
    return np.where(has_map, normalised, background).astype(np.float32)
