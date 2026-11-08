"""Zoom levels: how big a map pixel is at each level, which level a photo needs, and how to cut a window out of a level picture."""
import math

import numpy as np

import settings


def level_gsd(level):
    return settings.MAP_GSD * 2 ** (level / 2)


def level_scale(level):
    return settings.MAP_GSD / level_gsd(level)


def level_for_height(height_m, camera):
    # level whose 224 px window is about as wide as what the camera sees from this height
    footprint = camera["width"] * height_m / camera["fx"]
    level = math.ceil(2 * math.log2(footprint / (settings.TILE * settings.MAP_GSD)))
    return int(np.clip(level, min(settings.LEVELS), max(settings.LEVELS)))


def map_piece(level_image, level, x_m, y_m, size=None):
    """Window of `size` pixels centred on (x_m, y_m) metres, grey values 0..1, zeros where there is no map."""
    if size is None:
        size = settings.TILE
    gsd = level_gsd(level)
    top = int(round(y_m / gsd - size / 2))
    left = int(round(x_m / gsd - size / 2))
    piece = np.zeros((size, size), dtype=np.float32)
    row0 = max(top, 0)
    row1 = min(top + size, level_image.shape[0])
    col0 = max(left, 0)
    col1 = min(left + size, level_image.shape[1])
    if row1 > row0 and col1 > col0:
        piece[row0 - top:row1 - top, col0 - left:col1 - left] = level_image[row0:row1, col0:col1] / 255
    return piece
