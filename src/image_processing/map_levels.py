import math

import cv2
import numpy as np

from header import settings


def level_gsd(level):
    assert level in settings.LEVELS, "level %s is not one of %s" % (level, settings.LEVELS)
    return settings.MAP_GSD * 2 ** (level / 2)


def level_scale(level):
    assert level in settings.LEVELS, "level %s is not one of %s" % (level, settings.LEVELS)
    return settings.MAP_GSD / level_gsd(level)


def level_for_height(height, camera):
    assert height > 0, "height must be above 0 m, got %s" % height
    assert camera.fx > 0 and camera.width > 0, "camera needs a positive focal length and width"
    footprint = camera.width * height / camera.fx
    level = math.ceil(2 * math.log2(footprint / (settings.TILE * settings.MAP_GSD)))
    return int(np.clip(level, min(settings.LEVELS), max(settings.LEVELS)))


def build_level_images(satellite_map, levels, strip_rows=settings.STRIP_ROWS):
    assert len(levels) > 0 and strip_rows > 0, "need at least one level and a positive strip height"
    assert all(level in settings.LEVELS for level in levels), "levels must come from settings.LEVELS"
    images = {}
    for level in levels:
        out_h = int(satellite_map.height * level_scale(level))
        out_w = int(satellite_map.width * level_scale(level))
        images[level] = np.zeros((out_h, out_w), dtype=np.uint8)
    for top in range(0, satellite_map.height, strip_rows):
        rows = min(strip_rows, satellite_map.height - top)
        grey = satellite_map.read_strip(top, rows)
        for level in levels:
            scale = level_scale(level)
            out_h, out_w = images[level].shape
            y0 = int(round(top * scale))
            y1 = min(int(round((top + rows) * scale)), out_h)
            if y1 > y0:
                small = cv2.resize(grey, (out_w, y1 - y0), interpolation=cv2.INTER_AREA)
                images[level][y0:y1] = satellite_map.grey_to_uint8(small)
    return images


def normalise_reference(image):
    assert image.ndim == 2 and image.dtype == np.uint8, "level image must be 2-D uint8 (0 = no map), got %s %s" % (image.shape, image.dtype)
    has_map = image > 0
    grid = (max(image.shape[1] // 28, 1), max(image.shape[0] // 28, 1))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid)
    result = np.maximum(clahe.apply(image), 1)
    result[~has_map] = 0
    return result


def build_level_pictures(satellite_map):
    raw_levels = build_level_images(satellite_map, settings.LEVELS)
    level_images = {}
    for level in settings.LEVELS:
        level_images[level] = normalise_reference(raw_levels[level])
    return level_images


def map_piece(level_images, level, x_m, y_m, size=settings.TILE):
    assert level in level_images, "level %s was not built" % level
    assert size > 0, "size must be positive pixels"
    assert math.isfinite(x_m) and math.isfinite(y_m), "x_m and y_m must be finite metres"
    image = level_images[level]
    gsd = level_gsd(level)
    top = int(round(y_m / gsd - size / 2))
    left = int(round(x_m / gsd - size / 2))
    piece = np.zeros((size, size), dtype=np.float32)
    r0, r1 = max(top, 0), min(top + size, image.shape[0])
    c0, c1 = max(left, 0), min(left + size, image.shape[1])
    if r1 > r0 and c1 > c0:
        piece[r0 - top:r1 - top, c0 - left:c1 - left] = image[r0:r1, c0:c1] / 255
    return piece
