"""Read the needed piece of the satellite map and sample it into the camera image.

Two steps:
  read_ground_crop   one north-up grey piece of the map, averaged down to the pixel size the camera needs
  sample_into_view   for every camera pixel look up the crop at its ground point; a small stack of blurred copies
                     (mip levels) makes pixels that cover a lot of ground use averaged values instead of single map pixels
"""
import math

import cv2
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window

MAX_SOURCE_PIXELS = 60_000_000      # most map pixels one crop may be averaged from (keeps a single call cheap)
MAX_LEVELS = 6


def footprint_metres(east, north, valid):
    """Ground size of each camera pixel (metres): the larger of its east-west and north-south steps (larger = safer for aliasing)."""
    d_east_row, d_east_col = np.gradient(east)
    d_north_row, d_north_col = np.gradient(north)
    along_col = np.hypot(d_east_col, d_north_col)
    along_row = np.hypot(d_east_row, d_north_row)
    size = np.maximum(along_col, along_row)
    return np.where(valid, size, np.nan)


def count_pixels(wanted):
    """Round up, but a value that is a whole number up to float noise (224.0000001) stays that whole number."""
    nearest = round(wanted)
    if abs(wanted - nearest) < 1e-3:
        return int(nearest)
    return int(math.ceil(wanted))


def read_ground_crop(satellite_map, east_min, east_max, north_min, north_max, gsd):
    """Grey crop in 0..1 covering the box, averaged down to about `gsd` metres per pixel (never finer than the map).

    Returns dict(image, east0, north0, gsd): east0 / north0 are the coordinates of the centre of pixel (0, 0); north-up.
    """
    transform = satellite_map.file.transform
    assert transform.b == 0 and transform.d == 0, "map must be north-up (no rotation)"
    map_gsd = satellite_map.gsd
    gsd = max(gsd, map_gsd)

    col_left = (east_min - transform.c) / transform.a
    row_top = (north_max - transform.f) / transform.e
    width_map_pixels = (east_max - east_min) / map_gsd
    height_map_pixels = (north_max - north_min) / map_gsd
    assert width_map_pixels * height_map_pixels <= MAX_SOURCE_PIXELS, (
        "crop needs %.0f million map pixels, over the limit of %.0f million: use a coarser map level or a smaller view" % (
            width_map_pixels * height_map_pixels / 1e6, MAX_SOURCE_PIXELS / 1e6))

    out_width = max(2, count_pixels(width_map_pixels * map_gsd / gsd))
    out_height = max(2, count_pixels(height_map_pixels * map_gsd / gsd))
    window = Window(col_left, row_top, width_map_pixels, height_map_pixels)
    bands = satellite_map.file.read(list(satellite_map.bands), window=window, out_shape=(len(satellite_map.bands), out_height, out_width),
                                    resampling=Resampling.average, boundless=True, fill_value=0)
    grey = bands.astype(np.float32).mean(axis=0)
    grey = np.clip(grey / satellite_map.gray_max, 0, 1)

    pixel_east = (east_max - east_min) / out_width
    pixel_north = (north_max - north_min) / out_height
    return dict(image=grey, east0=east_min + pixel_east / 2, north0=north_max - pixel_north / 2, gsd_east=pixel_east, gsd_north=pixel_north)


def make_levels(image, count):
    levels = [image]
    for _ in range(count - 1):
        if min(levels[-1].shape) < 8:
            break
        levels.append(cv2.pyrDown(levels[-1]))
    return levels


def sample_into_view(crop, east, north, valid, footprint, values=None):
    """Look the crop up at every ground point. Returns (view, known): the sampled picture and where the map had data (0 = empty).

    values: a picture on the same grid as crop["image"] to sample instead of the grey map (the thermal look); known always comes from the map.
    """
    if values is None:
        values = crop["image"]
    base_gsd = 0.5 * (crop["gsd_east"] + crop["gsd_north"])
    col = (east - crop["east0"]) / crop["gsd_east"]
    row = (crop["north0"] - north) / crop["gsd_north"]

    lod = np.log2(np.maximum(np.nan_to_num(footprint, nan=base_gsd), base_gsd) / base_gsd)
    count = int(min(MAX_LEVELS, math.floor(float(np.max(lod))) + 2))
    levels = make_levels(values, count)
    known_levels = make_levels((crop["image"] > 0).astype(np.float32), count)
    lod = np.clip(lod, 0, len(levels) - 1)

    view = np.zeros(east.shape, np.float32)
    known = np.zeros(east.shape, np.float32)
    weight_total = np.zeros(east.shape, np.float32)
    for level_index in range(len(levels)):
        weight = np.clip(1.0 - np.abs(lod - level_index), 0, 1).astype(np.float32)
        if not weight.any():
            continue
        step = 2.0 ** level_index
        level_col = ((col + 0.5) / step - 0.5).astype(np.float32)
        level_row = ((row + 0.5) / step - 0.5).astype(np.float32)
        sampled = cv2.remap(levels[level_index], level_col, level_row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sampled_known = cv2.remap(known_levels[level_index], level_col, level_row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        view += weight * sampled
        known += weight * sampled_known
        weight_total += weight
    weight_total = np.maximum(weight_total, 1e-6)
    return view / weight_total, (known / weight_total) * valid
