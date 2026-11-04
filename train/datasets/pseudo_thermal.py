"""Pseudo-thermal queries made from ordinary satellite pictures, so the fusion can learn from as many places as we like.

A real thermal photo shows the same edges as the satellite map, but the grey values mean something else (heat, not sunlight).
So we cut a crop out of the map, keep where its edges are, and change what the grey values mean:
  * every region (grey band x flat/textured) gets its own random "temperature"
  * part of the original brightness is kept, sometimes upside down
  * hot spots, slow gain drift, patches that changed (new roofs, cars), sensor noise
  * a small heading and scale error, and a rotated rectangle footprint like a real camera view
The result looks like a canvas from image_processing.geometry.render_query: 224 x 224, north-up, with a valid mask.

`strength` makes the fake harder (0 = close to the map, 2 = very different). A query made from the same map it is matched
against is always easier than a real photo, so launch_synthetic.py picks the strength by comparing with real photos.
"""
import cv2
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window

from header.settings import TILE
from image_processing.map_levels import level_gsd
from image_processing.thermal import normalise_frame

BIG = int(TILE * 1.4)                       # read more than the canvas so the small rotation and scale bring in no empty corners
ASPECT = 0.8                                # camera view is 5 : 4 (1280 x 1024 and 320 x 256)


def read_map_crop(satellite_map, x_m, y_m, level, size=BIG):
    """Grey crop in 0..1 around a point, read from the 0.3 m map and averaged down to the pixel size of the level."""
    side_in_map_pixels = int(round(size * level_gsd(level) / satellite_map.gsd))
    column = int(round(x_m / satellite_map.gsd - side_in_map_pixels / 2))
    row = int(round(y_m / satellite_map.gsd - side_in_map_pixels / 2))
    window = Window(column, row, side_in_map_pixels, side_in_map_pixels)
    bands = satellite_map.file.read(list(satellite_map.bands), window=window, out_shape=(len(satellite_map.bands), size, size),
                                    resampling=Resampling.average)
    grey = bands.astype(np.float32).mean(axis=0)
    return np.clip(grey / satellite_map.gray_max, 0, 1)


def blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def local_contrast(image, sigma):
    mean = blur(image, sigma)
    mean_of_squares = blur(image * image, sigma)
    return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0))


def region_temperatures(picture, rng):
    """One random temperature per region, where a region is a grey band that is either flat or textured."""
    base = blur(picture, 4.0)
    texture = blur(local_contrast(picture, 3.0), 5.0)
    band_count = int(rng.integers(3, 6))
    cut_levels = np.sort(rng.uniform(0.1, 0.9, band_count - 1))
    band = np.digitize(base, np.quantile(base, cut_levels))
    textured = texture > np.quantile(texture, rng.uniform(0.4, 0.7))
    region = band * 2 + textured.astype(int)
    temperature_of_region = rng.uniform(0, 1, 2 * band_count)
    temperature = temperature_of_region[region].astype(np.float32)
    return blur(temperature, 1.5), base


def add_hot_spots(thermal, rng, strength):
    spots = np.zeros_like(thermal)
    count = int(rng.integers(0, 3 + int(3 * strength)))
    for _ in range(count):
        x = int(rng.integers(0, thermal.shape[1]))
        y = int(rng.integers(0, thermal.shape[0]))
        cv2.circle(spots, (x, y), int(rng.integers(3, 12)), float(rng.uniform(-0.4, 0.4)), -1)
    return thermal + blur(spots, 3.0)


def add_changed_patches(thermal, rng, strength):
    count = int(rng.integers(0, 2 + int(2 * strength)))
    for _ in range(count):
        patch_height = int(rng.integers(6, 40))
        patch_width = int(rng.integers(6, 40))
        top = int(rng.integers(0, thermal.shape[0] - patch_height))
        left = int(rng.integers(0, thermal.shape[1] - patch_width))
        thermal[top:top + patch_height, left:left + patch_width] = rng.uniform(0, 1)
    return thermal


def make_pseudo_thermal(crop, rng, strength=1.0):
    """crop: grey satellite picture in 0..1. Returns a float picture that behaves like a thermal view of the same ground."""
    picture = blur(crop, rng.uniform(1.0, 2.2))
    temperature, base = region_temperatures(picture, rng)
    detail = picture - blur(picture, 2.5)

    cluster_share = float(np.clip(0.3 + 0.25 * strength + rng.uniform(-0.1, 0.1), 0.2, 0.9))
    if rng.random() < 0.5:
        sunlit = base
    else:
        sunlit = 1.0 - base
    detail_gain = rng.uniform(0.8, 2.0) / (1.0 + 0.4 * strength)
    polarity = rng.choice([-1.0, 1.0])
    thermal = cluster_share * temperature + (1 - cluster_share) * sunlit + polarity * detail_gain * detail

    thermal = add_hot_spots(thermal, rng, strength)
    thermal = add_changed_patches(thermal, rng, strength)

    drift = rng.uniform(0.8, 1.2, (4, 4)).astype(np.float32)
    drift = cv2.resize(drift, (thermal.shape[1], thermal.shape[0]), interpolation=cv2.INTER_CUBIC)
    thermal = thermal * drift + rng.uniform(-0.1, 0.1)

    noise_level = rng.uniform(0.003, 0.012) * (1 + strength)
    thermal = thermal + rng.normal(0, noise_level, thermal.shape)
    stripes = rng.normal(0, 0.004 * strength, thermal.shape[1])
    return (thermal + stripes[None, :]).astype(np.float32)


def footprint_mask(rng, width_fraction=None):
    """Rotated 5 : 4 rectangle that fills 71 to 100 percent of the canvas width, like the camera view of a real photo."""
    if width_fraction is None:
        width_fraction = rng.uniform(0.71, 1.0)
    width = TILE * width_fraction
    middle = (TILE - 1) / 2
    corners = cv2.boxPoints(((middle, middle), (width, ASPECT * width), rng.uniform(0, 360)))
    mask = np.zeros((TILE, TILE), np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 1)
    return mask.astype(bool), width


def make_synthetic_query(satellite_map, x_m, y_m, level, rng, strength=1.0):
    """Returns (canvas, valid, info). The truth is the canvas middle, which is the point (x_m, y_m) of the map."""
    crop = read_map_crop(satellite_map, x_m, y_m, level)
    thermal = make_pseudo_thermal(crop, rng, strength)

    middle = (BIG - 1) / 2
    angle = rng.normal(0, 1.5 * strength)
    scale = 1 + rng.normal(0, 0.02 * strength)
    matrix = cv2.getRotationMatrix2D((middle, middle), angle, scale)
    matrix[:, 2] += (TILE - BIG) / 2
    warped = cv2.warpAffine(thermal, matrix, (TILE, TILE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    valid, width = footprint_mask(rng)
    rows = np.where(valid.any(axis=1))[0]
    columns = np.where(valid.any(axis=0))[0]
    top, bottom, left, right = rows[0], rows[-1] + 1, columns[0], columns[-1] + 1
    cleaned = np.zeros_like(warped)
    cleaned[top:bottom, left:right] = normalise_frame(warped[top:bottom, left:right])
    canvas = np.where(valid, cleaned, cleaned[valid].mean()).astype(np.float32)

    start = (BIG - TILE) // 2
    info = dict(footprint_m=width * level_gsd(level), crop=crop[start:start + TILE, start:start + TILE],
                map_fraction=float((crop > 0).mean()))
    return canvas, valid, info
