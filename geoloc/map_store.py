"""The map store: the satellite map at every zoom level, plus its edge-direction cells.

build_map_store()  reads the GeoTIFFs and writes the store to a folder (this is what ./preprocess does; it takes minutes to hours).
MapStore(folder)   opens that folder. The big arrays are memory-mapped, so opening is fast and only the windows that are used get read.

Files in the folder:  store.json,  level_<L>.npy (uint8 picture),  features_<L>.npy (float32, 8 x rows x cols)
Positions are metres on the store grid: x to the east from the left edge, y to the south from the top edge.
"""
import json
import os
import time

import cv2
import numpy as np
from pyproj import Transformer

import settings
from geoloc.edge_features import map_edge_features
from geoloc.levels import level_scale
from geoloc.map_reader import MosaicMap


def build_level_images(mosaic):
    images = {}
    for level in settings.LEVELS:
        height = int(mosaic.height * level_scale(level))
        width = int(mosaic.width * level_scale(level))
        images[level] = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, mosaic.height, settings.STRIP_ROWS):
        rows = min(settings.STRIP_ROWS, mosaic.height - top)
        grey = mosaic.read_strip(top, rows)
        for level in settings.LEVELS:
            scale = level_scale(level)
            height, width = images[level].shape
            row0 = int(round(top * scale))
            row1 = min(int(round((top + rows) * scale)), height)
            if row1 > row0:
                small = cv2.resize(grey, (width, row1 - row0), interpolation=cv2.INTER_AREA)
                images[level][row0:row1] = mosaic.grey_to_uint8(small)
    return images


def equalise_level(image):
    # local contrast stretch, so thermal edges and map edges look alike
    has_map = image > 0
    grid = (max(image.shape[1] // 28, 1), max(image.shape[0] // 28, 1))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid)
    result = np.maximum(clahe.apply(image), 1)
    result[~has_map] = 0
    return result


def build_map_store(map_files, out_dir, region_lonlat=None, log=print):
    start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    mosaic = MosaicMap(map_files, region_lonlat)
    log("map grid: %d x %d pixels of %.2f m" % (mosaic.width, mosaic.height, mosaic.gsd))
    images = build_level_images(mosaic)
    for level in settings.LEVELS:
        picture = equalise_level(images[level])
        np.save(os.path.join(out_dir, "level_%d.npy" % level), picture)
        np.save(os.path.join(out_dir, "features_%d.npy" % level), map_edge_features(picture))
        log("level %d done, picture %d x %d, %.0f s so far" % (level, picture.shape[1], picture.shape[0], time.time() - start))
        images[level] = None
    info = dict(east0=mosaic.east0, north0=mosaic.north0, gsd=mosaic.gsd, crs=mosaic.crs.to_string(),
                levels=list(settings.LEVELS), width=mosaic.width, height=mosaic.height)
    with open(os.path.join(out_dir, "store.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    log("map store written to %s in %.0f s" % (out_dir, time.time() - start))


class MapStore:
    def __init__(self, folder):
        with open(os.path.join(folder, "store.json")) as handle:
            self.info = json.load(handle)
        if abs(self.info["gsd"] - settings.MAP_GSD) > 1e-6:
            raise ValueError("the store was built with MAP_GSD %s but settings.MAP_GSD is %s, run preprocess again" % (self.info["gsd"], settings.MAP_GSD))
        self.levels = {}
        self.features = {}
        for level in self.info["levels"]:
            self.levels[level] = np.load(os.path.join(folder, "level_%d.npy" % level), mmap_mode="r")
            self.features[level] = np.load(os.path.join(folder, "features_%d.npy" % level), mmap_mode="r")
        missing = [level for level in settings.LEVELS if level not in self.levels]
        if missing:
            raise ValueError("the store has no level %s that settings.LEVELS asks for, run preprocess again" % missing)
        self.to_map = Transformer.from_crs("EPSG:4326", self.info["crs"], always_xy=True)
        self.to_gps = Transformer.from_crs(self.info["crs"], "EPSG:4326", always_xy=True)

    def gps_to_metres(self, lat, lon):
        easting, northing = self.to_map.transform(lon, lat)
        return easting - self.info["east0"], self.info["north0"] - northing

    def metres_to_gps(self, x_m, y_m):
        lon, lat = self.to_gps.transform(self.info["east0"] + x_m, self.info["north0"] - y_m)
        return lat, lon
