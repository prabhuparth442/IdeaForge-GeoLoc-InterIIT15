"""Several GeoTIFF scenes read as one map on one pixel grid (used by preprocess only).

The grid is in the scenes' metric CRS with pixels of settings.MAP_GSD metres. A pixel that no scene covers is 0 ("no map").
Where scenes overlap the first scene listed wins. Nothing big is kept in memory: read_strip(top, rows) reads only those rows.
"""
import math

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

import settings


class MosaicMap:
    def __init__(self, paths, region_lonlat=None):
        """region_lonlat: (west, south, east, north) in degrees, or None for everything the scenes cover."""
        self.gsd = float(settings.MAP_GSD)
        self.files = [rasterio.open(path) for path in paths]
        self.crs = self.files[0].crs
        for scene in self.files:
            if scene.crs != self.crs:
                raise ValueError("all map scenes must share one CRS, got %s and %s" % (self.crs, scene.crs))

        if region_lonlat is None:
            west = min(scene.bounds.left for scene in self.files)
            east = max(scene.bounds.right for scene in self.files)
            south = min(scene.bounds.bottom for scene in self.files)
            north = max(scene.bounds.top for scene in self.files)
        else:
            to_map = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            corners = []
            for lon in (region_lonlat[0], region_lonlat[2]):
                for lat in (region_lonlat[1], region_lonlat[3]):
                    corners.append(to_map.transform(lon, lat))
            west = min(c[0] for c in corners)
            east = max(c[0] for c in corners)
            south = min(c[1] for c in corners)
            north = max(c[1] for c in corners)

        self.east0 = math.floor(west / self.gsd) * self.gsd          # left edge of column 0
        self.north0 = math.ceil(north / self.gsd) * self.gsd         # top edge of row 0
        self.width = int(math.ceil((east - self.east0) / self.gsd))
        self.height = int(math.ceil((self.north0 - south) / self.gsd))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("empty map region")

        overview = self._compose(0, self.height, 16)
        if not (overview > 0).any():
            raise ValueError("the map region has no data in any scene")
        self.gray_max = float(np.percentile(overview[overview > 0], 99.5))

    def _overlap(self, scene, top, rows, step):
        """Box (row0, row1, col0, col1) of the output strip that this scene covers, or None."""
        pixel = self.gsd * step
        b = scene.bounds
        col0 = max(int(math.ceil((b.left - self.east0) / pixel)), 0)
        col1 = min(int(math.floor((b.right - self.east0) / pixel)), self.width // step)
        row0 = max(int(math.ceil((self.north0 - top * self.gsd - b.top) / pixel)), 0)
        row1 = min(int(math.floor((self.north0 - top * self.gsd - b.bottom) / pixel)), rows // step)
        if col1 <= col0 or row1 <= row0:
            return None
        return row0, row1, col0, col1

    def _compose(self, top, rows, step):
        out = np.zeros((rows // step, self.width // step), dtype=np.float32)
        pixel = self.gsd * step
        for scene in self.files:
            box = self._overlap(scene, top, rows, step)
            if box is None:
                continue
            row0, row1, col0, col1 = box
            west = self.east0 + col0 * pixel
            upper = self.north0 - top * self.gsd - row0 * pixel
            window = Window((west - scene.bounds.left) / scene.res[0], (scene.bounds.top - upper) / scene.res[1],
                            (col1 - col0) * pixel / scene.res[0], (row1 - row0) * pixel / scene.res[1])
            bands = [b for b in settings_bands(scene)]
            part = scene.read(bands, window=window, out_shape=(len(bands), row1 - row0, col1 - col0),
                              resampling=Resampling.average, boundless=True, fill_value=0)
            grey = part.astype(np.float32).mean(axis=0)
            target = out[row0:row1, col0:col1]
            empty = target == 0
            target[empty] = grey[empty]
        return out

    def read_strip(self, top, rows):
        return self._compose(top, rows, 1)

    def grey_to_uint8(self, grey):
        result = np.clip(grey / self.gray_max, 0, 1) * 254 + 1
        result[grey <= 0] = 0
        return result.astype(np.uint8)


def settings_bands(scene):
    # the official map is one band; older maps had RGB-like bands that are averaged
    bands = [band for band in (1, 2, 3) if band <= scene.count]
    if len(bands) == 0:
        bands = [1]
    return bands
