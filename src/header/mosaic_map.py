"""Block: several GeoTIFF scenes read as ONE satellite map, cut to a region of interest.

The new d1 base map comes as two 11.7 GB scenes (0.28 m pixels, one band, overlapping). The pipeline wants one map in one pixel grid, and
it only needs the area around the photos (the 5.5 km search disc around the hint), so MosaicMap builds its own grid on the region of interest (ROI):

  * the grid is in the scenes' metric CRS (all scenes must share it), pixel size = settings.MAP_GSD (the scenes are resampled to it while reading)
  * a pixel with no data in any scene is 0 (the pipeline treats 0 as "no map"); where scenes overlap, the first scene listed wins
  * nothing big is held in memory: read_strip(top, rows) reads only those rows from the scenes

It has the same members as header.satellite_map.SatelliteMap (width, height, gsd, gray_max, read_strip, grey_to_uint8, gps_to_map_pixel),
plus metres_to_gps for turning a pipeline answer (x, y in metres on this grid) back into latitude and longitude.
"""
import math

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

from header import settings


class MosaicMap:
    def __init__(self, paths, roi_lonlat, gsd=None, bands=settings.MAP_BANDS):
        """roi_lonlat = (west, south, east, north) in degrees."""
        assert len(paths) > 0, "need at least one map file"
        west, south, east, north = roi_lonlat
        assert -180 <= west < east <= 180 and -90 <= south < north <= 90, "roi_lonlat must be (west, south, east, north) in degrees, got %s" % (roi_lonlat,)
        self.gsd = float(settings.MAP_GSD if gsd is None else gsd)
        self.files = [rasterio.open(path) for path in paths]
        crs = self.files[0].crs
        assert all(f.crs == crs for f in self.files), "all scenes must share one CRS, got %s" % [str(f.crs) for f in self.files]
        self.bands = bands
        self.to_map_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.to_gps = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        corners = [self.to_map_crs.transform(lon, lat) for lon in (west, east) for lat in (south, north)]
        eastings, northings = [c[0] for c in corners], [c[1] for c in corners]
        self.east0 = math.floor(min(eastings) / self.gsd) * self.gsd           # left edge of column 0
        self.north0 = math.ceil(max(northings) / self.gsd) * self.gsd          # top edge of row 0
        self.width = int(math.ceil((max(eastings) - self.east0) / self.gsd))
        self.height = int(math.ceil((self.north0 - min(northings)) / self.gsd))
        assert self.width > 0 and self.height > 0, "empty region of interest"
        assert any(self._overlap(f, 0, self.height, 1) is not None for f in self.files), "the region of interest is outside every scene"

        overview = self._compose(0, self.height, 16)
        assert (overview > 0).any(), "the region of interest has no map data"
        self.gray_max = float(np.percentile(overview[overview > 0], 99.5))

    def _overlap(self, scene, top, rows, step):
        """Output pixel box (r0, r1, c0, c1) of the strip that this scene covers (None if it does not), strip pixels are step * gsd wide."""
        pixel = self.gsd * step
        b = scene.bounds
        c0 = max(int(math.ceil((b.left - self.east0) / pixel)), 0)
        c1 = min(int(math.floor((b.right - self.east0) / pixel)), self.width // step)
        r0 = max(int(math.ceil((self.north0 - top * self.gsd - b.top) / pixel)), 0)
        r1 = min(int(math.floor((self.north0 - top * self.gsd - b.bottom) / pixel)), rows // step)
        if c1 <= c0 or r1 <= r0:
            return None
        return r0, r1, c0, c1

    def _compose(self, top, rows, step):
        """The grey picture of strip rows top..top+rows, shrunk by the integer step (1 = full resolution)."""
        out = np.zeros((rows // step, self.width // step), dtype=np.float32)
        pixel = self.gsd * step
        for scene in self.files:
            box = self._overlap(scene, top, rows, step)
            if box is None:
                continue
            r0, r1, c0, c1 = box
            west = self.east0 + c0 * pixel
            upper = self.north0 - top * self.gsd - r0 * pixel
            window = Window((west - scene.bounds.left) / scene.res[0], (scene.bounds.top - upper) / scene.res[1], (c1 - c0) * pixel / scene.res[0], (r1 - r0) * pixel / scene.res[1])
            used = [b for b in self.bands if b <= scene.count] or [1]
            part = scene.read(used, window=window, out_shape=(len(used), r1 - r0, c1 - c0), resampling=Resampling.average, boundless=True, fill_value=0)
            grey = part.astype(np.float32).mean(axis=0)
            target = out[r0:r1, c0:c1]
            target[target == 0] = grey[target == 0]                             # the first scene listed wins where scenes overlap
        return out

    def read_strip(self, top, rows):
        assert rows > 0 and 0 <= top and top + rows <= self.height, "strip rows %d..%d outside the map (height %d)" % (top, top + rows, self.height)
        return self._compose(top, rows, 1)

    def grey_to_uint8(self, grey):
        assert grey.ndim == 2, "grey must be a 2-D array, got shape %s" % (grey.shape,)
        result = np.clip(grey / self.gray_max, 0, 1) * 254 + 1
        result[grey <= 0] = 0
        return result.astype(np.uint8)

    def gps_to_map_pixel(self, lat, lon):
        assert -90 <= lat <= 90, "latitude must be in degrees -90..90 (lat first, lon second), got %s" % lat
        assert -180 <= lon <= 180, "longitude must be in degrees -180..180, got %s" % lon
        easting, northing = self.to_map_crs.transform(lon, lat)
        return (self.north0 - northing) / self.gsd, (easting - self.east0) / self.gsd           # row, column

    def metres_to_gps(self, x_m, y_m):
        """A position on this map in metres (x to the right, y down, from the top-left corner) -> (lat, lon)."""
        assert math.isfinite(x_m) and math.isfinite(y_m), "x_m and y_m must be finite metres"
        lon, lat = self.to_gps.transform(self.east0 + x_m, self.north0 - y_m)
        return lat, lon
