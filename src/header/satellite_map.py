import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

from header import settings


class SatelliteMap:
    def __init__(self, path, bands=settings.MAP_BANDS):
        self.file = rasterio.open(path)
        self.bands = bands
        self.width = self.file.width
        self.height = self.file.height
        self.gsd = self.file.res[0]
        self.to_map_crs = Transformer.from_crs("EPSG:4326", self.file.crs, always_xy=True)
        self.pixel_from_coords = ~self.file.transform

        overview = self.read_grey(out_shape=(self.height // 16, self.width // 16))
        self.gray_max = float(np.percentile(overview[overview > 0], 99.5))

    def read_grey(self, window=None, out_shape=None, boundless=False):
        if out_shape is not None:
            bands = self.file.read(list(self.bands), out_shape=(len(self.bands),) + out_shape, resampling=Resampling.average)
        else:
            bands = self.file.read(list(self.bands), window=window, boundless=boundless, fill_value=0)
        return bands.astype(np.float32).mean(axis=0)

    def read_strip(self, top, rows):
        return self.read_grey(window=Window(0, top, self.width, rows))

    def grey_to_uint8(self, grey):
        result = np.clip(grey / self.gray_max, 0, 1) * 254 + 1
        result[grey <= 0] = 0
        return result.astype(np.uint8)

    def gps_to_map_pixel(self, lat, lon):
        easting, northing = self.to_map_crs.transform(lon, lat)
        col, row = self.pixel_from_coords * (easting, northing)
        return row, col
