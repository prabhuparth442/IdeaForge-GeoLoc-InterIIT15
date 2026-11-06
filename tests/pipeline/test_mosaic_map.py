"""MosaicMap: two overlapping scenes read as one map, and positions turned back into latitude / longitude (synthetic GeoTIFFs, no dataset needed)."""
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from common import run_tests
from rasterio.transform import from_origin

EAST, NORTH = 500000.0, 2100000.0


def _scene(folder, name, east, north, size, value, res=0.28):
    path = Path(folder) / name
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=1, dtype="uint16", crs="EPSG:32643", transform=from_origin(east, north, res, res)) as f:
        f.write(np.full((1, size, size), value, dtype=np.uint16))
    return path


def _lonlat(easting, northing):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform(easting, northing)


def test_two_scenes_become_one_map_and_first_wins_in_the_overlap():
    from header.mosaic_map import MosaicMap
    with tempfile.TemporaryDirectory() as folder:
        a = _scene(folder, "a.tif", EAST, NORTH, 1000, 1000)                     # covers x 0..280 m, y 0..280 m
        b = _scene(folder, "b.tif", EAST + 140, NORTH - 140, 1000, 3000)          # shifted by half: overlap x 140..280, y 140..280
        west, south = _lonlat(EAST + 20, NORTH - 260)
        east, north = _lonlat(EAST + 400, NORTH - 20)
        mosaic = MosaicMap([a, b], (west, south, east, north), gsd=0.3)
        grey = mosaic.read_strip(0, mosaic.height // 16 * 16)
        assert grey.shape[1] == mosaic.width
        assert (grey > 0).mean() > 0.5
        row, col = mosaic.gps_to_map_pixel(*_lonlat(EAST + 100, NORTH - 60)[::-1])      # only in scene a
        assert abs(grey[int(row), int(col)] - 1000) < 1
        row, col = mosaic.gps_to_map_pixel(*_lonlat(EAST + 200, NORTH - 200)[::-1])     # in the overlap: a (listed first) wins
        assert abs(grey[int(row), int(col)] - 1000) < 1
        row, col = mosaic.gps_to_map_pixel(*_lonlat(EAST + 350, NORTH - 250)[::-1])     # only in scene b
        assert abs(grey[int(row), int(col)] - 3000) < 1


def test_metres_to_gps_undoes_gps_to_map_pixel():
    from header.mosaic_map import MosaicMap
    with tempfile.TemporaryDirectory() as folder:
        a = _scene(folder, "a.tif", EAST, NORTH, 1000, 1000)
        west, south = _lonlat(EAST, NORTH - 280)
        east, north = _lonlat(EAST + 280, NORTH)
        mosaic = MosaicMap([a], (west, south, east, north), gsd=0.3)
        lon, lat = _lonlat(EAST + 123.0, NORTH - 77.0)
        row, col = mosaic.gps_to_map_pixel(lat, lon)
        back_lat, back_lon = mosaic.metres_to_gps(col * mosaic.gsd, row * mosaic.gsd)
        assert abs(back_lat - lat) < 1e-9 and abs(back_lon - lon) < 1e-9


def test_region_outside_every_scene_is_refused():
    from header.mosaic_map import MosaicMap
    with tempfile.TemporaryDirectory() as folder:
        a = _scene(folder, "a.tif", EAST, NORTH, 100, 1000)
        west, south = _lonlat(EAST + 5000, NORTH - 5100)
        east, north = _lonlat(EAST + 5100, NORTH - 5000)
        try:
            MosaicMap([a], (west, south, east, north), gsd=0.3)
        except AssertionError:
            return
        raise AssertionError("a region outside the scenes must be refused")


run_tests(globals())
