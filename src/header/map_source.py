"""Block to REPLACE when the base map changes: which map files the official (d1) photos are searched on.

load_map(items) is the one function the pipeline calls (train.datasets.photo_sets.load_official_photos). It returns an object with the members
of header.satellite_map.SatelliteMap (width, height, gsd, gray_max, read_strip, grey_to_uint8, gps_to_map_pixel) and metres_to_gps.

To use another map: pick one by name from MAPS ("new", "new_931", "new_921", "old": the MAP setting of the notebooks), add your own entry to MAPS (any number of GeoTIFFs in one CRS), or write your own load_map(items) here.
  "old"  -> the old single map: dataset/small_sat_area/*.tif (6 bands, 0.3 m), read as before
Several files are read as one map (header.mosaic_map.MosaicMap), cut to the photos' bounding box plus MARGIN_M on every side.
MARGIN_M: the search disc (settings.DISC_RADIUS, 5.5 km) reaches that far from a photo, so the map needs more than that around the photos (6000); a smaller margin (say 2000) builds a smaller, faster map store
but places the edge of the map closer to the photos.
"""
import math

from header import data, settings
from header.mosaic_map import MosaicMap
from header.satellite_map import SatelliteMap

NEW_BASE_MAP = settings.DATASET_DIR / "Dataset_Official_2"
SCENE_A = NEW_BASE_MAP / "268234931" / "BAND.tif"       # covers all the d1 photos
SCENE_B = NEW_BASE_MAP / "268234921" / "BAND.tif"       # reaches only the south of them
OLD_MAP = "old"        # the old single map: dataset/small_sat_area
MAPS = {                                                # the maps you can choose by name (the MAP setting of the notebooks)
    "new": [SCENE_A, SCENE_B],          # both scenes of the new base map, read as one (first listed wins in the overlap)
    "new_931": [SCENE_A],               # only scene 268234931
    "new_921": [SCENE_B],               # only scene 268234921
    "old": OLD_MAP,
}
DEFAULT_MAP = "new"
MAP_FILES = MAPS[DEFAULT_MAP]
MARGIN_M = 6000


def region_of_interest(items, margin_m=MARGIN_M):
    """(west, south, east, north) in degrees: the bounding box of the photos, grown by margin_m."""
    assert len(items) > 0, "no photos"
    lats = [item["lat"] for item in items]
    lons = [item["lon"] for item in items]
    middle = (min(lats) + max(lats)) / 2
    d_lat = margin_m / 111320.0
    d_lon = margin_m / (111320.0 * math.cos(math.radians(middle)))
    return min(lons) - d_lon, min(lats) - d_lat, max(lons) + d_lon, max(lats) + d_lat


def load_map(items, map_files=None, margin_m=None):
    """The satellite map for these photos. map_files / margin_m override MAP_FILES / MARGIN_M for this call (map_files=None means: use MAP_FILES; OLD_MAP: the old map)."""
    files = MAP_FILES if map_files is None else map_files
    if isinstance(files, str) and files != OLD_MAP:
        assert files in MAPS, "unknown map %r, choose one of %s" % (files, list(MAPS))
        files = MAPS[files]
    if files is None or files == OLD_MAP:
        return SatelliteMap(data.find_map_file())
    for path in files:
        assert path.exists(), "map file not found: %s" % path
    return MosaicMap(files, region_of_interest(items, MARGIN_M if margin_m is None else margin_m))
