import time

from header import settings
from image_processing.edge_features import build_map_features
from image_processing.map_levels import build_level_pictures, level_gsd


def build_map_store(satellite_map):
    assert abs(satellite_map.gsd - settings.MAP_GSD) < 1e-3, "map pixel size %.3f m differs from settings.MAP_GSD %.3f" % (satellite_map.gsd, settings.MAP_GSD)
    start = time.time()
    level_images = build_level_pictures(satellite_map)
    for level in settings.LEVELS:
        shape = level_images[level].shape
        print("level %d: %.2f m/pixel, picture %d x %d, map coverage %.0f%%" % (
            level, level_gsd(level), shape[1], shape[0], 100 * (level_images[level] > 0).mean()))
    map_features = build_map_features(level_images)
    print("map store ready, took %.0f s" % (time.time() - start))
    return {"levels": level_images, "features": map_features}
