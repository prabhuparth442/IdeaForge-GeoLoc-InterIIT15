"""Block: the photos to test on, as a TestSet, for the two data sets we have.

  load_official_photos  the official development set (d1 photos; the map comes from header/map_source.py: the new two-scene base map, or the old small_sat_area one)
  load_nudt_photos      NUDT Thermal-UAV test photos (Changsha), with the geometry found by the overnight run

Both return dict(test_set, camera, satellite_map) and leave the shared settings (zoom levels, geometry, photo folder) set for that data.
Import this module before anything changes the settings: it saves the development-set defaults when it is imported.
"""
import json

import numpy as np

from evaluation.testset import TestSet
from header import data, map_source, settings
from header.sampling import pick_sample_ids
from header.satellite_map import SatelliteMap
from train import config
from train.datasets import nudt_data

DEV_DEFAULTS = dict(levels=list(settings.LEVELS), geometry=dict(settings.GEOMETRY))


def load_official_photos(n_photos, seed=config.SEED, map_files=None, margin_m=None):
    dev_dir = config.find_dev_dir()
    settings.LEVELS = list(DEV_DEFAULTS["levels"])
    settings.GEOMETRY.clear()
    settings.GEOMETRY.update(DEV_DEFAULTS["geometry"])
    settings.IMAGES_DIR = dev_dir / "d1" / "images"
    settings.MAP_DIR = dev_dir / "small_sat_area"
    table = data.read_metadata()
    camera = data.Camera(table)
    all_items = data.make_items(table)
    all_flights = data.assign_flights(all_items)
    chosen = pick_sample_ids(all_flights, n_photos, seed)
    items = [all_items[i] for i in chosen]
    flights = np.array([all_flights[i] for i in chosen])
    satellite_map = map_source.load_map(items, map_files, margin_m)         # the base map(s): see header/map_source.py
    return dict(test_set=TestSet(items, camera, satellite_map, flights, np.arange(len(items))), camera=camera, satellite_map=satellite_map)


def load_nudt_photos(n_photos, seed=config.SEED, split="test", geometry_file=None):
    nudt = nudt_data.configure()
    saved = json.load(open(geometry_file or config.OUT_DIR / "geometry.json"))
    settings.GEOMETRY.update(saved["geometry"])
    camera = nudt_data.NudtCamera()
    satellite_map = SatelliteMap(nudt_data.find_reference_map(nudt))
    all_items = nudt_data.read_split(nudt, split)       # split: "test", "valid", "train" or "all" (the three together)
    for item in all_items:
        item["yaw"] = saved["yaw_sign"] * item["yaw_raw"]
    chosen = nudt_data.spread_sample(all_items, n_photos, seed)
    items = [all_items[i] for i in chosen]
    flights = nudt_data.flight_labels(items)
    return dict(test_set=TestSet(items, camera, satellite_map, flights, np.arange(len(items))), camera=camera, satellite_map=satellite_map)
