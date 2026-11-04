"""Stage setup (synthetic run): NUDT test photos for the check, the places fake photos may come from, the map store (levels 3 to 7)."""
import json
from pathlib import Path

import numpy as np

from evaluation.testset import TestSet, truth_in_metres
from header import settings
from header.satellite_map import SatelliteMap
from image_processing.map_store import build_map_store
from stages.helpers import build_families
from train import config
from train.datasets import nudt_data
from train.datasets.place_picker import PlacePicker
from train.datasets.synthetic_candidates import LEVELS


def run_stage(run):
    real_out = Path(run.args.real_out)
    nudt = nudt_data.configure()
    saved = json.load(open(real_out / "geometry.json"))
    settings.GEOMETRY.update(saved["geometry"])
    camera = nudt_data.NudtCamera()
    satellite_map = SatelliteMap(nudt_data.find_reference_map(nudt))
    run.log("map: %d x %d px, %.2f m/px" % (satellite_map.width, satellite_map.height, satellite_map.gsd))

    avoid_points = []
    test_items = []
    for split in ("valid", "test"):
        items = nudt_data.read_split(nudt, split)
        for item in items:
            item["yaw"] = saved["yaw_sign"] * item["yaw_raw"]
            avoid_points.append(truth_in_metres(item, satellite_map))
        if split == "test":
            test_items = items
    chosen = nudt_data.spread_sample(test_items, run.args.n_eval_nudt, config.SEED)
    items = [test_items[i] for i in chosen]
    test_set = TestSet(items, camera, satellite_map, nudt_data.flight_labels(items), np.arange(len(items)))
    run.log("real NUDT test photos for the check: %d" % len(items))

    picker = PlacePicker(satellite_map, avoid_points)
    run.log("places for fake photos: %.0f km2 allowed, %.0f km2 kept free around the valid/test photos" % (picker.allowed_km2, picker.avoided_km2))

    settings.LEVELS = list(LEVELS)          # the real test photos got their level above (3 or 4); fake photos use 3 to 7
    run.log("building the map store (levels %s) ..." % settings.LEVELS)
    store = build_map_store(satellite_map)
    names, families = build_families(store, real_out / "models" / "f4.pt", run.log)
    run.shared.update(satellite_map=satellite_map, camera=camera, test_set=test_set, picker=picker, store=store, names=names, families=families)
