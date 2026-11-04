"""Stage setup: NUDT photos of all three splits, the camera, the 0.3 m map and the map store (levels 3 and 4)."""
import numpy as np

from evaluation.testset import TestSet
from header import settings
from image_processing.map_store import build_map_store
from train import config
from train.datasets import nudt_data


def load_world(args):
    if args.smoke:
        from train.datasets.smoke_world import make_smoke_world
        satellite_map, camera, by_split = make_smoke_world()
        return satellite_map, camera, by_split, {"train": 100, "valid": 100, "test": 100}
    from header.satellite_map import SatelliteMap
    nudt = nudt_data.configure()
    satellite_map = SatelliteMap(nudt_data.find_reference_map(nudt))
    by_split = {split: nudt_data.read_split(nudt, split) for split in ("train", "valid", "test")}
    counts = {"train": args.n_train or config.N_TRAIN_PHOTOS, "valid": args.n_valid or config.N_VALID_PHOTOS, "test": args.n_test or config.N_TEST_PHOTOS}
    return satellite_map, nudt_data.NudtCamera(), by_split, counts


def run_stage(run):
    satellite_map, camera, by_split, counts = load_world(run.args)
    run.log("map: %d x %d px, %.2f m/px" % (satellite_map.width, satellite_map.height, satellite_map.gsd))
    sets = {}
    for split, items in by_split.items():
        ids = nudt_data.spread_sample(items, counts[split], config.SEED)
        chosen = [items[i] for i in ids]
        flights = nudt_data.flight_labels(chosen)
        sets[split] = TestSet(chosen, camera, satellite_map, flights, np.arange(len(chosen)))
        run.log("  %-5s: %d photos from %d flights" % (split, len(chosen), len(set(flights))))
    run.log("building the map store (levels %s) ..." % settings.LEVELS)
    store = build_map_store(satellite_map)
    run.shared.update(sets=sets, store=store, camera=camera)
