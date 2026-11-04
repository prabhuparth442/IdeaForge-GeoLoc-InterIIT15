"""Block: the same check on the official development set (d1 photos on the small_sat_area map: another city, camera, sensor and map).
If a gain seen on NUDT shows here too, it is not overfitting to Changsha."""
from evaluation.testset import TestSet
from header import data, settings
from header.sampling import pick_sample_ids
from header.satellite_map import SatelliteMap
from image_processing.map_store import build_map_store
from stages.dev_defaults import apply_dev_settings
from stages.helpers import build_families, log_rows
from train import config
from train.checks import full_eval


def evaluate_on_official_set(run, load_fused_models, result_key, n_photos, f4_file, title="OFFICIAL"):
    fused_models = load_fused_models()
    run.free_memory()
    apply_dev_settings(config.find_dev_dir())
    table = data.read_metadata()
    items = data.make_items(table)
    camera = data.Camera(table)
    flights = data.assign_flights(items)
    satellite_map = SatelliteMap(data.find_map_file())
    ids = pick_sample_ids(flights, n_photos, config.SEED)
    run.log("official set: %d photos tested of %d, map %d x %d px at %.2f m/px, geometry %s" % (
        len(ids), len(items), satellite_map.width, satellite_map.height, satellite_map.gsd, dict(settings.GEOMETRY)))
    test_set = TestSet(items, camera, satellite_map, flights, ids)
    store = build_map_store(satellite_map)
    names, families = build_families(store, f4_file, run.log)
    results = full_eval.evaluate_photos(test_set, store, families, [int(i) for i in ids], names, fused_models, log=run.log)
    rows = full_eval.summarise(results)
    run.set_result(result_key, rows)
    log_rows(run, title, rows)
    run.free_memory()
