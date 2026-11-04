"""Shared helpers for the test scripts.

* run_tests(globals())  : runs every function called test_* in the calling script, prints PASS/FAIL.
* --swap target=spec    : replace any function of the repo with your own, for this run only.
        target = "package.module.function"            e.g. fine_matching.ransac.fit_transform
        spec   = "package.module:function"             a function from an installed/importable module
              or "path/to/your_script.py:function"    a function from a script file
  Example:  python tests/pipeline/test_result_thresholding.py --swap result_thresholding.decision.decide=my_rules.py:decide
* world()               : a small synthetic map + 3 photos with known truth (no dataset needed), built once.
"""
import argparse
import importlib
import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


def load_function(spec):
    where, name = spec.rsplit(":", 1)
    if where.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(Path(where).stem, where)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    return getattr(module, name)


def apply_swap(text):
    assert "=" in text, "--swap needs target=spec, got %r" % text
    target, spec = text.split("=", 1)
    module_name, attribute = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, attribute), "%s has no attribute %s" % (module_name, attribute)
    setattr(module, attribute, load_function(spec))
    print("swapped %s -> %s" % (target, spec))
    # other modules that did 'from x import f' keep the old name, so patch those too
    for other in list(sys.modules.values()):
        if other is not module and getattr(other, attribute, None) is not None and getattr(getattr(other, attribute), "__module__", None) == module_name:
            setattr(other, attribute, getattr(module, attribute))


def read_arguments(extra=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap", action="append", default=[], help="target=spec, see common.py")
    parser.add_argument("--only", default="", help="run only tests whose name contains this text")
    if extra:
        extra(parser)
    args, _ = parser.parse_known_args()
    return args


def run_tests(namespace, extra=None):
    args = read_arguments(extra)
    for text in args.swap:
        apply_swap(text)
    tests = [(n, f) for n, f in namespace.items() if n.startswith("test_") and callable(f) and args.only in n]
    failed = 0
    for name, function in tests:
        try:
            function()
            print("PASS  %s" % name)
        except Exception:
            failed += 1
            print("FAIL  %s" % name)
            traceback.print_exc()
    print("%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)


_WORLD = {}


def world(size=6000, gsd=0.3):
    """Synthetic map with three 'photos' cut from it. Returns a dict with items, camera, map, store, testset."""
    if _WORLD:
        return _WORLD
    import cv2
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    from header import settings
    work = Path(tempfile.mkdtemp())
    (work / "images").mkdir()
    (work / "map").mkdir()
    settings.IMAGES_DIR = work / "images"
    settings.MAP_DIR = work / "map"
    settings.GEOMETRY.update(attitude=False, yaw_offset=0.0, scale=1.0, height_offset=0.0)

    rng = np.random.default_rng(3)
    base = rng.random((size, size)).astype(np.float32)
    tex = sum(cv2.GaussianBlur(base, (0, 0), s) / cv2.GaussianBlur(base, (0, 0), s).std() for s in (2, 6, 20))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    grey = (20 + 230 * tex).astype(np.uint8)
    with rasterio.open(work / "map" / "m.tif", "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8",
                       crs="EPSG:32644", transform=from_origin(500000.0, 2000000.0, gsd, gsd)) as dst:
        for band in (1, 2, 3):
            dst.write(grey, band)

    fx = 200.0
    camera_columns = dict(cam_fx=fx, cam_fy=fx, cam_cx=159.5, cam_cy=127.5, cam_k1=0.0, cam_k2=0.0, cam_k3=0.0, cam_p1=0.0, cam_p2=0.0)
    to_lonlat = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    rows = []
    for k, (r, c, alt) in enumerate([(3000, 3000, 100.0), (2500, 3400, 120.0), (3500, 2600, 140.0)]):
        u, v = np.meshgrid(np.arange(320), np.arange(256))
        east = (u - 159.5) / fx * alt
        south = (v - 127.5) / fx * alt
        soft = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), alt / fx / gsd / 2)
        photo = cv2.remap(soft, (c + east / gsd).astype(np.float32), (r + south / gsd).astype(np.float32), cv2.INTER_LINEAR)
        cv2.imwrite(str(work / "images" / ("p%d.png" % k)), (photo * 200).astype(np.uint16))
        lon, lat = to_lonlat.transform(500000.0 + c * gsd, 2000000.0 - r * gsd)
        rows.append(dict(filename="p%d.png" % k, lat_gtruth=lat, lon_gtruth=lon, alt=alt, yaw=0.0, pitch=0.0, roll=0.0, **camera_columns))
    pd.DataFrame(rows).to_csv(work / "images" / "metadata.csv", index=False)

    from evaluation.testset import TestSet
    from header.data import Camera, assign_flights, find_map_file, make_items, read_metadata
    from header.satellite_map import SatelliteMap
    from image_processing.map_store import build_map_store
    table = read_metadata()
    items = make_items(table)
    camera = Camera(table)
    flights = assign_flights(items)
    satellite_map = SatelliteMap(find_map_file())
    testset = TestSet(items, camera, satellite_map, flights, np.array([0, 1, 2]))
    store = build_map_store(satellite_map)
    _WORLD.update(items=items, camera=camera, map=satellite_map, store=store, testset=testset, flights=flights)
    return _WORLD
