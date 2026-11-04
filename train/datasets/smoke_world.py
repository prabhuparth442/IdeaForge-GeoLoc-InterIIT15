"""A synthetic stand-in for NUDT (random texture map, photos cut from it) so the whole launcher can be tried in minutes, no dataset.
Photos are grouped into fake flights; splits train / valid / test are made of whole flights."""
import tempfile
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from header import settings
from header.satellite_map import SatelliteMap


def make_smoke_world(n_photos=48, flights=6, size=7000, gsd=0.3, seed=3):
    work = Path(tempfile.mkdtemp(prefix="smoke_world_"))
    (work / "images").mkdir()
    settings.IMAGES_DIR = work / "images"
    settings.LEVELS = [3, 4, 5]
    settings.GEOMETRY.update(attitude=False, yaw_offset=0.0, scale=1.0, height_offset=0.0)
    rng = np.random.default_rng(seed)
    base = rng.random((size, size)).astype(np.float32)
    tex = sum(cv2.GaussianBlur(base, (0, 0), s) / cv2.GaussianBlur(base, (0, 0), s).std() for s in (2, 6, 20))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    grey = (20 + 230 * tex).astype(np.uint8)
    path = work / "ref.tif"
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8",
                       crs="EPSG:32644", transform=from_origin(500000.0, 2000000.0, gsd, gsd)) as dst:
        for band in (1, 2, 3):
            dst.write(grey, band)
    fx = 200.0
    to_lonlat = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    items = []
    for k in range(n_photos):
        r, c = int(rng.integers(900, size - 900)), int(rng.integers(900, size - 900))
        alt = float(rng.uniform(100, 140))
        u, v = np.meshgrid(np.arange(320), np.arange(256))
        east = (u - 159.5) / fx * alt
        south = (v - 127.5) / fx * alt
        soft = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), alt / fx / gsd / 2)
        photo = cv2.remap(soft, (c + east / gsd).astype(np.float32), (r + south / gsd).astype(np.float32), cv2.INTER_LINEAR)
        name = work / "images" / ("p%d.png" % k)
        cv2.imwrite(str(name), (photo * 200).astype(np.uint16))
        lon, lat = to_lonlat.transform(500000.0 + c * gsd, 2000000.0 - r * gsd)
        items.append(dict(name=str(name), lat=float(lat), lon=float(lon), rel_height=alt, yaw=0.0, yaw_raw=0.0, pitch=0.0, roll=0.0,
                          flight="flight%d" % (k * flights // n_photos)))

    camera = SimpleNamespace(fx=fx, fy=fx, cx=159.5, cy=127.5, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, width=320, height=256)
    split_of = {"flight%d" % f: ("train" if f < flights - 2 else "valid" if f == flights - 2 else "test") for f in range(flights)}
    by_split = {s: [it for it in items if split_of[it["flight"]] == s] for s in ("train", "valid", "test")}
    return SatelliteMap(path), camera, by_split


def write_dev_dir(root, n_photos=12, size=7000, gsd=0.3, seed=11):
    """A tiny stand-in for the official development set (d1/images + metadata.csv, small_sat_area/*.tif), heights spread over the three groups."""
    import pandas as pd
    root = Path(root)
    (root / "d1" / "images").mkdir(parents=True)
    (root / "small_sat_area").mkdir()
    rng = np.random.default_rng(seed)
    base = rng.random((size, size)).astype(np.float32)
    tex = sum(cv2.GaussianBlur(base, (0, 0), s) / cv2.GaussianBlur(base, (0, 0), s).std() for s in (2, 6, 20))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    grey = (20 + 230 * tex).astype(np.uint8)
    with rasterio.open(root / "small_sat_area" / "m.tif", "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8",
                       crs="EPSG:32644", transform=from_origin(500000.0, 2000000.0, gsd, gsd)) as dst:
        for band in (1, 2, 3):
            dst.write(grey, band)
    fx = 200.0
    to_lonlat = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    heights = [110, 130, 140, 120, 170, 200, 230, 240, 260, 280, 300, 320][:n_photos]
    rows = []
    for k, alt in enumerate(heights):
        r, c = int(rng.integers(1200, size - 1200)), int(rng.integers(1200, size - 1200))
        u, v = np.meshgrid(np.arange(320), np.arange(256))
        east = (u - 159.5) / fx * alt
        south = (v - 127.5) / fx * alt
        soft = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), alt / fx / gsd / 2)
        photo = cv2.remap(soft, (c + east / gsd).astype(np.float32), (r + south / gsd).astype(np.float32), cv2.INTER_LINEAR)
        cv2.imwrite(str(root / "d1" / "images" / ("d%d.png" % k)), (photo * 200).astype(np.uint16))
        lon, lat = to_lonlat.transform(500000.0 + c * gsd, 2000000.0 - r * gsd)
        rows.append(dict(filename="d%d.png" % k, lat_gtruth=lat, lon_gtruth=lon, alt=float(alt), yaw=180.0, pitch=0.0, roll=0.0,
                         cam_fx=fx, cam_fy=fx, cam_cx=159.5, cam_cy=127.5, cam_k1=0.0, cam_k2=0.0, cam_k3=0.0, cam_p1=0.0, cam_p2=0.0))
    pd.DataFrame(rows).to_csv(root / "d1" / "images" / "metadata.csv", index=False)
    return root
