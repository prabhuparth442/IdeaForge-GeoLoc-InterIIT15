"""NUDT Thermal-UAV (Changsha) as input for the existing pipeline: photos, camera, map, splits.

The pipeline code expects the layout of the official development set. NUDT differs in three ways, handled here:
  * file names are full paths (load_thermal_frame joins them with IMAGES_DIR; an absolute path wins, so it works untouched);
  * the camera is 1280x1024, focal 12 mm on a 9.83 mm wide sensor -> fx = 1562.5 px, no lens distortion in the table;
  * pitch is always -90 and roll 180 (camera straight down in the vendor convention) -> the tilt correction is switched off.
"""
import glob
import os

import numpy as np
import pandas as pd

from header import settings
from train import config

SENSOR_WIDTH_MM = 9.83
FOCAL_MM = 12.0
FRAME_W, FRAME_H = 1280, 1024


class NudtCamera:
    """Same attributes as header.data.Camera (fx, fy, cx, cy, k1.., width, height), made from the lens numbers."""

    def __init__(self):
        self.fx = self.fy = FOCAL_MM / SENSOR_WIDTH_MM * FRAME_W
        self.cx, self.cy = (FRAME_W - 1) / 2, (FRAME_H - 1) / 2
        self.k1 = self.k2 = self.k3 = self.p1 = self.p2 = 0.0
        self.width, self.height = FRAME_W, FRAME_H


def configure(levels=None):
    """Point the shared settings at NUDT. Call once, before building the map store."""
    nudt = config.find_nudt_dir()
    settings.LEVELS = list(levels or config.NUDT_LEVELS)
    settings.IMAGES_DIR = nudt / "Thermal-UAV"
    settings.GEOMETRY.update(attitude=False, yaw_offset=0.0, scale=1.0, height_offset=0.0)
    settings.USE_MINIMA = False
    return nudt


def find_reference_map(nudt):
    path = nudt / "Reference_map" / "changsha" / "ref.tif"
    assert path.exists(), "reference map not found: %s (dsm.tif next to it is the height map, not the picture)" % path
    return path


def read_split(nudt, split):
    """All frames of one split as pipeline items, plus the flight name of each (the unit that must never be split)."""
    assert split in ("train", "valid", "test"), "split must be train, valid or test"
    items = []
    for flight_dir in sorted(glob.glob(str(nudt / "Thermal-UAV" / split / "changsha" / "*"))):
        flight = os.path.basename(flight_dir)
        table = pd.read_csv(os.path.join(flight_dir, "Thermal_info.csv"))
        for column in ["filename", "lat", "lon", "rel_height", "yaw"]:
            assert column in table.columns, "%s/Thermal_info.csv has no column %s" % (flight, column)
        for _, row in table.iterrows():
            path = os.path.join(flight_dir, "Thermal", row["filename"])
            items.append(dict(name=path, lat=float(row["lat"]), lon=float(row["lon"]), rel_height=float(row["rel_height"]),
                              yaw=float(row["yaw"]), yaw_raw=float(row["yaw"]), pitch=0.0, roll=0.0, flight=flight, split=split))
    assert len(items) > 0, "no frames found for split " + split
    return items


def spread_sample(items, n, seed):
    """n photo indices spread evenly over the flights (every flight contributes, big flights contribute more)."""
    assert n > 0, "n must be positive"
    rng = np.random.default_rng(seed)
    by_flight = {}
    for i, item in enumerate(items):
        by_flight.setdefault(item["flight"], []).append(i)
    n = min(n, len(items))
    chosen = []
    for flight, members in sorted(by_flight.items()):
        share = max(1, int(round(n * len(members) / len(items))))
        chosen += [int(m) for m in rng.choice(members, size=min(share, len(members)), replace=False)]
    chosen = sorted(set(chosen))
    if len(chosen) > n:
        chosen = sorted(int(c) for c in rng.choice(chosen, size=n, replace=False))
    return np.array(chosen)


def flight_labels(items):
    """Per photo, a flight name that satisfies assign_flights-style checks (we use the real flight)."""
    return np.array([item["flight"] for item in items])
