"""Random poses and saving examples with their metadata."""
import json
from pathlib import Path

import cv2
import numpy as np

from train.simulation.pose import make_pose
from train.simulation.render import render_view


def random_poses(rng, count, east_range, north_range, height_range, tilt_deg=0.0, yaw_range=(0.0, 360.0)):
    """count poses, uniform in position, height and yaw; pitch and roll uniform in +-tilt_deg."""
    poses = []
    for _ in range(count):
        poses.append(make_pose(rng.uniform(*east_range), rng.uniform(*north_range), rng.uniform(*height_range),
                               rng.uniform(*yaw_range), rng.uniform(-tilt_deg, tilt_deg), rng.uniform(-tilt_deg, tilt_deg)))
    return poses


def save_example(folder, name, result):
    """Writes name.png (the image), name_valid.png (mask, 255 = reliable) and name.json (all metadata)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / (name + ".png")), result["image"])
    cv2.imwrite(str(folder / (name + "_valid.png")), result["valid"].astype(np.uint8) * 255)
    with open(folder / (name + ".json"), "w") as handle:
        json.dump(result["meta"], handle, indent=1)


def generate(satellite_map, camera, poses, folder, seed, appearance="regional_detail", strength=1.0, prefix="sim", **render_options):
    """One example per pose. The seed of example i is (seed, i), so any single example can be remade alone."""
    results = []
    for index, pose in enumerate(poses):
        rng = np.random.default_rng([seed, index])
        result = render_view(satellite_map, camera, pose, rng, appearance, strength,
                             extra_meta=dict(seed=[seed, index], index=index), **render_options)
        save_example(folder, "%s_%04d" % (prefix, index), result)
        results.append(result)
    return results
