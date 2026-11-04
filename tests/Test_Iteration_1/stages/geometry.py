"""Stage geometry: try yaw signs, compass offsets and scales on a few train photos and keep the best (saved in geometry.json)."""
import itertools
import json

import numpy as np

from evaluation.geometry_sweep import share_within_4_sigma
from header import settings
from stages.helpers import apply_yaw_sign
from train import config


def run_stage(run):
    path = run.out / "geometry.json"
    test_sets = list(run.shared["sets"].values())
    if path.exists() and "geometry" not in run.args.redo:
        saved = json.load(open(path))
        settings.GEOMETRY.update(saved["geometry"])
        apply_yaw_sign(test_sets, saved["yaw_sign"])
        run.log("geometry from earlier run:", saved)
        return
    train_set, store = run.shared["sets"]["train"], run.shared["store"]
    rng = np.random.default_rng(config.SEED)
    count = 8 if run.args.smoke else config.GEOMETRY_PHOTOS
    ids = [int(i) for i in rng.choice(len(train_set.items), size=min(count, len(train_set.items)), replace=False)]
    base = dict(attitude=False, yaw_offset=0.0, scale=1.0, height_offset=0.0)
    signs = [1] if run.args.smoke else [1, -1]
    scales = [1.0] if run.args.smoke else [0.8, 1.0]
    scored = {}
    for sign, offset, scale in itertools.product(signs, (0, 90, 180, 270), scales):
        apply_yaw_sign(test_sets, sign)
        share, errors = share_within_4_sigma(train_set, store, dict(base, yaw_offset=float(offset), scale=scale), ids)
        scored[(sign, offset, scale)] = (share, -float(np.median(errors)))
        run.log("yaw sign %+d offset %3d scale %.2f: %.2f of photos within 4 sigma (median error %.1f sigma)" % (sign, offset, scale, share, -scored[(sign, offset, scale)][1]))
    sign, offset, scale = max(scored, key=lambda key: scored[key])
    apply_yaw_sign(test_sets, sign)
    if not run.args.smoke:
        for other_scale in (0.7, 0.75, 0.85, 0.9, 0.95, 1.05, 1.1):
            share, errors = share_within_4_sigma(train_set, store, dict(base, yaw_offset=float(offset), scale=other_scale), ids)
            scored[(sign, offset, other_scale)] = (share, -float(np.median(errors)))
            run.log("refine: scale %.2f: %.2f within 4 sigma" % (other_scale, share))
        sign, offset, scale = max(scored, key=lambda key: scored[key])
        apply_yaw_sign(test_sets, sign)
    geometry = dict(attitude=False, yaw_offset=float(offset), scale=float(scale), height_offset=0.0)
    settings.GEOMETRY.update(geometry)
    best_share = scored[(sign, offset, scale)][0]
    json.dump(dict(geometry=geometry, yaw_sign=sign, share_within_4_sigma=best_share), open(path, "w"))
    run.log("chosen: yaw sign %+d, geometry %s -> %.2f within 4 sigma" % (sign, geometry, best_share))
    run.set_result("geometry", dict(geometry, yaw_sign=sign, share_within_4_sigma=best_share))
    if best_share < 0.3 and not run.args.smoke:
        run.log("WARNING: fewer than 30 %% of photos land within 4 sigma even with the best geometry: the photo-to-map conventions are probably still wrong. "
                "Training on bad geometry teaches nothing. Look at geometry.json and the first lines of run.log before trusting later stages.")
