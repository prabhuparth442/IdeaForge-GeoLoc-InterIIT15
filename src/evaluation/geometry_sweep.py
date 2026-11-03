import time

import numpy as np
import torch

from coarse_search.search import search_frame
from header import settings
from header.settings import DEVICE, SEED
from image_processing.geometry import render_query


def share_within_4_sigma(testset, store, geometry, ids):
    assert len(ids) > 0, "no photo ids given"
    errors_in_sigma = []
    for i in ids:
        level = testset.query_levels[i]
        canvas, valid = render_query(testset.items[i], testset.camera, level, geometry)
        found, _ = search_frame(canvas, valid, level, testset.priors[i], store, variants=[("zncc", 0.0)])
        peak = found[("zncc", 0.0)][0]
        errors_in_sigma.append(testset.error_m(i, peak[1], peak[2]) / testset.sigmas[i])
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    errors_in_sigma = np.array(errors_in_sigma)
    return (errors_in_sigma <= 4).mean(), errors_in_sigma


def tilt_of(item):
    return abs(item["pitch"]) + abs(item["roll"])


def sweep_geometry(testset, store):
    items = testset.items
    n_frames = len(items)
    geometry = dict(attitude=True, pitch_sign=1, roll_sign=1, yaw_offset=0.0, scale=1.0, height_offset=0.0)

    tilted = [i for i in range(n_frames) if tilt_of(items[i]) >= 6]
    rng = np.random.default_rng(SEED)
    check_ids = [int(i) for i in rng.choice(tilted, size=min(40, len(tilted)), replace=False)]
    print("testing %d tilted photos (of %d)" % (len(check_ids), len(tilted)))

    flat = [i for i in range(n_frames) if abs(items[i]["pitch"]) < 8 and abs(items[i]["roll"]) < 8]
    flat_ids = [int(i) for i in np.random.default_rng(SEED + 1).choice(flat, size=min(24, len(flat)), replace=False)]
    yaw_sweep = {}
    for offset in [0, 90, 180, 270]:
        share, errors = share_within_4_sigma(testset, store, dict(geometry, attitude=False, yaw_offset=offset), flat_ids)
        yaw_sweep[offset] = (share, float(np.median(errors)))
        print("yaw offset %3d degrees: share within 4 sigma %.2f, median error %.1f sigma" % (offset, share, yaw_sweep[offset][1]))
    best_offset = max(yaw_sweep, key=lambda o: (yaw_sweep[o][0], -yaw_sweep[o][1]))
    geometry["yaw_offset"] = float(best_offset)
    print("chosen yaw offset:", best_offset, "degrees")

    hypotheses = [("no tilt correction", dict(attitude=False)),
                  ("p+ r+", dict(attitude=True, pitch_sign=1, roll_sign=1)),
                  ("p- r+", dict(attitude=True, pitch_sign=-1, roll_sign=1)),
                  ("p+ r-", dict(attitude=True, pitch_sign=1, roll_sign=-1)),
                  ("p- r-", dict(attitude=True, pitch_sign=-1, roll_sign=-1))]
    results = {}
    start = time.time()
    for name, change in hypotheses:
        share, errors = share_within_4_sigma(testset, store, dict(geometry, **change), check_ids)
        results[name] = (share, errors)
        print("%-20s share within 4 sigma: %.2f   median error: %.1f sigma" % (name, share, np.median(errors)))
    print("took %.0f s" % (time.time() - start))

    best_name = max(results, key=lambda name: (results[name][0], -np.median(results[name][1])))
    best_change = dict(hypotheses)[best_name]
    share_minus4, errors_minus4 = share_within_4_sigma(testset, store, dict(geometry, **best_change, height_offset=-4.0), check_ids)
    print("%-20s share within 4 sigma: %.2f   median error: %.1f sigma" % ("best, alt - 4", share_minus4, np.median(errors_minus4)))

    geometry.update(best_change)
    if share_minus4 > results[best_name][0] + 0.05:
        geometry["height_offset"] = -4.0
    print("chosen geometry:", geometry)

    scale_values = [0.96, 1.0, 1.04, 1.08, 1.12, 1.16]
    scale_share = []
    for value in scale_values:
        share, errors = share_within_4_sigma(testset, store, dict(geometry, scale=value), check_ids)
        scale_share.append(share)
        print("scale %.2f: share within 4 sigma %.2f, median error %.1f sigma" % (value, share, np.median(errors)))
    geometry["scale"] = scale_values[int(np.argmax(scale_share))]

    print("final geometry:", geometry)
    print("settings.GEOMETRY used so far:", settings.GEOMETRY, "->", "same" if geometry == settings.GEOMETRY else "DIFFERENT: update GEOMETRY in header/settings.py")
    return geometry
