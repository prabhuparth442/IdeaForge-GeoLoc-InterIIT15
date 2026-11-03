import numpy as np
import torch

from coarse_search.search import search_frame
from header import settings
from header.settings import DEVICE, SEED
from image_processing.geometry import render_query


def share_for(testset, store, ids, geometry):
    hits = []
    for i in ids:
        level = testset.query_levels[i]
        canvas, valid = render_query(testset.items[i], testset.camera, level, geometry)
        found, _ = search_frame(canvas, valid, level, testset.priors[i], store, variants=[("zncc", 0.0)])
        peak = found[("zncc", 0.0)][0]
        hits.append(testset.error_m(i, peak[1], peak[2]) / testset.sigmas[i] <= 4)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(hits))


def run_sensitivity(testset, store):
    rng = np.random.default_rng(SEED + 1)
    ids = [int(i) for i in rng.choice(testset.sample_ids, size=min(18, len(testset.sample_ids)), replace=False)]
    base = settings.GEOMETRY
    for delta in [-12, -6, -3, 0, 3, 6, 12]:
        share = share_for(testset, store, ids, dict(base, yaw_offset=base["yaw_offset"] + delta))
        print("heading error %+3d degrees: share within 4 sigma %.2f" % (delta, share))
    for factor in [0.85, 0.92, 1.0, 1.08, 1.15]:
        share = share_for(testset, store, ids, dict(base, scale=base["scale"] * factor))
        print("scale factor %.2f: share within 4 sigma %.2f" % (factor, share))
