"""Final, honest check: run the whole search over the FULL score maps (not candidate lists) for photos never used to learn.

For every photo it records the error of the best peak and of the best of 25 peaks, for each single map and for the
fused map. Reports the shares within 1, 2 and 4 sigma, the median error in sigma, and a paired bootstrap against edge_zncc.
"""
import time
import traceback

import numpy as np
import torch

from coarse_search.fusion import fuse_maps
from header import settings
from header.settings import CELL, DEVICE
from image_processing.geometry import render_query
from train import config
from train.maps.map_bank import compute_maps, nms_peaks, z_stack
from train.train_fusion import paired_bootstrap


def peak_errors(score, usable, geometry, truth_m, n_peaks=25):
    gap = max(settings.TILE // CELL, geometry["side"] // 2)
    peaks = nms_peaks(score, usable, gap, n_peaks)
    errs = [float(np.hypot(float(geometry["x_m"][c]) - truth_m[0], float(geometry["y_m"][r]) - truth_m[1])) for r, c, v in peaks]
    return errs if errs else [1e9]


def evaluate_photos(testset, store, families, ids, single_maps, fused_models, log=print):
    """fused_models: dict label -> (model, subset of map names). Returns dict label -> arrays of [first_error_sigma, best25_error_sigma]."""
    names = sorted(set(single_maps) | {n for _, subset in fused_models.values() for n in subset}, key=lambda n: (n != "edge_zncc", n))
    fused_models = {label: (model.to(DEVICE).eval(), subset) for label, (model, subset) in fused_models.items()}
    results = {n: [] for n in single_maps}
    results.update({label: [] for label in fused_models})
    start = time.time()
    for count, i in enumerate(ids):
        i = int(i)
        level = int(testset.query_levels[i])
        row = {}
        try:
            canvas, valid = render_query(testset.items[i], testset.camera, level)
            maps, geometry = compute_maps(canvas, valid, level, testset.priors[i], families, names)
            z, usable, context = z_stack(maps, geometry, names)
            sigma = float(testset.sigmas[i])
            for n in single_maps:
                errs = peak_errors(maps[n], usable & (maps[n] > -1), geometry, testset.truths[i])
                row[n] = (errs[0] / sigma, min(errs) / sigma)
            for label, (model, subset) in fused_models.items():
                cols = [names.index(n) for n in subset]
                ctx_cols = [c for k in cols for c in (2 * k, 2 * k + 1)] + [2 * len(names)]
                fused = fuse_maps(model, z[..., cols], context[ctx_cols])
                errs = peak_errors(fused, usable, geometry, testset.truths[i])
                row[label] = (errs[0] / sigma, min(errs) / sigma)
        except Exception as error:
            log("photo %d failed: %s: %s" % (i, type(error).__name__, str(error)[:150]))
            if count == 0:
                log(traceback.format_exc())
            row = {}
        finally:
            for key in results:
                results[key].append(row.get(key, (np.inf, np.inf)))
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        if count % 20 == 0:
            log("full-map check: %d / %d photos, %.0f s" % (count + 1, len(ids), time.time() - start))
    return {k: np.array(v) for k, v in results.items()}


def summarise(results, baseline="edge_zncc"):
    rows = []
    base_hits = results[baseline][:, 0] <= 4
    for label, arr in results.items():
        first, best = arr[:, 0], arr[:, 1]
        d, lo, hi = paired_bootstrap(first <= 4, base_hits, n_boot=config.BOOTSTRAP)
        rows.append(dict(map=label, n=int(len(arr)), top1_within_1sigma=float((first <= 1).mean()), top1_within_2sigma=float((first <= 2).mean()),
                         top1_within_4sigma=float((first <= 4).mean()), ceiling25_within_4sigma=float((best <= 4).mean()),
                         median_error_sigma=float(np.median(first)), vs_baseline_4sigma=dict(diff=d, lo=lo, hi=hi)))
    return rows
