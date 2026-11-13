"""Block: every signal available after fine matching, for every candidate of a rich dump (train/checks/rich_dump.py). No GPU, no plotting.

Test_Iteration_7 looks only at the decision after fine matching, without multi-query agreement. Two levers:
  (a) more lines for right places: other fits of the same stored LoFTR matches (lower confidence cut guided by the 0.22 fit, translation-only
      consensus, LO-RANSAC (least-squares re-fit on the inliers));
  (b) better judging: signals that should mean the same on every data set (a-contrario significance, spatial coverage of the inliers, fit
      plausibility, DINO z, coarse rank), instead of the raw inlier count, whose meaning differs between official and NUDT.

candidate_signals(match, inside, level_gsd_m, ...)   one candidate's matches -> dict of signals (all floats)
photo_signal_table(photo, photo_id, data, ...)        one row per stored candidate (27 queries x M) with the pipeline's stored inliers, z, coarse
                                                      score, rank, error in sigma and every signal of candidate_signals
build_signal_table(dump_dir, data, workers)           photo_signal_table over every photo of a dump folder, in parallel processes
log10_nfa(n, k, p, n_tests)                           a-contrario number of false alarms (log10) of k inliers among n pairs with chance rate p

Why each signal (the logic, before any data):
- translation consensus (inl_t2, inl_t3): the query canvas is already north-up at the guessed scale, so a right place is close to a pure shift.
  A similarity fit has 4 degrees of freedom and a +-15 % / +-12 deg gate in which chance structures (repeated buildings) can line up; a shift
  has 2, so chance consensus is rarer while the right place keeps its lines.
- a-contrario NFA (lnfa_t3): how surprising k consistent lines are among n pairs if lines were random. 8 inliers out of 12 pairs is strong,
  8 out of 300 is not. The inlier count alone ignores n.
- guided lines (guided15): matches between confidence 0.15 and 0.22 that agree with the 0.22 fit. A right fit should collect them (they are real
  but faint lines); a chance fit should not.
- coverage (cover9, hull, spread): a right match lines up the whole photographed area; a look-alike building gives a local cluster.
- fit plausibility (scale_dev, angle, shift_px): a right fit sits near scale 1, angle 0 (the query already holds the heading guess) and a shift
  inside the coarse cell; a chance fit anywhere in the gate.
- high-confidence share (hc_ratio): LoFTR's confident pairs mostly belong to the right transform on a right place.
"""
import math

import cv2
import numpy as np
from scipy.stats import binom

from fine_matching.ransac import fit_transform
from header.settings import CELL, MAX_SHIFT_CELLS, RANSAC_PIXELS, SEED

CONF = 0.22            # the pipeline's matcher confidence cut
LOW_CONF = 0.15        # lowest confidence kept in the rich dump
SHIFT_LIMIT_PX = MAX_SHIFT_CELLS * CELL      # the pipeline drops fits whose centre moves further (margin 0 in every dump)
DISPLACEMENT_AREA = (2 * 224.0) ** 2         # a random pair's shift b - a is spread over about this many square pixels (canvas and tile 224)

EMPTY = dict(n15=0.0, n22=0.0, inl_s3=0.0, inl_t2=0.0, inl_t3=0.0, inl_t3_15=0.0, guided15=0.0, inl_lo=0.0, lnfa_t3=0.0, lnfa_s3=0.0,
             cover9=0.0, hull=0.0, spread=0.0, resid=RANSAC_PIXELS, conf_inl=0.0, hc_ratio=0.0, scale_dev=0.15, angle=12.0, shift_px=99.0,
             t_shift_px=99.0, t_dx=0.0, t_dy=0.0)


def log10_nfa(n, k, p, n_tests):
    """log10 of n_tests * P(Binomial(n, p) >= k). Very negative = very unlikely by chance. 0 when k = 0."""
    if k <= 0 or n <= 0:
        return 0.0
    tail = max(binom.logsf(k - 1, n, p) / math.log(10), -1000.0)       # floor: the tail underflows to -inf for very strong fits
    return float(math.log10(max(n_tests, 1)) + tail)


def translation_consensus(a, b, radius):
    """Largest set of pairs whose shift b - a lies within `radius` px of one shift (exact: every pair's shift is tried as the centre, then the
    centre is refined to the mean of its set once). Returns (count, mask, centre)."""
    if len(a) == 0:
        return 0, np.zeros(0, bool), np.zeros(2)
    d = (b - a).astype(np.float64)
    dist = np.hypot(d[:, None, 0] - d[None, :, 0], d[:, None, 1] - d[None, :, 1])
    counts = (dist <= radius).sum(1)
    centre = d[dist[int(np.argmax(counts))] <= radius].mean(0)
    mask = np.hypot(*(d - centre).T) <= radius
    return int(mask.sum()), mask, centre


def _coverage(points, inside, cells=3):
    """Share of the photographed 3 x 3 cells (cells with >= 25 % photographed pixels) that hold at least one inlier."""
    size = inside.shape[0]
    edges = np.linspace(0, size, cells + 1).astype(int)
    total = hit = 0
    for i in range(cells):
        for j in range(cells):
            block = inside[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            if block.mean() < 0.25:
                continue
            total += 1
            x0, x1, y0, y1 = edges[j], edges[j + 1], edges[i], edges[i + 1]
            if np.any((points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)):
                hit += 1
    return hit / total if total else 0.0


def _local_optimise(a, b, matrix, inl, pixels, rounds=3):
    """LO-RANSAC style: re-fit the similarity by least squares on the inliers, re-count at `pixels`, repeat. The plain RANSAC model comes from a
    2-pair sample and is noisy; the least-squares model of its inliers usually collects more of the right lines. Gate and shift rule kept."""
    best = int(inl.sum())
    for _ in range(rounds):
        if inl.sum() < 3:
            break
        m, _ = cv2.estimateAffinePartial2D(a[inl], b[inl], method=cv2.LMEDS)
        if m is None:
            break
        scale = math.hypot(m[0, 0], m[1, 0])
        angle = math.degrees(math.atan2(m[1, 0], m[0, 0]))
        centre = m @ np.array([111.5, 111.5, 1.0]) - 111.5
        if abs(scale - 1) > 0.15 or abs(angle) > 12 or math.hypot(*centre) > SHIFT_LIMIT_PX:
            break
        new = np.hypot(*(a @ m[:, :2].T + m[:, 2] - b).T) <= pixels
        if new.sum() <= best:
            break
        best, inl = int(new.sum()), new
    return best


def candidate_signals(match, inside, pixels=RANSAC_PIXELS, conf=CONF, with_lo=True):
    """match: dict(points_a, points_b, confidence) as load_rich gives it, or None. inside: the photographed-pixel mask of the canvas."""
    out = dict(EMPTY)
    if match is None or len(match["points_a"]) == 0:
        return out
    a15 = match["points_a"].astype(np.float32)
    b15 = match["points_b"].astype(np.float32)
    c15 = match["confidence"]
    keep = c15 >= conf
    a, b, c = a15[keep], b15[keep], c15[keep]
    size = inside.shape[0]
    out.update(n15=float(len(a15)), n22=float(len(a)))

    # translation-only consensus (2 degrees of freedom)
    k2, _, _ = translation_consensus(a, b, 2.0)
    k3, m3, centre = translation_consensus(a, b, 3.0)
    t_shift = float(np.hypot(*centre))
    if t_shift > SHIFT_LIMIT_PX:
        k2 = k3 = 0
    out.update(inl_t2=float(k2), inl_t3=float(k3), t_shift_px=t_shift, t_dx=float(centre[0]), t_dy=float(centre[1]))
    k3_15, _, centre15 = translation_consensus(a15, b15, 3.0)
    out["inl_t3_15"] = float(k3_15) if np.hypot(*centre15) <= SHIFT_LIMIT_PX else 0.0
    p_t = math.pi * 3.0 ** 2 / DISPLACEMENT_AREA
    out["lnfa_t3"] = log10_nfa(len(a), k3, p_t, len(a))

    # the pipeline's similarity fit (same cut, same RANSAC, same seed)
    if len(a) >= 4:
        cv2.setRNGSeed(SEED)
        matrix, inl = fit_transform(a, b, pixels)
    else:
        matrix, inl = None, np.zeros(len(a), bool)
    if matrix is not None:
        centre_s = matrix @ np.array([(size - 1) / 2, (size - 1) / 2, 1.0]) - (size - 1) / 2
        shift = float(np.hypot(*centre_s))
        if shift <= SHIFT_LIMIT_PX and inl.sum() >= 1:
            ai, bi, ci = a[inl], b[inl], c[inl]
            mapped = ai @ matrix[:, :2].T + matrix[:, 2]
            hc = c >= 0.5
            # guided: every 0.15+ pair within `pixels` of the 0.22 fit
            mapped15 = a15 @ matrix[:, :2].T + matrix[:, 2]
            guided = int((np.hypot(*(mapped15 - b15).T) <= pixels).sum())
            hull = cv2.contourArea(cv2.convexHull(ai)) / float(size ** 2) if len(ai) >= 3 else 0.0
            n = len(a)
            out.update(inl_s3=float(inl.sum()), inl_lo=float(_local_optimise(a, b, matrix, inl, pixels)) if with_lo else float(inl.sum()), guided15=float(guided), shift_px=shift,
                       scale_dev=abs(math.hypot(matrix[0, 0], matrix[1, 0]) - 1.0),
                       angle=abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))),
                       resid=float(np.sqrt(np.mean(np.sum((mapped - bi) ** 2, axis=1)))),
                       hull=float(hull), spread=float(np.sqrt(np.mean(np.var(ai, axis=0)))) / size,
                       cover9=_coverage(ai, inside), conf_inl=float(ci.mean()),
                       hc_ratio=float((inl & hc).sum() / hc.sum()) if hc.any() else 0.0,
                       lnfa_s3=log10_nfa(n, int(inl.sum()), p_t, n * (n - 1) / 2))
    return out


def photo_signal_table(photo, photo_id, data, with_lo=True):
    """One row per stored candidate of every query (rank < M). Columns: identifiers, the pipeline's stored values, the error, every signal."""
    from image_processing.map_levels import level_gsd
    from train.checks.replay import error_sigma
    gsd = level_gsd(photo["level"])
    rows = []
    for qi, q in enumerate(photo["queries"]):
        for r, match in enumerate(q["matches"]):
            x = q["peaks"][r, 1] + q["shifts"][r, 0]
            y = q["peaks"][r, 2] + q["shifts"][r, 1]
            row = dict(data=data, photo=photo_id, query=qi, heading=float(q["heading"]), scale=float(q["scale"]), rank=r,
                       inliers=float(q["inliers"][r]), z=float(q["z"][r]), coarse=float(q["peaks"][r, 0]), sharp=float(q["peaks"][r, 3]),
                       x=float(x), y=float(y), err_sigma=error_sigma(photo, x, y), peak_err_sigma=error_sigma(photo, q["peaks"][r, 1], q["peaks"][r, 2]),
                       height=float(photo["height"]), sigma=float(photo["sigma"]), valid_fraction=float(q["valid_fraction"]))
            row.update(candidate_signals(match, q["inside"], with_lo=with_lo))
            # position from the translation fit (same convention as the pipeline's shift: canvas centre mapped into the tile, times the gsd)
            tx, ty = q["peaks"][r, 1] + row["t_dx"] * gsd, q["peaks"][r, 2] + row["t_dy"] * gsd
            row.update(t_err_sigma=error_sigma(photo, tx, ty) if row["inl_t3"] > 0 else row["peak_err_sigma"], tx=float(tx), ty=float(ty))
            rows.append(row)
    return rows


def _worker_init(paths):
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the workers only re-fit matches; keep them off the GPU (it may be busy with a dump)
    sys.path[:0] = paths


def _table_of_file(args):
    path, data = args
    from train.checks.rich_dump import load_rich
    cv2.setNumThreads(1)
    return photo_signal_table(load_rich(path), int(path.stem.split("_")[1]), data)


def build_signal_table(dump_dir, data, workers=8):
    """Every candidate of every photo_<id>.npz in dump_dir -> one DataFrame (photo order)."""
    import multiprocessing
    import sys
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    import pandas as pd
    files = sorted(Path(dump_dir).glob("photo_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    rows = []
    context = multiprocessing.get_context("spawn")    # fork would copy an initialised CUDA context
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_worker_init, initargs=(list(sys.path),)) as pool:
        for part in pool.map(_table_of_file, [(f, data) for f in files], chunksize=4):
            rows.extend(part)
    return pd.DataFrame(rows)
