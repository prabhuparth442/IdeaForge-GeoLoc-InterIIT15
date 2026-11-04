"""Candidate sets from pseudo-thermal queries, in the same shard format as candidates.dump_split.

Because the shards look the same, fusion_stage.run_ablation and candidates.load_shards work on them unchanged.
The "flight" of a synthetic example is the 1 km block it was cut from, so a validation split by group keeps whole blocks apart.
"""
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from evaluation.testset import random_prior
from header.scoring import sigma_m
from header.settings import DEVICE
from train import config
from train.datasets.candidates import build_candidate_set
from train.maps.map_bank import compute_maps
from train.datasets.pseudo_thermal import make_synthetic_query

LEVELS = [3, 4, 5, 6, 7]
LEVEL_WEIGHTS = [0.25, 0.25, 0.2, 0.15, 0.15]       # the official set uses all five levels, NUDT only the first two


def make_example(satellite_map, families, names, picker, camera, rng, strength, levels=LEVELS, weights=LEVEL_WEIGHTS):
    """One fake photo at a random place -> candidate set (dict) or None if it could not be made."""
    level = int(rng.choice(levels, p=weights))
    x_m, y_m, block = picker.pick(rng)
    canvas, valid, info = make_synthetic_query(satellite_map, x_m, y_m, level, rng, strength)
    if info["map_fraction"] < 0.9:
        return None
    truth = np.array([x_m, y_m])
    prior = random_prior(truth, rng)
    maps, geometry = compute_maps(canvas, valid, level, prior, families, names)
    record = build_candidate_set(maps, geometry, names, truth, rng)
    if record is None:
        return None
    height = info["footprint_m"] * camera.fx / camera.width
    record["block"] = block
    record["level"] = level
    record["sigma"] = sigma_m(height, camera.fx)
    return record


def make_shard_records(satellite_map, families, names, picker, camera, rng, strength, wanted, log):
    records = []
    tries = 0
    while len(records) < wanted and tries < wanted * 3:
        tries += 1
        try:
            record = make_example(satellite_map, families, names, picker, camera, rng, strength)
        except Exception as error:
            log("synthetic photo skipped: %s: %s" % (type(error).__name__, str(error)[:150]))
            if tries == 1:
                log(traceback.format_exc())
            record = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if record is not None:
            records.append(record)
    return records


def save_shard(path, records, first, names):
    np.savez_compressed(path, z=np.stack([r["z"] for r in records]), pos=np.stack([r["pos"] for r in records]),
                        kind=np.stack([r["kind"] for r in records]), context=np.stack([r["context"] for r in records]),
                        ids=np.arange(first, first + len(records)), flights=np.array([r["block"] for r in records]),
                        levels=np.array([r["level"] for r in records]), sigma=np.array([r["sigma"] for r in records]),
                        names=np.array(names))


def dump_synthetic(satellite_map, families, names, picker, camera, shard_dir, count, strength, deadline, log=print, label="train"):
    """Write shards of fake-photo candidate sets. Finished shards are skipped, so a restart carries on. Stops at the deadline."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    made = 0
    for first in range(0, count, config.SHARD_SIZE):
        path = shard_dir / ("%s_%05d.npz" % (label, first))
        if path.exists():
            made += config.SHARD_SIZE
            continue
        if time.time() > deadline:
            log("time budget used up: stopping the synthetic dump after about %d photos" % made)
            break
        rng = np.random.default_rng([config.SEED, first])
        records = make_shard_records(satellite_map, families, names, picker, camera, rng, strength, min(config.SHARD_SIZE, count - first), log)
        if records:
            save_shard(path, records, first, names)
        made += len(records)
        log("synthetic: %d photos, %.0f s (%.2f s per photo)" % (made, time.time() - start, (time.time() - start) / max(made, 1)))


def truth_wins(z):
    """z (photos, candidates, maps): share of photos where the truth (candidate 0) beats every other candidate, per map."""
    truth = z[:, 0, :]
    best_other = z[:, 1:, :].max(axis=1)
    return (truth > best_other).mean(axis=0)


def pick_strength(satellite_map, families, names, picker, camera, real_rates, log=print, per_strength=30):
    """The fake photos must be about as hard as the real ones: try a few strengths and keep the one whose per-map
    'truth wins' rates are closest to the real training photos."""
    best_strength, best_gap = 1.0, 1e9
    for strength in [0.5, 1.0, 1.5, 2.0]:
        rng = np.random.default_rng(config.SEED + int(strength * 10))
        records = make_shard_records(satellite_map, families, names, picker, camera, rng, strength, per_strength, log)
        z = np.stack([r["z"].astype(np.float32) for r in records])
        rates = truth_wins(z)
        gap = float(np.abs(rates - real_rates).mean())
        log("strength %.1f: truth wins per map %s | gap to real photos %.3f" % (strength, np.round(rates, 2), gap))
        if gap < best_gap:
            best_strength, best_gap = strength, gap
    log("real photos:     truth wins per map %s" % np.round(real_rates, 2))
    log("chosen strength: %.1f (gap %.3f)" % (best_strength, best_gap))
    return best_strength


def save_sample_picture(path, satellite_map, picker, real_set, camera, strength, seed=0, count=6):
    """Row 1: map crop, row 2: its pseudo-thermal canvas, row 3: real NUDT canvases, for a quick look."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from image_processing.geometry import render_query

    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(3, count, figsize=(3 * count, 9))
    for column in range(count):
        level = [3, 3, 4, 4, 5, 7][column % 6]
        x_m, y_m, block = picker.pick(rng)
        canvas, valid, info = make_synthetic_query(satellite_map, x_m, y_m, level, rng, strength)
        axes[0, column].imshow(info["crop"], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title("map crop, level %d" % level, fontsize=8)
        axes[1, column].imshow(canvas, cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title("pseudo-thermal, strength %.1f" % strength, fontsize=8)
        photo = column * max(1, len(real_set.items) // count)
        real_canvas, real_valid = render_query(real_set.items[photo], camera, int(real_set.query_levels[photo]))
        axes[2, column].imshow(real_canvas, cmap="gray", vmin=0, vmax=1)
        axes[2, column].set_title("real NUDT canvas", fontsize=8)
    for axis in axes.ravel():
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(figure)
