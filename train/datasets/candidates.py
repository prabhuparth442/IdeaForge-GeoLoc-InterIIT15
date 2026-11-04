"""Turn full score maps into one small training example per photo ("candidate set").

A candidate set holds n positions: index 0 = the true position, then the strongest WRONG peaks of every map and of the
plain average of all maps (hard negatives), then random wrong positions in the disc. For every candidate we keep the
robust z-score of every map. The 6.8 M-position maps are thrown away; ~100 numbers per candidate are kept.
The truth is placed on the best cell within +-2 cells of the nearest grid cell, so a half-cell grid offset does not count against it.
"""
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from header.settings import CELL, DEVICE
from image_processing.map_levels import level_gsd
from train import config
from train.maps.map_bank import cell_of, compute_maps, nms_peaks, z_stack


def n_candidates(n_maps):
    return 1 + n_maps * config.N_HARD_PER_MAP + config.N_HARD_FUSED + config.N_RANDOM


def build_candidate_set(maps, geometry, names, truth_m, rng):
    """Returns a dict (z, pos, kind, context, rank_info) or None if the truth is outside the usable area."""
    z, usable, context = z_stack(maps, geometry, names)
    found = cell_of(geometry, truth_m[0], truth_m[1])
    if found is None:
        return None
    row, col = found
    r0, r1 = max(row - 2, 0), min(row + 3, z.shape[0])
    c0, c1 = max(col - 2, 0), min(col + 3, z.shape[1])
    local = z[r0:r1, c0:c1].mean(-1)
    local = torch.where(usable[r0:r1, c0:c1], local, torch.full_like(local, -1e9))
    best = int(local.reshape(-1).argmax())
    if float(local.reshape(-1)[best]) < -1e8:
        return None
    row, col = r0 + best // (c1 - c0), c0 + best % (c1 - c0)

    gsd = level_gsd(geometry["level"])
    cell_m = CELL * gsd
    gap = max(224 // CELL, geometry["side"] // 2)
    exclude_cells = config.EXCLUDE_M / cell_m
    chosen = [(row, col)]
    kinds = [0]

    def far_enough(r, c):
        if np.hypot(r - row, c - col) <= exclude_cells:
            return False
        return all(np.hypot(r - a, c - b) > 3 for a, b in chosen)

    mean_z = z.mean(-1)
    sources = [(z[..., k], config.N_HARD_PER_MAP) for k in range(z.shape[-1])] + [(mean_z, config.N_HARD_FUSED)]
    for score, want in sources:
        got = 0
        for r, c, value in nms_peaks(score, usable, gap, want + 3):
            if got >= want:
                break
            if far_enough(r, c):
                chosen.append((r, c)); kinds.append(1); got += 1
    total = n_candidates(len(names))
    tries = 0
    usable_cpu_rows, usable_cpu_cols = z.shape[0], z.shape[1]
    while len(chosen) < total and tries < 20000:
        tries += 1
        r, c = int(rng.integers(usable_cpu_rows)), int(rng.integers(usable_cpu_cols))
        if bool(usable[r, c]) and far_enough(r, c):
            chosen.append((r, c)); kinds.append(2)
    if len(chosen) < total:
        return None
    rows = torch.tensor([r for r, c in chosen], device=z.device)
    cols = torch.tensor([c for r, c in chosen], device=z.device)
    values = z[rows, cols].float().cpu().numpy()
    pos = np.stack([geometry["x_m"][cols].cpu().numpy(), geometry["y_m"][rows].cpu().numpy()], 1)
    return dict(z=values.astype(np.float16), pos=pos.astype(np.float32), kind=np.array(kinds, np.int8), context=context)


def dump_split(testset, store, families, names, ids, shard_dir, label, deadline=None, log=print, extra_maps=None, extra_names=()):
    """Write shards of candidate sets for the photos `ids`. Finished shards are skipped, so the run can be restarted.

    Maps that compute_maps cannot make (for example the pooled DINO similarity) come in through `extra_maps(maps, geometry, canvas, valid, level, prior_m)`,
    which adds them to the dict of maps; their names go in `extra_names` and must also be in `names`."""
    from image_processing.geometry import render_query
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED + zlib.crc32(label.encode()) % 1000)
    ids = [int(i) for i in ids]
    start = time.time()
    done_photos = 0
    for first in range(0, len(ids), config.SHARD_SIZE):
        chunk = ids[first:first + config.SHARD_SIZE]
        path = shard_dir / ("%s_%05d.npz" % (label, first))
        if path.exists():
            done_photos += len(chunk)
            continue
        if deadline is not None and time.time() > deadline:
            log("time budget used up: stopping the %s dump after %d photos" % (label, done_photos))
            break
        records, kept = [], []
        for i in chunk:
            level = int(testset.query_levels[i])
            try:
                canvas, valid = render_query(testset.items[i], testset.camera, level)
                maps, geometry = compute_maps(canvas, valid, level, testset.priors[i], families, [n for n in names if n not in extra_names])
                if extra_maps is not None:
                    extra_maps(maps, geometry, canvas, valid, level, testset.priors[i])
                record = build_candidate_set(maps, geometry, names, testset.truths[i], rng)
            except Exception as error:
                log("photo %d skipped: %s: %s" % (i, type(error).__name__, str(error)[:150]))
                record = None
            finally:
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            if record is not None:
                records.append(record); kept.append(i)
        if records:
            np.savez_compressed(path, z=np.stack([r["z"] for r in records]), pos=np.stack([r["pos"] for r in records]),
                                kind=np.stack([r["kind"] for r in records]), context=np.stack([r["context"] for r in records]),
                                ids=np.array(kept), flights=np.array([testset.items[i]["flight"] for i in kept]),
                                levels=np.array([int(testset.query_levels[i]) for i in kept]),
                                sigma=np.array([testset.sigmas[i] for i in kept]), names=np.array(names))
        done_photos += len(chunk)
        log("%s: %d / %d photos, %d kept in this shard, %.0f s" % (label, done_photos, len(ids), len(records), time.time() - start))


def load_shards(shard_dir, label, names):
    """All shards of one split as tensors: z (P, n, K), context (P, C), groups (list of flight names), plus extras."""
    files = sorted(Path(shard_dir).glob("%s_*.npz" % label))
    assert files, "no shards named %s_*.npz in %s" % (label, shard_dir)
    parts = [np.load(f, allow_pickle=True) for f in files]
    for p in parts:
        assert list(p["names"]) == list(names), "shard %s was made with maps %s, not %s" % (label, list(p["names"]), list(names))
    cat = lambda key: np.concatenate([p[key] for p in parts])
    return dict(z=torch.from_numpy(cat("z").astype(np.float32)), context=torch.from_numpy(cat("context")),
                groups=[str(g) for g in cat("flights")], kind=torch.from_numpy(cat("kind")), pos=cat("pos"),
                ids=cat("ids"), levels=cat("levels"), sigma=cat("sigma"))
