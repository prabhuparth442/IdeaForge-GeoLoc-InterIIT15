"""C2: learn how to fuse the score maps. Trains the three model kinds on several subsets of maps and reports which pays.

Uses candidate sets (candidates.py). Photos are split by FLIGHT between fitting and checking, and the NUDT valid split is
a third, untouched set that picks the winner. All comparisons are paired bootstraps against the plain edge_zncc.
"""
import json
from pathlib import Path

import numpy as np
import torch

from coarse_search.fusion import FusionModel, save_model
from train import config
from train.train_fusion import correct_flags, paired_bootstrap, split_by_group, train


def subsets_of(names):
    """Which map sets to try: the baseline alone, then groups that add one idea at a time, then everything."""
    wanted = [["edge_zncc"],
              ["edge_zncc", "edge_cosine", "edge_mag"],
              ["edge_zncc", "edge_cosine", "edge_mag", "edge_zncc_w"],
              ["edge_zncc", "edge_cosine", "edge_mag", "mind_zncc"],
              ["edge_zncc", "edge_cosine", "edge_mag", "edge_zncc_w", "mind_zncc"],
              ["edge_zncc", "learned_zncc"],
              list(names)]
    out, seen = [], set()
    for s in wanted:
        if all(n in names for n in s) and tuple(s) not in seen:
            out.append(s); seen.add(tuple(s))
    return out


def columns(data, names, subset):
    idx = [names.index(n) for n in subset]
    ctx_cols = []
    for k in idx:
        ctx_cols += [2 * k, 2 * k + 1]
    ctx_cols.append(2 * len(names))                                 # log ground pixel size
    return data["z"][..., idx], data["context"][:, ctx_cols]


def run_ablation(train_data, valid_data, names, out_dir, log=print):
    """Returns a list of result rows and saves every model. Selection uses valid_data (a different set of flights)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = split_by_group(train_data["groups"], 0.2, config.SEED)         # inner split of the train flights (early stopping)
    rows, best = [], None
    for subset in subsets_of(names):
        z, ctx = columns(train_data, names, subset)
        zv, cv = columns(valid_data, names, subset)
        single_flags = None
        for kind in config.FUSION_KINDS:
            if kind == "linear" and len(subset) == 1:
                continue
            model, history = train(kind, z, ctx, tr, va, epochs=config.FUSION_EPOCHS)
            flags = correct_flags(model, zv, cv, np.arange(len(zv)))
            row = dict(maps="+".join(subset), kind=kind, valid_top1=float(flags.mean()), n_valid=int(len(flags)), model_description=model.describe())
            rows.append(row); row["_flags"] = flags
            tag = "%s__%s" % ("+".join(subset), kind)
            save_model(model, out_dir / (tag + ".pt"))
            json.dump(subset, open(out_dir / (tag + ".maps.json"), "w"))
            log("fusion %-70s %-9s valid top-1 %.3f  (%s)" % ("+".join(subset), kind, row["valid_top1"], model.describe()[:60]))
            if best is None or row["valid_top1"] > best["valid_top1"]:
                best = row; best["tag"] = tag; best["subset"] = subset
    base = next(r for r in rows if r["maps"] == "edge_zncc")["_flags"]
    for r in rows:
        d, lo, hi = paired_bootstrap(r["_flags"], base, n_boot=config.BOOTSTRAP)
        r["vs_edge_zncc"] = dict(diff=d, lo=lo, hi=hi)
    for r in rows:
        r.pop("_flags")
    best = {k: v for k, v in best.items() if k != "_flags"}
    return rows, best
