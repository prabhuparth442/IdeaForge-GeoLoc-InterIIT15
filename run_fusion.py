"""Main run file for the fusion network.

  python run_fusion.py demo                      train + compare on simulated data with a known gating rule (no dataset needed)
  python run_fusion.py train --data cands.npz    train on dumped candidate sets; prints linear vs monotone vs gated
                                                 and saves the best model to --out

cands.npz holds: z (photos, candidates, maps) float32 with candidate 0 = the true position,
                 context (photos, C) float32, groups (photos,) flight names.
Producing cands.npz from real photos needs the NUDT reader (see train/README.md); everything after it is here.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import torch
from coarse_search.fusion import save_model
from train.simulate import simulate
from train.train_fusion import split_by_group, train, correct_flags, paired_bootstrap


def compare(z, context, groups, out=None, epochs=300):
    tr, va = split_by_group(groups, 0.25, 0)
    print("%d train photos, %d valid photos (flight-disjoint), %d candidates each" % (len(tr), len(va), z.shape[1]))
    models, flags = {}, {}
    for kind in ("linear", "monotone", "gated"):
        m, _ = train(kind, z, context, tr, va, epochs=epochs)
        models[kind], flags[kind] = m, correct_flags(m, z, context, va)
        print("%-9s valid top-1 %.3f   %s" % (kind, flags[kind].mean(), m.describe()))
    for kind in ("monotone", "gated"):
        d, lo, hi = paired_bootstrap(flags[kind], flags["linear"])
        print("%s - linear: %+.3f  (95%% interval %+.3f .. %+.3f)%s" % (kind, d, lo, hi, "" if lo > 0 else "   not clearly better"))
    best = max(flags, key=lambda k: flags[k].mean())
    if out:
        save_model(models[best], out); print("saved", best, "->", out)
    return models


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "train"])
    ap.add_argument("--data"); ap.add_argument("--out", default="fusion_model.pt"); ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    if a.mode == "demo":
        z, c, g = simulate(800, seed=1)
        compare(z, c, g, epochs=a.epochs)
    else:
        assert a.data, "--data cands.npz is required"
        d = np.load(a.data, allow_pickle=True)
        compare(torch.from_numpy(d["z"]), torch.from_numpy(d["context"]), list(d["groups"]), a.out, a.epochs)
