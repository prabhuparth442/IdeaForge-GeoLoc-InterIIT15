"""Simulated candidate sets with a KNOWN fusion rule, to test that training recovers it.

Two regimes (the context says which): in regime 0 map A is informative and map B is noise; in regime 1 the reverse.
Linear fusion must compromise; a gated model can learn to follow the context.
"""
import numpy as np
import torch


def simulate(n_photos=600, n_cand=200, signal=3.0, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.randn(n_photos, n_cand, 2).astype(np.float32)
    regime = rng.randint(0, 2, n_photos)
    for p in range(n_photos):
        z[p, 0, regime[p]] += signal                    # the truth (index 0) stands out in the informative map only
    context = np.stack([regime, rng.randn(n_photos)], 1).astype(np.float32)
    groups = ["f%d" % (p // 20) for p in range(n_photos)]
    return torch.from_numpy(z), torch.from_numpy(context), groups
