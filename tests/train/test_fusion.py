"""Fusion network tests. Run: python tests/train/test_fusion.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np
import torch
from coarse_search.fusion import FusionModel, robust_z, save_model, load_model
from train.simulate import simulate
from train.train_fusion import split_by_group, train, correct_flags, paired_bootstrap


def test_robust_z():
    s = torch.randn(50, 50) * 3 + 7
    z = robust_z(s, torch.ones(50, 50, dtype=torch.bool))
    assert abs(float(z.median())) < 1e-4 and abs(float((z - z.median()).abs().median()) * 1.4826 - 1) < 0.05


def test_monotone():
    for kind in ("linear", "monotone", "gated"):
        m = FusionModel(kind, 2, 2)
        for p in m.parameters():
            p.data += torch.randn_like(p) * 0.5
        base = torch.randn(100, 2)
        ctx = torch.randn(2)
        up = base.clone(); up[:, 0] += 1.0
        assert (m(up, ctx) >= m(base, ctx) - 1e-5).all(), kind + " not monotone in map 0"


def test_split_is_flight_disjoint():
    _, _, groups = simulate(100)
    tr, va = split_by_group(groups, 0.25, 0)
    assert not set(np.array(groups)[tr]) & set(np.array(groups)[va])


def test_gated_beats_linear():
    z, ctx, groups = simulate(800, seed=1)
    tr, va = split_by_group(groups, 0.3, 0)
    res = {}
    for kind in ("linear", "gated"):
        m, hist = train(kind, z, ctx, tr, va, epochs=250)
        assert hist[-1][0] < hist[0][0], kind + ": training loss did not decrease"
        res[kind] = correct_flags(m, z, ctx, va)
    d, lo, hi = paired_bootstrap(res["gated"], res["linear"])
    print("gated - linear accuracy %.3f (95%% %.3f..%.3f)" % (d, lo, hi))
    assert d > 0 and lo > 0, "gated should clearly beat linear on gated data"


def test_save_load(tmp="/tmp/_fusion_test.pt"):
    m = FusionModel("gated", 2, 2)
    z = torch.randn(5, 2); c = torch.randn(2)
    save_model(m, tmp)
    assert torch.allclose(m(z, c), load_model(tmp)(z, c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
