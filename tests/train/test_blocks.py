"""Tests for the iteration-1 building blocks (train/ package) on the synthetic world, no dataset needed.
Run: python tests/train/test_blocks.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests" / "pipeline"))
import numpy as np
import torch
from common import run_tests, world


def families_for(w, mind=True):
    from train.features.edge import EdgeFamily
    from train.features.mind import MindFamily
    fam = {"edge": EdgeFamily(w["store"])}
    if mind:
        fam["mind"] = MindFamily(w["store"], [3, 4, 5, 6, 7])
    return fam


def test_maps_peak_at_truth():
    from image_processing.geometry import render_query
    from train.maps.map_bank import compute_maps, cell_of
    w = world(); t = w["testset"]; fam = families_for(w)
    names = ["edge_zncc", "edge_cosine", "edge_mag", "edge_zncc_w", "mind_zncc"]
    i = 0; level = int(t.query_levels[i])
    canvas, valid = render_query(t.items[i], t.camera, level)
    prior = t.truths[i] + np.array([300.0, -200.0])
    import coarse_search.search as s
    maps, geo = compute_maps(canvas, valid, level, prior, fam, names)
    r, c = cell_of(geo, *t.truths[i])
    for n in names:
        m = maps[n]; peak = int(torch.argmax(torch.where(geo["inside"], m, torch.full_like(m, -9))))
        pr, pc = peak // m.shape[1], peak % m.shape[1]
        err = np.hypot(float(geo["x_m"][pc]) - t.truths[i][0], float(geo["y_m"][pr]) - t.truths[i][1])
        print("   %-12s peak error %.1f m" % (n, err))
        if n in ("edge_zncc", "edge_cosine"):
            assert err < 20


def test_candidate_set_shapes():
    from image_processing.geometry import render_query
    from train.maps.map_bank import compute_maps
    from train.datasets.candidates import build_candidate_set, n_candidates
    w = world(); t = w["testset"]; fam = families_for(w)
    names = ["edge_zncc", "edge_cosine", "mind_zncc"]
    canvas, valid = render_query(t.items[1], t.camera, int(t.query_levels[1]))
    maps, geo = compute_maps(canvas, valid, int(t.query_levels[1]), t.truths[1] + np.array([100.0, 50.0]), fam, names)
    rec = build_candidate_set(maps, geo, names, t.truths[1], np.random.default_rng(0))
    assert rec["z"].shape == (n_candidates(3), 3) and rec["pos"].shape == (n_candidates(3), 2)
    assert rec["kind"][0] == 0 and (rec["kind"][1:] > 0).all()
    d = np.hypot(*(rec["pos"][1:] - t.truths[1]).T)
    assert d.min() > 40, "a 'wrong' candidate lies on the truth"
    print("   truth z per map:", rec["z"][0], " best wrong z:", rec["z"][1:].max(0))
    assert rec["z"][0, 0] > rec["z"][1:, 0].max(), "truth should be the top edge_zncc candidate on the synthetic world"


def test_whiten_keeps_scale_and_mask():
    from train.features.whitening import estimate_power_profile, whiten
    from header.settings import DEVICE
    x = torch.from_numpy(np.random.rand(8, 300, 300).astype(np.float32)).to(DEVICE)    # whiten works on the compute device
    prof = estimate_power_profile(x.cpu().numpy())
    keep = torch.ones(300, 300, dtype=torch.bool, device=DEVICE); keep[:, :20] = False
    y = whiten(x, prof, 0.5, keep)
    assert abs(float((y[:, keep] ** 2).mean().sqrt() / (x[:, keep] ** 2).mean().sqrt()) - 1) < 1e-3
    assert float(y[:, :, :20].abs().max()) == 0.0
    assert torch.equal(whiten(x, prof, 0.0, keep), x)


def test_zncc_conv_equals_fft_version():
    from coarse_search.correlation import correlate_everywhere
    from train.features.learned_net import zncc_conv
    torch.manual_seed(0)
    window = torch.rand(8, 70, 70); kernel = torch.rand(8, 20, 20); cells = (torch.rand(20, 20) > 0.2).float()
    kernel = kernel * cells                       # the pipeline always zeroes the kernel outside its valid cells
    a = zncc_conv(window, kernel, cells)
    b = correlate_everywhere(window, kernel, cells, min_energy=0, min_variance=0)["zncc"]
    assert torch.allclose(a, b, atol=2e-4), float((a - b).abs().max())


def test_learned_features_train_and_search():
    import tempfile
    from train import config
    from train.features.learned_net import make_pairs, train_features
    from train.features.learned_family import LearnedFamily
    from train.features.edge import EdgeFamily
    from train.maps.map_bank import compute_maps
    from image_processing.geometry import render_query
    w = world(); t = w["testset"]
    pairs = make_pairs(t, w["store"], [0, 1, 2], margin=8)
    assert pairs["crop"].shape[1] == 4 * (56 + 16)
    out = Path(tempfile.mkdtemp()) / "f4.pt"
    hist, best, base = train_features(pairs, pairs, out, minutes=0.4, batch=3)
    assert out.exists() and len(hist) >= 1
    fam = {"edge": EdgeFamily(w["store"]), "learned": LearnedFamily(w["store"], [3, 4, 5, 6, 7], out)}
    canvas, valid = render_query(t.items[0], t.camera, int(t.query_levels[0]))
    maps, geo = compute_maps(canvas, valid, int(t.query_levels[0]), t.truths[0] + np.array([200.0, 100.0]), fam, ["edge_zncc", "learned_zncc"])
    assert maps["learned_zncc"].shape == maps["edge_zncc"].shape
    print("   learned map computed; valid hit rate", best, "edge", base)


run_tests(globals())
