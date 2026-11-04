"""Tests for the DINO blocks (train/features/dino_*.py, train/maps/dino_similarity.py, train/coarse_methods) on the synthetic world.
Needs the DINOv2-S weights in the local cache. Run: python tests/train/test_dino_blocks.py"""
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests" / "pipeline"))
import numpy as np
import torch
from common import run_tests, world

_ENCODER = []


def encoder():
    if not _ENCODER:
        from train.features.dino_tokens import load_dino
        _ENCODER.append(load_dino())
    return _ENCODER[0]


def small_level_image():
    w = world()
    return w["store"]["levels"][5]


def test_token_grid_shape():
    from train.features.dino_tokens import token_grids
    pictures = np.random.rand(2, 224, 224).astype(np.float32)
    grids = token_grids(encoder(), pictures)
    assert tuple(grids.shape) == (2, 384, 16, 16), grids.shape


def test_pca_and_cells_shape():
    from train.features.dino_dense import fit_pca, tokens_to_cells
    samples = torch.randn(500, 384)
    pca = fit_pca(samples, 8)
    assert tuple(pca["components"].shape) == (384, 8) and 0 < pca["explained"] <= 1
    cells = tokens_to_cells(torch.randn(3, 384, 16, 16, device=pca["mean"].device), pca, 56)
    assert tuple(cells.shape) == (3, 8, 56, 56)


def test_map_features_shape_and_zero_outside_the_map():
    from train.features.dino_dense import fit_pca, map_token_features, sample_tokens
    image = small_level_image()
    pca = fit_pca(sample_tokens({5: image}, encoder(), windows_per_level=20), 8)
    features = map_token_features(image, encoder(), pca)
    assert features.shape == (8, image.shape[0] // 4, image.shape[1] // 4), features.shape
    assert np.isfinite(features).all() and features.std() > 0


def test_family_cache_round_trip():
    from train.features.dino_dense import DenseDinoFamily, fit_pca, sample_tokens
    image = small_level_image()
    pca = fit_pca(sample_tokens({5: image}, encoder(), windows_per_level=20), 8)
    with tempfile.TemporaryDirectory() as folder:
        first = DenseDinoFamily({5: image}, [5], encoder(), pca, cache_dir=folder, log=lambda *a: None)
        second = DenseDinoFamily({5: image}, [5], encoder(), pca, cache_dir=folder, log=lambda *a: None)
        assert np.array_equal(first.map_features(5), second.map_features(5))
        cells, cell_valid = first.query(np.random.rand(224, 224).astype(np.float32), np.ones((224, 224), bool))
        assert cells.shape == (8, 56, 56) and cell_valid.shape == (56, 56)


def test_a_window_of_the_map_is_found_exactly_by_dense_dino():
    """Plumbing check: a query identical to one map window must be found at its place with score about 1.
    (A query shifted by a few pixels is NOT found reliably: the known limit written in train/features/dino_dense.py.)"""
    from train.features.dino_dense import DenseDinoFamily, fit_pca, sample_tokens
    from train.maps.map_bank import compute_maps
    from image_processing.map_levels import level_gsd
    image = small_level_image()
    level = 5
    pca = fit_pca(sample_tokens({level: image}, encoder(), windows_per_level=30), 8)
    family = DenseDinoFamily({level: image}, [level], encoder(), pca, log=lambda *a: None)
    top, left = 224, 448                                     # on the window grid of the map
    canvas = (image[top:top + 224, left:left + 224] / 255.0).astype(np.float32)
    valid = np.ones((224, 224), bool)
    gsd = level_gsd(level)
    truth = np.array([(left + 112) * gsd, (top + 112) * gsd])
    maps, geometry = compute_maps(canvas, valid, level, truth + np.array([150.0, -100.0]), {"dino": family}, ["dino_zncc"])
    score = maps["dino_zncc"]
    peak = int(torch.argmax(torch.where(geometry["inside"], score, torch.full_like(score, -9))))
    row, column = peak // score.shape[1], peak % score.shape[1]
    error = float(np.hypot(float(geometry["x_m"][column]) - truth[0], float(geometry["y_m"][row]) - truth[1]))
    print("   dense DINO peak error %.1f m, score %.3f" % (error, float(score[row, column])))
    assert error < 8 and float(score[row, column]) > 0.95


def test_coarse_methods_share_one_contract():
    from train.coarse_methods import build_methods, run_methods
    from train.coarse_methods.inputs import make_inputs
    from train.features.edge import EdgeFamily
    from coarse_search.search import pick_peaks
    from image_processing.geometry import render_query
    w = world(); t = w["testset"]
    level = int(t.query_levels[0])
    canvas, valid = render_query(t.items[0], t.camera, level)
    inputs = make_inputs(canvas, valid, level, t.truths[0] + np.array([200.0, -100.0]), {"edge": EdgeFamily(w["store"])}, ["edge_zncc", "edge_cosine", "edge_mag"])
    methods = build_methods(["edge_zncc", "edge_cosine"], hints=[("edge_zncc", "edge_cosine", 0.5)])
    scores = run_methods(methods, inputs)
    assert list(scores) == ["edge_zncc", "edge_cosine", "edge_zncc + 0.5 * edge_cosine"]
    for label, score in scores.items():
        peaks = pick_peaks(score, inputs["geometry"])
        assert peaks.shape == (25, 5), label


def test_rerank_peaks_orders_by_dino_and_keeps_missing_last():
    import numpy as np
    from train.coarse_methods.dino_rerank import rerank_peaks
    peaks = np.array([[0.9, 0, 0, 0, 0], [0.8, 1, 1, 0, 1], [0.7, 2, 2, 0, 2], [0.0, 1e9, 1e9, 0, -1]])
    dino = np.array([0.1, 0.2, 0.9, 0.0])
    assert rerank_peaks(peaks, dino, 0.0, keep=4)[0, 1] == 0           # weight 0: first stage order
    assert rerank_peaks(peaks, dino, 100.0, keep=4)[0, 1] == 2          # big weight: DINO decides
    assert rerank_peaks(peaks, dino, 100.0, keep=4)[-1, 1] == 1e9       # missing peak stays last


def test_shortlist_gives_50_peaks_and_restores_the_peak_count():
    import torch
    from coarse_search import search
    from header import settings
    from semantic_rerank.dino_rerank import shortlist_peaks
    torch.manual_seed(0)
    size = 400
    score = torch.rand(size, size, device=settings.DEVICE) * 0.5
    geometry = dict(inside=torch.ones(size, size, dtype=torch.bool, device=settings.DEVICE), side=56,
                    x_m=torch.arange(size, device=settings.DEVICE).float() * 16, y_m=torch.arange(size, device=settings.DEVICE).float() * 16)
    before = search.NUM_PEAKS
    assert shortlist_peaks(score, geometry, 50).shape == (50, 5)
    assert search.NUM_PEAKS == before


run_tests(globals())
