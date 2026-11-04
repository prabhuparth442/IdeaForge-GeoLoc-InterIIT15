import numpy as np
from common import run_tests, world


def test_levels_and_heights():
    from image_processing.map_levels import level_for_height, level_gsd
    class Camera: fx = 200.0; width = 320
    assert [round(level_gsd(l), 2) for l in range(3, 8)] == [0.85, 1.2, 1.7, 2.4, 3.39]
    assert [level_for_height(h, Camera()) for h in (100, 150, 200, 250, 300, 350)] == [3, 4, 5, 6, 6, 7]


def test_normalise_range_and_gain_invariance():
    from image_processing.thermal import normalise_frame
    x = np.random.default_rng(0).random((64, 80)) * 1000
    a, b = normalise_frame(x), normalise_frame(3 * x + 500)
    assert a.shape == x.shape and 0 <= a.min() and a.max() <= 1
    assert np.abs(a - b).max() < 0.02          # percentile stretch removes gain and offset


def test_vertical_edge_goes_to_bin_0():
    from image_processing.edge_features import edge_features, gradient_parts
    x = np.arange(32)[None, :].repeat(32, 0).astype(np.float32)
    edge = (x >= 16).astype(np.float32)
    assert gradient_parts(edge)[5][10, 15] == 0 and edge_features(edge).shape == (8, 8, 8)


def test_canvas_and_features_shapes():
    from image_processing.edge_features import query_edge_features
    from image_processing.geometry import render_query
    w = world()
    canvas, valid = render_query(w["items"][0], w["camera"], int(w["testset"].query_levels[0]))
    assert canvas.shape == (224, 224) and valid.dtype == bool and valid.mean() > 0.3
    features, cells = query_edge_features(canvas, valid)
    assert features.shape == (8, 56, 56) and cells.shape == (56, 56)


def test_map_store_has_all_levels():
    w = world()
    assert sorted(w["store"]["levels"]) == [3, 4, 5, 6, 7] and sorted(w["store"]["features"]) == [3, 4, 5, 6, 7]


def test_map_piece_centre():
    from image_processing.map_levels import map_piece
    w = world()
    piece = map_piece(w["store"]["levels"], 3, 900.0, 900.0, 224)
    assert piece.shape == (224, 224) and 0 <= piece.min() and piece.max() <= 1


run_tests(globals())
