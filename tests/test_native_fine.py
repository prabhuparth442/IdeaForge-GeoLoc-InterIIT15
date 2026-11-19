"""Tests of the native-resolution LoFTR path (geoloc/native_fine.py and the size / tile_filter / limit_level arguments). No data, no weights. Run: python tests/test_native_fine.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import settings
from geoloc import native_fine, photo
from geoloc.levels import level_gsd

CAMERA = dict(fx=200.0, fy=200.0, cx=159.5, cy=127.5, k1=0, k2=0, k3=0, p1=0, p2=0, width=320, height=256)
POSE = dict(height=1400.0, yaw=30.0, pitch=0.0, roll=0.0)


def test_off_by_default():
    assert settings.NATIVE_FINE_FROM_HEIGHT_M is None
    assert native_fine.fine_level(1500.0, 11, {9: 1, 11: 1}) is None


def test_fine_level_follows_gate_and_store():
    old = settings.NATIVE_FINE_FROM_HEIGHT_M
    try:
        settings.NATIVE_FINE_FROM_HEIGHT_M = 800.0
        assert native_fine.fine_level(700.0, 10, {8: 1, 10: 1}) is None            # below the gate
        assert native_fine.fine_level(1000.0, 10, {8: 1, 10: 1}) == 8
        assert native_fine.fine_level(1000.0, 10, {9: 1, 10: 1}) is None           # the store has no level 8
        assert native_fine.fine_level(1000.0, 4, {2: 1, 4: 1}) is None             # below the lowest level of the run
    finally:
        settings.NATIVE_FINE_FROM_HEIGHT_M = old


def test_canvas_size_doubles_for_two_levels():
    assert native_fine.canvas_size(2) == 2 * settings.TILE
    assert abs(level_gsd(10) / level_gsd(8) - 2.0) < 1e-9


def test_draw_canvas_size_argument():
    picture = np.random.default_rng(0).random((256, 320)).astype(np.float32)
    canvas, valid = photo.draw_canvas(picture, POSE, CAMERA, 10, 0.0, 1.0)
    assert canvas.shape == (settings.TILE, settings.TILE)
    big, big_valid = photo.draw_canvas(picture, POSE, CAMERA, 8, 0.0, 1.0, 2 * settings.TILE)
    assert big.shape == (2 * settings.TILE, 2 * settings.TILE) and big_valid.shape == big.shape
    # the same footprint: about the same share of the canvas is covered by the photo
    assert abs(valid.mean() - big_valid.mean()) < 0.03, (valid.mean(), big_valid.mean())


def test_clean_tile_keeps_the_shape_range_and_the_no_map_area():
    rng = np.random.default_rng(1)
    tile = (rng.random((448, 448)) * 0.6 + 0.2).astype(np.float32)
    tile[:, :100] = 0.0
    out = native_fine.clean_tile(tile, 8.0)
    assert out.shape == tile.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(float(out[:, :100].std())) < 1e-6          # the no-map area stays flat


def test_tile_sigma_scales_with_the_pixels():
    assert abs(native_fine.tile_sigma(7.0, 6.8) - 15.0 * 7.0 / 6.8) < 1e-9
    assert native_fine.tile_sigma(1.0, 13.6) == 3.0        # never below 3 pixels


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
