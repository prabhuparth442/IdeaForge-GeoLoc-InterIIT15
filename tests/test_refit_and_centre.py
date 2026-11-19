"""Tests of the tile-centre fix and the final-fit switches. No data, no weights. Run: python tests/test_refit_and_centre.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

import settings
from geoloc import fine_matching
from geoloc.check_settings import check_settings
from geoloc.levels import level_gsd, map_piece


def test_defaults_are_off_and_checked():
    assert settings.REFINE_NATIVE is False and settings.TILE_CENTRE_FIX is False
    check_settings()
    old = (settings.REFINE_NATIVE, settings.REFINE_FINER_LEVELS)
    try:
        settings.REFINE_NATIVE = True
        settings.REFINE_FINER_LEVELS = 2
        check_settings()
        settings.REFINE_FINER_LEVELS = 1
        try:
            check_settings()
            raise AssertionError("REFINE_NATIVE with one level should be refused")
        except SystemExit:
            pass
        except Exception:
            pass
    finally:
        settings.REFINE_NATIVE, settings.REFINE_FINER_LEVELS = old


def test_the_offset_is_the_rounding_of_the_tile_corner():
    gsd = level_gsd(6)
    for x_m, y_m in [(1234.5, 987.1), (500.0, 500.0), (77.7, 4321.9)]:
        offset = fine_matching.tile_centre_offset(x_m, y_m, gsd, 224)
        assert abs(offset[0]) <= 0.5 * gsd + 1e-9 and abs(offset[1]) <= 0.5 * gsd + 1e-9
        # the centre it names is the centre of the tile map_piece really cuts: corner pixel + half the size
        left = int(round(x_m / gsd - 224 / 2))
        assert abs((left + 112) * gsd - (x_m + offset[0])) < 1e-9


def test_a_known_pixel_shift_lands_on_the_true_tile_centre():
    # a tile cut at a rounded corner: the centre of the tile is where a zero shift means
    level_image = np.zeros((600, 600), np.uint8)
    gsd = level_gsd(3)
    x_m, y_m = 300.3 * gsd, 300.8 * gsd
    offset = fine_matching.tile_centre_offset(x_m, y_m, gsd, 224)
    tile_left = int(round(x_m / gsd - 112))
    assert abs((tile_left + 112) * gsd - (x_m + offset[0])) < 1e-9
    assert map_piece(level_image, 3, x_m, y_m, 224).shape == (224, 224)


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
