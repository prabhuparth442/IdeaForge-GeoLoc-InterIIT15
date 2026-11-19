"""Tests of the photo cleaning (geoloc/flat_field.py). No data, no weights. Run: python tests/test_flat_field.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import settings
from geoloc import flat_field

CAMERA = dict(cx=159.0, cy=127.0)


def vignetted_frame(seed=1):
    rng = np.random.default_rng(seed)
    texture = rng.normal(0, 200.0, (256, 320))
    rows, cols = np.mgrid[0:256, 0:320].astype(float)
    radius2 = ((cols - 159.0) ** 2 + (rows - 127.0) ** 2) / (159.0 ** 2 + 127.0 ** 2)
    vignette = 6000.0 * (1 - radius2) ** 2
    return 20000.0 + vignette + texture, texture, radius2


def test_the_vignette_is_removed_and_the_texture_stays():
    frame, texture, radius2 = vignetted_frame()
    out = flat_field.remove_vignette(frame, CAMERA["cx"], CAMERA["cy"], 3)
    corr_before = abs(np.corrcoef(frame.ravel(), radius2.ravel())[0, 1])
    corr_after = abs(np.corrcoef(out.ravel(), radius2.ravel())[0, 1])
    assert corr_before > 0.8 and corr_after < 0.05, (corr_before, corr_after)
    assert abs(out.std() - texture.std()) < 0.15 * texture.std()
    assert abs(out.mean() - frame.mean()) < 1.0


def test_a_bright_object_does_not_pull_the_fit():
    frame, texture, radius2 = vignetted_frame()
    frame[100:140, 200:240] += 15000.0
    out = flat_field.remove_vignette(frame, CAMERA["cx"], CAMERA["cy"], 3)
    assert out[100:140, 200:240].mean() - out[0:40, 0:40].mean() > 12000.0


def test_mask_fills_only_the_rectangle_with_the_median():
    frame, _, _ = vignetted_frame()
    out = flat_field.mask_rects(frame, [(300, 220, 312, 250)])
    assert np.allclose(out[220:250, 300:312], np.median(frame), atol=1e-3)
    outside = np.ones(frame.shape, bool); outside[220:250, 300:312] = False
    assert np.allclose(out[outside], frame[outside].astype(np.float32), atol=1e-3)


def test_local_normalise_equalises_a_weak_and_a_strong_region():
    rng = np.random.default_rng(2)
    frame = rng.normal(0, 1.0, (256, 320)); frame[:, 160:] *= 10.0
    out = flat_field.local_normalise(frame, 10.0)
    weak, strong = out[:, 20:120].std(), out[:, 200:300].std()
    assert 0.5 < weak / strong < 2.0, (weak, strong)


def test_everything_off_returns_the_frame_untouched():
    frame, _, _ = vignetted_frame()
    saved = (settings.FLAT_FIELD_FROM_HEIGHT_M, settings.LOCAL_NORM_SIGMA, settings.PHOTO_MASK_RECTS)
    try:
        settings.FLAT_FIELD_FROM_HEIGHT_M = None; settings.LOCAL_NORM_SIGMA = None; settings.PHOTO_MASK_RECTS = None
        assert flat_field.clean(frame, dict(height=1500.0), CAMERA) is frame
    finally:
        settings.FLAT_FIELD_FROM_HEIGHT_M, settings.LOCAL_NORM_SIGMA, settings.PHOTO_MASK_RECTS = saved


def test_the_height_gate_decides_who_is_cleaned():
    frame, _, _ = vignetted_frame()
    saved = settings.FLAT_FIELD_FROM_HEIGHT_M
    try:
        settings.FLAT_FIELD_FROM_HEIGHT_M = 800.0
        assert flat_field.clean(frame, dict(height=799.0), CAMERA) is frame
        assert flat_field.clean(frame, dict(height=800.0), CAMERA) is not frame
    finally:
        settings.FLAT_FIELD_FROM_HEIGHT_M = saved


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function(); print("ok", name)
