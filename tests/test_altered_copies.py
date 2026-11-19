"""Tests of the altered copies (geoloc/altered_copies.py). No data, no weights. Run: python tests/test_altered_copies.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from geoloc import altered_copies


def frame_16bit():
    rng = np.random.default_rng(3)
    return (rng.random((64, 80)) * 40000 + 5000).astype(np.uint16)


def test_invert_flips_the_order_and_keeps_the_type():
    frame = frame_16bit()
    flipped = altered_copies.make_copy(frame, "invert")
    assert flipped.dtype == frame.dtype and flipped.shape == frame.shape
    assert flipped.min() == frame.min() and flipped.max() == frame.max()
    assert np.array_equal(altered_copies.make_copy(flipped, "invert"), frame)
    assert np.corrcoef(frame.ravel().astype(float), flipped.ravel().astype(float))[0, 1] < -0.999


def test_sharpen_raises_the_edge_contrast_and_keeps_the_type():
    frame = np.zeros((64, 64), np.uint16)
    frame[:, 32:] = 20000
    sharper = altered_copies.make_copy(frame, "sharpen")
    assert sharper.dtype == frame.dtype
    assert sharper[:, 33].astype(int).max() > 20000        # overshoot next to the step
    assert sharper[:, 0].astype(int).max() == 0 or sharper[:, 0].astype(int).max() < 100   # flat parts stay flat


def test_unknown_kind_is_refused():
    try:
        altered_copies.make_copy(frame_16bit(), "mirror")
    except ValueError:
        return
    raise AssertionError("an unknown kind must raise ValueError")


if __name__ == "__main__":
    test_invert_flips_the_order_and_keeps_the_type()
    test_sharpen_raises_the_edge_contrast_and_keeps_the_type()
    test_unknown_kind_is_refused()
    print("altered copies: 3 tests passed")
