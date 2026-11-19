"""Tests of the aligned-count rescue (geoloc/aligned_count.py). No data, no weights. Run: python tests/test_aligned_count.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import settings
from geoloc import aligned_count

GSD = 0.85


def matches(count, spread_px, seed):
    rng = np.random.default_rng(seed)
    points_a = rng.uniform(20, 200, (count, 2))
    points_b = points_a + rng.normal(0, spread_px, (count, 2))
    confidence = np.full(count, 0.1)
    return points_a, points_b, confidence


def test_aligned_matches_score_high_and_scattered_ones_low():
    valid = np.ones((224, 224), bool)
    aligned = aligned_count.aligned_z(*matches(60, 1.0, 1), valid, GSD)
    scattered = aligned_count.aligned_z(*matches(60, 60.0, 2), valid, GSD)
    assert aligned > 5 and scattered < 2, (aligned, scattered)


def test_matches_outside_the_photo_are_ignored():
    valid = np.zeros((224, 224), bool)
    assert aligned_count.aligned_z(*matches(60, 1.0, 3), valid, GSD) <= 0.0


def test_pick_place_needs_a_clear_winner():
    peaks = np.array([[1.0, 0.0, 0.0, 0, 0], [1.0, 5000.0, 0.0, 0, 0], [1.0, 10.0, 0.0, 0, 0]])
    shifts = np.zeros((3, 2))
    sigma = 5.0
    clear = np.array([settings.ALIGNED_MIN_Z + 20, 2.0, -np.inf])
    assert aligned_count.pick_place(peaks, shifts, clear, sigma) == (0.0, 0.0)
    rival = np.array([settings.ALIGNED_MIN_Z + 20, settings.ALIGNED_MIN_Z + 18, -np.inf])
    assert aligned_count.pick_place(peaks, shifts, rival, sigma) is None
    weak = np.array([settings.ALIGNED_MIN_Z - 1, 0.0, -np.inf])
    assert aligned_count.pick_place(peaks, shifts, weak, sigma) is None


if __name__ == "__main__":
    test_aligned_matches_score_high_and_scattered_ones_low()
    test_matches_outside_the_photo_are_ignored()
    test_pick_place_needs_a_clear_winner()
    print("aligned count: 3 tests passed")
