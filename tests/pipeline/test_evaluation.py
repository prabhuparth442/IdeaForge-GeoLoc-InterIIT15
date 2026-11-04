import numpy as np
from common import run_tests


def test_auc_basic_and_ties():
    from evaluation.metrics import auc
    assert auc([1, 2, 3, 4], [False, True, False, True]) == 0.75
    assert auc([0, 0, 0, 5], [False, False, True, True]) == 0.75
    assert np.isnan(auc([1, 2], [True, True]))


def test_auc_matches_pair_counting():
    from evaluation.metrics import auc
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.integers(0, 4, 40); r = rng.random(40) < .4
        pairs = np.mean([(a > b) + 0.5 * (a == b) for a in v[r] for b in v[~r]])
        assert abs(pairs - auc(v, r)) < 1e-9


def test_prior_is_uniform_in_disc():
    from evaluation.testset import random_prior
    rng = np.random.default_rng(0)
    d = np.array([np.hypot(*random_prior(np.zeros(2), rng)) for _ in range(20000)])
    assert abs(np.median(d) - 5000 * 0.5 ** 0.5) < 60 and d.max() <= 5000


def test_honest_gate_removes_noise_gain():
    from evaluation.metrics import best_gate, honest_gate
    rng = np.random.default_rng(3)
    points = rng.choice([10, 6, 3, 0, -20], 200, p=[.25, .1, .1, .2, .35])
    signal = rng.random(200)
    tuned = best_gate(signal, points)[0]
    honest = honest_gate(signal, points)[0]
    assert honest < tuned


def test_gated_curve_shapes():
    from evaluation.metrics import gated_curve
    rng = np.random.default_rng(1)
    shares, scores = gated_curve(rng.random(50), rng.choice([10, -20], 50))
    assert len(shares) == len(scores) == 40 and shares[0] == 1.0


run_tests(globals())
