"""Test_Iteration_6 blocks on made-up data (no GPU, no dataset): rich dump packing, offline decision replay, query policies, verifier features."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from common import ROOT, run_tests

sys.path.insert(0, str(ROOT))


def _photo(best, sigma=8.0, truth=(0.0, 0.0)):
    """A photo in load_rich's format with all 27 wide queries. best: {(heading, scale): (x, inliers)}; other queries find nothing."""
    from train.checks.replay import WIDE
    queries = []
    for h, s in WIDE:
        x, inl = best.get((h, s), (5000.0, 0.0))
        peaks = np.zeros((10, 5))
        peaks[:, 1] = 5000.0 + 100 * np.arange(10)
        peaks[0, 1] = x
        peaks[:, 0] = np.linspace(0.5, 0.1, 10)
        inliers = np.zeros(10)
        inliers[0] = inl
        queries.append(dict(peaks=peaks, z=np.zeros(10), inliers=inliers, shifts=np.zeros((10, 2)), heading=h, scale=s, valid_fraction=0.8,
                            seconds=dict(render=0.1, fine=0.4), inside=np.ones((8, 8), bool), margin=0, border_px=0, matches=[None] * 10))
    return dict(truth=np.array(truth), sigma=sigma, height=300.0, level=6, yaw=0.0, pitch=0.0, roll=0.0, queries=queries)


def test_pack_and_load_keep_matches_to_one_64th_pixel():
    from train.checks.rich_dump import load_rich, pack_photo
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 447, (40, 2)).astype(np.float32)
    b = a + 3.3
    conf = rng.uniform(0.1, 1.0, 40)
    stored = dict(inside=rng.uniform(size=(448, 448)) > 0.3, margin=0, border_px=0,
                  items=[dict(points_a=a, points_b=b, confidence=conf), None])
    query = dict(peaks=np.ones((2, 5)), z=np.zeros(2), stage2=dict(inliers=np.array([7.0, 0.0]), shifts=np.zeros((2, 2))), heading=-2.5,
                 scale_factor=1.05, valid_fraction=0.7, seconds=dict(fine=0.3, render=0.01), matches=stored)
    item = dict(rel_height=210.0, yaw=12.0, pitch=-1.0, roll=0.5)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "photo_0.npz"
        np.savez_compressed(path, **pack_photo(item, (1.0, 2.0), 8.3, 5, [query]))
        photo = load_rich(path)
    q = photo["queries"][0]
    kept = conf >= 0.15
    assert q["matches"][1] is None and len(q["matches"][0]["points_a"]) == kept.sum()
    assert np.abs(q["matches"][0]["points_a"] - a[kept]).max() <= 1 / 128 + 1e-6
    assert np.abs(q["matches"][0]["confidence"] - conf[kept]).max() < 1e-4
    assert (q["inside"] == stored["inside"]).all() and q["heading"] == -2.5 and q["seconds"]["fine"] == 0.3 and photo["roll"] == 0.5


def test_replay_answers_at_the_bar_by_agreement_by_escalation_or_skips():
    from train.checks.replay import decide_current
    bar = decide_current(_photo({(0.0, 1.0): (3.0, 9.0)}))
    assert bar["stage"] == "bar" and bar["points"] == 10 and bar["n_queries"] == 6
    agree = decide_current(_photo({(0.0, 1.0): (3.0, 5.0), (5.0, 1.05): (-4.0, 3.0)}))
    assert agree["stage"] == "agreement" and agree["agree"] == 2
    lone = decide_current(_photo({(0.0, 1.0): (3.0, 5.0), (10.0, 0.95): (2.0, 10.0)}))
    assert lone["stage"] == "skip" and lone["n_queries"] == 27            # 10 inliers in escalation is below the bar 11
    esc = decide_current(_photo({(0.0, 1.0): (3.0, 5.0), (10.0, 0.95): (2.0, 12.0)}))
    assert esc["stage"] == "escalation" and esc["points"] == 10
    wrong = decide_current(_photo({(-10.0, 1.0): (900.0, 15.0)}))
    assert wrong["stage"] == "escalation" and wrong["points"] == -20


def test_policy_early_exit_and_query_cap():
    from train.checks.query_policy import run_policy
    photo = _photo({(-5.0, 1.0): (1.0, 30.0), (10.0, 0.95): (2.0, 12.0)})
    assert run_policy(photo, early_bar=20)["n_queries"] == 1
    assert run_policy(photo)["n_queries"] == 6                           # no early exit: all standard queries, then the bar
    late = _photo({(10.0, 0.95): (2.0, 12.0)})
    full = run_policy(late)
    assert full["stage"] == "escalation" and full["answered"]
    capped = run_policy(late, max_queries=8)
    assert not capped["answered"] and capped["n_queries"] == 8


def test_z_veto_skips_a_low_z_answer_unless_it_has_many_inliers():
    from train.checks.query_policy import run_policy
    photo = _photo({(0.0, 1.0): (900.0, 9.0)})               # a wrong place at 9 inliers (above bar 8) ...
    for q in photo["queries"]:
        q["z"][:] = 0.5                                       # ... with a low DINO z
    assert run_policy(photo)["points"] == -20
    assert not run_policy(photo, z_veto=(1.3, 15))["answered"]
    strong = _photo({(0.0, 1.0): (2.0, 20.0)})
    for q in strong["queries"]:
        q["z"][:] = 0.5
    assert run_policy(strong, z_veto=(1.3, 15))["points"] == 10     # 20 inliers pass the veto whatever the z


def test_fit_features_of_a_clean_similarity_and_of_noise():
    from train.checks.verifier_features import candidate_fit_features
    rng = np.random.default_rng(3)
    a = rng.uniform(20, 430, (60, 2)).astype(np.float32)
    clean = candidate_fit_features(dict(points_a=a, points_b=a + np.float32([5, -3]), confidence=np.full(60, 0.9)), 448)
    assert clean["n_inl"] == 60 and clean["resid"] < 1e-3 and clean["fit_scale_dev"] < 1e-3 and clean["fit_angle"] < 1e-2 and clean["hull"] > 0.5
    noise = candidate_fit_features(dict(points_a=a, points_b=rng.uniform(0, 448, (60, 2)).astype(np.float32), confidence=np.full(60, 0.9)), 448)
    assert noise["n_inl"] < 15
    assert candidate_fit_features(None, 448)["n_pairs"] == 0


def test_paired_bootstrap_of_a_constant_gain():
    from train.checks.replay import paired_bootstrap
    mean, low, high = paired_bootstrap(np.zeros(50), np.full(50, 10.0))
    assert abs(mean - 60.0) < 1e-9 and abs(low - 60.0) < 1e-9 and abs(high - 60.0) < 1e-9     # all 0 -> all +10 is a 0 -> 60 score


run_tests(globals())
