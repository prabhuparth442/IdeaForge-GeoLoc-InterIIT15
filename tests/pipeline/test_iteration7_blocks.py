"""Test_Iteration_7 blocks on made-up data (no GPU, no dataset): post-matching signals, the candidate judge's decision replay, the second look's fit."""
import math
import sys

import numpy as np
import pandas as pd
from common import ROOT, run_tests

sys.path.insert(0, str(ROOT))


def _similarity(a, angle_deg, scale, shift):
    t = math.radians(angle_deg)
    r = scale * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (a @ r.T + shift).astype(np.float32)


def test_signals_of_a_clean_shift_and_of_noise():
    from train.checks.decision_signals import candidate_signals
    rng = np.random.default_rng(0)
    a = rng.uniform(20, 200, (80, 2)).astype(np.float32)
    inside = np.ones((224, 224), bool)
    clean = candidate_signals(dict(points_a=a, points_b=a + np.float32([4, -2]), confidence=np.full(80, 0.8)), inside)
    assert clean["inl_s3"] == 80 and clean["inl_t3"] == 80 and clean["cover9"] > 0.5 and clean["lnfa_t3"] < -50
    assert abs(clean["t_dx"] - 4) < 1e-3 and abs(clean["t_dy"] + 2) < 1e-3 and clean["resid"] < 1e-3
    noise = candidate_signals(dict(points_a=a, points_b=rng.uniform(0, 224, (80, 2)).astype(np.float32), confidence=np.full(80, 0.8)), inside)
    assert noise["inl_t3"] < 6 and noise["lnfa_t3"] > clean["lnfa_t3"]
    assert candidate_signals(None, inside)["inl_s3"] == 0


def test_translation_consensus_ignores_a_shift_beyond_the_cell():
    from train.checks.decision_signals import candidate_signals
    a = np.random.default_rng(1).uniform(20, 200, (40, 2)).astype(np.float32)
    far = candidate_signals(dict(points_a=a, points_b=a + np.float32([30, 0]), confidence=np.full(40, 0.9)), np.ones((224, 224), bool))
    assert far["inl_t3"] == 0 and far["inl_s3"] == 0          # 30 px is outside the 12 px coarse cell: the pipeline drops such a fit too


def test_guided_lines_count_faint_pairs_that_agree_with_the_fit():
    from train.checks.decision_signals import candidate_signals
    a = np.random.default_rng(2).uniform(20, 200, (60, 2)).astype(np.float32)
    conf = np.r_[np.full(30, 0.8), np.full(30, 0.18)]          # half the pairs under the 0.22 cut
    s = candidate_signals(dict(points_a=a, points_b=a + 2, confidence=conf), np.ones((224, 224), bool))
    assert s["n22"] == 30 and s["inl_s3"] == 30 and s["guided15"] == 60


def test_nfa_falls_with_more_inliers_and_rises_with_more_pairs():
    from train.checks.decision_signals import log10_nfa
    p = 0.001
    assert log10_nfa(50, 10, p, 50) < log10_nfa(50, 5, p, 50) < log10_nfa(500, 5, p, 500)
    assert log10_nfa(10, 0, p, 10) == 0.0


def _table():
    """Two photos, standard and one extra query each. Photo 1: right place at 6 inliers (judge P 0.97). Photo 2: wrong place at 9 inliers (P 0.1)."""
    from train.checks.candidate_judge import add_features
    rows = []
    for photo, (inl, err, p) in enumerate([(6, 0.5, 0.97), (9, 300.0, 0.1)]):
        for h, s in [(-5.0, 1.0), (0.0, 1.0), (5.0, 1.0), (-5.0, 1.05), (0.0, 1.05), (5.0, 1.05), (-10.0, 1.0)]:
            best = (h, s) == (0.0, 1.0)
            rows.append(dict(data="d", photo=photo, query=0, heading=h, scale=s, rank=0, inliers=inl if best else 0, z=2.0, coarse=0.1,
                             err_sigma=err if best else 500.0, n22=20, inl_t3=inl if best else 0, p=p if best else 0.0))
    return add_features(pd.DataFrame(rows))


def test_judge_decision_answers_on_p_and_baseline_on_the_bar():
    from train.checks.candidate_judge import baseline, decide, summary
    t = _table()
    judged = decide(t, t["p"].to_numpy(), 0.95, min_choice=3)
    assert list(judged["stage"]) == ["std6", "skip"] and list(judged["pts"]) == [10, 0]
    bars = baseline(t, 8, 11)
    assert list(bars["stage"]) == ["skip", "std6"] and list(bars["pts"]) == [0, -20]       # the bar skips the right 6 and answers the wrong 9
    over = decide(t, t["p"].to_numpy(), 0.95, override=9, min_choice=3)
    assert list(over["pts"]) == [10, -20]                                                    # the override answers on inliers alone
    assert summary(judged).loc["d", "wrong"] == 0


def test_extra_query_cut_and_early_exit():
    from train.checks.candidate_judge import decide
    t = _table()
    t.loc[(t.photo == 0) & (t.heading == 0.0) & (t.scale == 1.0), ["inliers", "p"]] = 0.0             # photo 0: right place only on the extra query
    t.loc[(t.photo == 0) & (t.heading == -10.0), ["inliers", "p", "err_sigma"]] = [6.0, 0.93, 0.5]
    p = t["p"].to_numpy()
    assert decide(t, p, 0.9, min_choice=3, jetson=True)["stage"].iloc[0] == "wide"
    assert decide(t, p, 0.9, min_choice=3, jetson=True, wide_tau=0.95)["stage"].iloc[0] == "skip"        # stricter cut after the standard queries
    t2 = _table()
    early = decide(t2, t2["p"].to_numpy(), 0.9, min_choice=3, early_tau=0.95)
    assert early["stage"].iloc[0] == "early" and early["n_queries"].iloc[0] == 2                           # (0, 1.0) is the 2nd standard query


def test_jetson_replay_caps_the_queries():
    from train.checks.candidate_judge import baseline
    t = _table()
    jet = baseline(t, 20, 20, jetson=True, cap=8)
    assert list(jet["n_queries"]) == [8, 8] and list(jet["stage"]) == ["skip", "skip"]


def test_smallest_safe_tau_needs_zero_wrong_at_every_larger_cut():
    from train.checks.candidate_judge import smallest_safe_tau
    scan = pd.DataFrame(dict(tau=[0.5, 0.6, 0.7, 0.8, 0.9], data="d", wrong=[2, 0, 1, 0, 0], score=0))
    assert smallest_safe_tau(scan, ["d"], 0) == 0.8 and smallest_safe_tau(scan, ["d"], 1) == 0.9


def test_query_local_uniqueness():
    from train.checks.candidate_judge import add_context
    t = pd.DataFrame(dict(data="d", photo=0, query=0, x=[0.0, 1.0, 500.0], y=0.0, sigma=8.0, inliers=[10.0, 3.0, 4.0], z=[2.5, 1.0, 1.5]))
    c = add_context(t)
    assert abs(c.loc[0, "zgap"] - 1.0) < 1e-9 and abs(c.loc[0, "ligap"] - (math.log1p(10) - math.log1p(4))) < 1e-9   # the near one is not a rival


def test_signed_fit_reads_angle_scale_and_shift():
    from train.checks.second_look import signed_fit
    a = np.random.default_rng(3).uniform(20, 200, (60, 2)).astype(np.float32)
    centre = np.array([111.5, 111.5])
    b = _similarity(a - centre, 4.0, 1.06, centre + np.array([5.0, -3.0]))
    n, angle, scale, (dx, dy) = signed_fit(dict(points_a=a, points_b=b, confidence=np.full(60, 0.9)))
    assert n == 60 and abs(angle - 4.0) < 0.05 and abs(scale - 1.06) < 1e-3 and abs(dx - 5) < 0.05 and abs(dy + 3) < 0.05


run_tests(globals())
