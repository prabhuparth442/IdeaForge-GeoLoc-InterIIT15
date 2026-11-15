"""Test_Iteration_8 blocks on made-up data (no GPU, no dataset): the place split, the answer-change split of a gain, the two-tier rule."""
import sys

import numpy as np
import pandas as pd
from common import ROOT, run_tests

sys.path.insert(0, str(ROOT))


def test_split_keeps_places_together_and_balances_heights():
    from train.checks.holdout import official_split
    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 5000, (12, 2))
    rows = []
    for b, (cx, cy) in enumerate(centres):
        for k in range(20):
            rows.append(dict(photo=len(rows), x=cx + rng.normal(0, 20), y=cy + rng.normal(0, 20), h=[110, 150, 200, 300][(b + k) % 4]))
    m = official_split(pd.DataFrame(rows))
    assert m.groupby("block")["half"].nunique().max() == 1                      # a place is never in both halves
    ct = pd.crosstab(m["half"], m["band"])
    assert (ct.loc["try"] - ct.loc["test"]).abs().max() <= 10


def test_answer_change_splits_a_gain():
    from train.checks.holdout import answer_change
    base = pd.DataFrame(dict(data="d", photo=[0, 1, 2, 3], stage=["std6", "skip", "std6", "skip"], pts=[10, 0, -20, 0]))
    new = pd.DataFrame(dict(data="d", photo=[0, 1, 2, 3], stage=["std6", "wide", "skip", "wide"], pts=[10, 10, 0, -20]))
    a = answer_change(base, new).iloc[0]
    assert (a.added, a.added_right, a.added_wrong, a.removed, a.removed_wrong, a.wrong_base, a.wrong_new) == (2, 1, 1, 1, 1, 1, 1)
    assert abs(a.gain - 6 * (0 / 4 - (-10) / 4)) < 1e-9                      # mean points 0 vs -2.5 per photo


def test_two_tier_answers_weak_places_only_when_dino_agrees():
    from train.checks.two_tier import two_tier_p
    t = pd.DataFrame(dict(inliers=[6, 6, 6, 13, 4], z=[2.5, 1.2, 2.5, 0.5, 2.5], zgap=[0.5, 0.5, -0.1, -1.0, 0.5]))
    p = two_tier_p(t, a_low=5, z0=2.0, g0=0.0, i_hi=12)
    assert list(p > 0) == [True, False, False, True, False]
    pe = two_tier_p(t, a_low=5, z0=2.0, g0=0.0, i_hi=12, early=12)
    assert pe[3] == 1.0 and pe[0] == 0.97


run_tests(globals())
