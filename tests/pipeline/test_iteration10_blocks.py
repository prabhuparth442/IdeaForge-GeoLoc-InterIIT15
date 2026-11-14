"""Test_Iteration_10 evaluation fixes on made-up data (no GPU, no dataset): empty shortlists skip, failed calibration answers nothing,
Jetson context only from the candidates it matched, the per-photo failure log."""
import sys

import numpy as np
import pandas as pd
from common import ROOT, run_tests

sys.path.insert(0, str(ROOT))


def _table(rows_per_query=10, inliers_of=None):
    from train.checks.candidate_judge import STANDARD, add_features
    rows = []
    for photo in (0, 1):
        for h, s in STANDARD:
            for r in range(rows_per_query):
                inl = 0.0 if inliers_of is None else inliers_of(photo, r)
                rows.append(dict(data="d", photo=photo, query=0, heading=h, scale=s, rank=r, inliers=inl, z=1.0, coarse=0.1, x=100.0 * r, y=0.0,
                                 sigma=8.0, err_sigma=0.5 if r == 0 else 50.0, peak_err_sigma=0.5 if r == 0 else 50.0, n22=10, inl_t3=0))
    return add_features(pd.DataFrame(rows))


def test_no_usable_candidate_is_a_skip_not_an_error():
    from train.checks.candidate_judge import decide, baseline
    t = _table()                                                   # every candidate has 0 inliers
    for kw in (dict(), dict(jetson=True, m=5), dict(early_tau=0.9)):
        r = decide(t, np.zeros(len(t)), 0.9, min_choice=3, **kw)
        assert sorted(r["photo"]) == [0, 1] and (r["stage"] == "skip").all() and (r["pts"] == 0).all()
    t2 = t[(t.photo == 0) | (t["rank"] >= 5)]                      # photo 1 keeps only ranks >= 5: an empty shortlist at M = 5
    r = baseline(t2, 8, 8, jetson=True, m=5)
    assert sorted(r["photo"]) == [0, 1] and (r["stage"] == "skip").all()


def test_failed_calibration_answers_nothing():
    from train.checks.candidate_judge import NO_ANSWER, decide, ltt_tau
    scan = pd.DataFrame(dict(tau=[0.9, 0.95, 0.99], data="d", wrong=[5, 5, 5], n_answered=[10, 10, 10], score=0))
    assert ltt_tau(scan, ["d"], 0.005) is None                     # no cut passes: no fallback cut
    t = _table(inliers_of=lambda p, r: 40.0 if r == 0 else 0.0)
    r = decide(t, np.ones(len(t)), NO_ANSWER, override=12, min_choice=3)
    assert (r["stage"] == "skip").all() and len(r) == 2           # not even the 40-inlier override answers


def test_jetson_context_uses_only_matched_candidates():
    from train.checks.candidate_judge import add_context
    t = pd.DataFrame(dict(data="d", photo=0, query=0, rank=[0, 1, 7], x=[0.0, 1.0, 500.0], y=0.0, sigma=8.0, inliers=[10.0, 3.0, 9.0], z=[2.5, 1.0, 2.4]))
    full, m5 = add_context(t), add_context(t, m=5)
    assert abs(full.loc[0, "zgap"] - 0.1) < 1e-9                   # rank 7 is the rival when all 10 are used
    assert m5.loc[0, "zgap"] > 5 and np.isnan(m5.loc[2, "zgap"])    # the Jetson (M = 5) never matched rank 7: no rival, no value


def test_failure_log_outcomes():
    from train.checks.eval_tools import failure_log
    t = _table(inliers_of=lambda p, r: (20.0 if p == 0 else 1.0) if r == 0 else 0.0)
    result = pd.DataFrame(dict(data="d", photo=[0, 1], stage=["std6", "skip"], pts=[10, 0], n_queries=[6, 6]))
    log = failure_log(t, result, "pc").set_index("photo")["outcome"]
    assert log[0] == "answered_right" and log[1] == "retrieved_but_unmatched"
    t3 = t.assign(err_sigma=50.0, peak_err_sigma=50.0)
    assert failure_log(t3, result.assign(stage="skip", pts=0), "pc")["outcome"].eq("never_retrieved").all()


run_tests(globals())
