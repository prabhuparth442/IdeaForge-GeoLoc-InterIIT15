"""Regression tests for evaluation integrity and isolated policies."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evaluation_fixes_codex import available_candidate_features, collect_answers, decide_certified, select_certified_tau
from policies_codex import early_twelve, eligible_rescue, headless_policy
from train.checks.candidate_judge import add_context, add_features
from train.checks.rich_dump import load_rich


class ReviewTests(unittest.TestCase):
    def test_failed_calibration_abstains_instead_of_returning_a_threshold(self):
        scan = pd.DataFrame(dict(data="d", tau=[0.9, 0.99], wrong=[10, 10], n_answered=[100, 100]))
        self.assertIsNone(select_certified_tau(scan, ["d"]))

    def test_calibration_requires_enough_independent_answers(self):
        scan = pd.DataFrame(dict(data="d", tau=[0.9, 0.99], wrong=[0, 0], n_answered=[100, 100]))
        self.assertIsNone(select_certified_tau(scan, ["d"]))
        scan["n_answered"] = 1000
        self.assertEqual(select_certified_tau(scan, ["d"]), 0.9)

    def test_calibration_missing_set_is_not_silently_accepted(self):
        scan = pd.DataFrame(dict(data=["d"], tau=[0.9], wrong=[0], n_answered=[1000]))
        with self.assertRaises(ValueError):
            select_certified_tau(scan, ["missing"])

    def test_failed_certificate_disables_override_and_early_exit(self):
        table = pd.DataFrame(dict(data=["d", "d"], photo=[1, 2], inliers=[100, 100]))
        result = decide_certified(table, np.ones(2), None, override=12, early_tau=0.98)
        self.assertEqual(result.stage.tolist(), ["skip", "skip"])
        self.assertEqual(result.pts.tolist(), [0, 0])

    def test_fixed_sequence_stops_at_first_failure(self):
        scan = pd.DataFrame(dict(data="d", tau=[0.8, 0.9, 0.99], wrong=[0, 10, 0], n_answered=[1000] * 3))
        self.assertEqual(select_certified_tau(scan, ["d"]), 0.99)

    def test_uncomputed_candidates_cannot_change_context(self):
        source = ROOT / "tests/Test_Iteration_7/results/signals_ours_nudt707.csv.gz"
        table = pd.read_csv(source, nrows=270)
        expected = available_candidate_features(table, 5)
        table.loc[table["rank"] >= 5, ["z", "inliers"]] = 1000
        actual = available_candidate_features(table, 5)
        pd.testing.assert_frame_equal(expected, actual)
        leaked = add_context(add_features(table))
        leaked = leaked[leaked["rank"] < 5].reset_index(drop=True)
        self.assertFalse(np.allclose(actual.zgap, leaked.zgap))

    def make_testset(self):
        testset = SimpleNamespace(sample_ids=np.array([0, 1]), camera=SimpleNamespace(fx=200),
                                  items=[dict(rel_height=100), dict(rel_height=100)])
        testset.error_m = lambda index, x, y: float(np.hypot(x, y))
        return testset

    def test_empty_search_keeps_photo_in_denominator_as_skip(self):
        missing = np.array([[0, 1e9, 1e9, 0, 0], [0, 1e9, 1e9, 0, 0]], float)
        valid = np.array([[1, 0, 0, 0, 0], [0.5, 10, 0, 0, 0]], float)
        disc = {0: dict(peaks={("zncc", 0.0): missing}), 1: dict(peaks={("zncc", 0.0): valid})}
        result, closest, ratios, counts = collect_answers(self.make_testset(), disc, {}, {}, False)
        np.testing.assert_array_equal(result["correlation"]["points"], [0, 10])
        self.assertTrue(np.isinf(closest[0]))
        self.assertEqual(ratios[0], 0)
        self.assertEqual(len(counts), 0)

    def test_all_empty_searches_are_supported(self):
        missing = np.array([[0, 1e9, 1e9, 0, 0]], float)
        disc = {index: dict(peaks={("zncc", 0.0): missing}) for index in (0, 1)}
        result, closest, ratios, counts = collect_answers(self.make_testset(), disc, {}, {}, True)
        for method in result.values():
            np.testing.assert_array_equal(method["points"], [0, 0])
        self.assertTrue(np.isinf(closest).all())
        self.assertEqual(len(counts), 2)

    def test_policy_choices_do_not_use_truth(self):
        paths = sorted((ROOT / "tests/Test_Iteration_7/results/rich_dump/nudt_fresh707").glob("photo_*.npz"))
        photo = load_rich(paths[0], with_matches=False)
        other = dict(photo, truth=np.array([1e7, 1e7]))
        for policy in (eligible_rescue, early_twelve, headless_policy):
            first, second = policy(photo), policy(other)
            for key in ("x", "y", "stage", "n_queries", "answered"):
                self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
