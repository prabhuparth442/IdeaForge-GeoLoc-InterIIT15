import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from decision_ideas_codex import choose, decide_photo, heldout_geometry


class DecisionTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame(dict(inliers=[10, 8], z=[2., 1.], coarse=[0.7, 0.6], p=[0.999, 0.95],
                                 x=[0., 100.], y=[0., 0.], sigma=[8., 8.], query=[0, 1], rank=[0, 0],
                                 heading=[-5., 0.], scale=[1., 1.]))

    def test_strong_but_conflicting_locations_abstain(self):
        self.assertIsNone(choose(self.table(), "unique_location", 0.98))

    def test_nearby_queries_are_one_location_not_competing_places(self):
        table = self.table()
        table.loc[1, "x"] = 2
        self.assertEqual(choose(table, "unique_location", 0.98)["query"], 0)

    def test_large_inlier_count_cannot_override_judge(self):
        table = self.table()
        table["inliers"] = 1000
        table["p"] = 0.5
        self.assertIsNone(choose(table, "strict_judge", 0.98))

    def matches(self):
        x, y = np.meshgrid(np.arange(20, 220, 20), np.arange(20, 220, 20))
        first = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        return dict(points_a=first, points_b=first + [4., -2.], confidence=np.full(len(first), 0.8))

    def test_correct_transform_predicts_withheld_correspondences(self):
        self.assertTrue(heldout_geometry(self.matches(), np.ones((224, 224), bool)))

    def test_different_transforms_in_each_half_are_rejected(self):
        matches = self.matches()
        split = (np.floor(matches["points_a"] / 56).sum(axis=1) % 2) == 0
        matches["points_b"][split] += [0., 10.]
        self.assertFalse(heldout_geometry(matches, np.ones((224, 224), bool)))

    def test_insufficient_or_unphotographed_evidence_is_rejected(self):
        self.assertFalse(heldout_geometry(None, np.ones((224, 224), bool)))
        self.assertFalse(heldout_geometry(self.matches(), np.zeros((224, 224), bool)))

    def test_selection_does_not_read_ground_truth_columns(self):
        table = self.table()
        first = decide_photo(table, {}, "strict_judge")
        table["err_sigma"] = [100000., 0.]
        table["pts"] = [-20, 10]
        second = decide_photo(table, {}, "strict_judge")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
