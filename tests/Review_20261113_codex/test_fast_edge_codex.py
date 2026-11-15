"""Check cache invalidation and full synthetic agreement-pipeline equivalence."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/pipeline")]

from fast_edge_codex import FastEdgeAgreementPipeline, edge_inputs
from agreement_query_pipeline import AgreementQueryPipeline
from train.features.edge import EdgeFamily
from train.coarse_methods.inputs import make_inputs
from train.coarse_methods.single_map import make_single_map_method
from train.maps import map_bank


class FastEdgeTests(unittest.TestCase):
    def test_cache_changes_when_prior_level_or_map_changes(self):
        rng = np.random.default_rng(14)
        features = rng.random((8, 160, 160)).astype(np.float32)
        family = EdgeFamily(dict(features={6: features, 7: features}))
        canvas = rng.random((224, 224)).astype(np.float32)
        valid = np.ones_like(canvas, bool)
        cache = {}
        method = make_single_map_method("edge_zncc")
        with patch.object(map_bank, "DISC_RADIUS", 300):
            for level, prior in ((7, [500, 500]), (7, [600, 500]), (6, [600, 500]), (7, [-50, 500])):
                original = make_inputs(canvas, valid, level, prior, {"edge": family}, ["edge_zncc"])
                fast = edge_inputs(canvas, valid, level, prior, family, cache)
                self.assertTrue(torch.equal(method(original), method(fast)))
            family.store["features"][7] = rng.random((8, 160, 160)).astype(np.float32)
            original = make_inputs(canvas, valid, 7, [-50, 500], {"edge": family}, ["edge_zncc"])
            fast = edge_inputs(canvas, valid, 7, [-50, 500], family, cache)
            self.assertTrue(torch.equal(method(original), method(fast)))

    def test_complete_synthetic_pipeline_answers_and_resets_per_photo(self):
        from common import world
        from fine_matching import second_stage
        import stand_in_matcher

        synthetic = world()
        original = AgreementQueryPipeline(coarse="edge_zncc", semantic_layer=False, m_fine=5, min_inliers=8)
        original.camera = synthetic["camera"]
        original.families = {"edge": EdgeFamily(synthetic["store"])}
        original.level_images = synthetic["store"]["levels"]
        original.names = ["edge_zncc"]
        original.coarse_name = "edge_zncc"
        original.coarse_method = make_single_map_method("edge_zncc")
        original.matcher = object()
        original.ready = True
        optimized = FastEdgeAgreementPipeline.__new__(FastEdgeAgreementPipeline)
        optimized.__dict__.update(original.__dict__)
        with patch.object(map_bank, "DISC_RADIUS", 600), \
                patch.object(second_stage, "find_keypoints", stand_in_matcher.find_keypoints), \
                patch.object(second_stage, "match_keypoints", stand_in_matcher.match_keypoints):
            for index, item in enumerate(synthetic["items"]):
                prior = synthetic["testset"].truths[index] + [250, -180]
                before = original.run(item, prior)
                after = optimized.run(item, prior)
                for key in ("x", "y", "inliers", "answered", "stage", "n_queries_run"):
                    self.assertEqual(before[key], after[key], key)
                self.assertEqual(optimized._edge_cache_codex, {})
                for old, new in zip(before["queries"], after["queries"]):
                    np.testing.assert_array_equal(old["peaks"], new["peaks"])
                    np.testing.assert_array_equal(old["stage2"]["inliers"], new["stage2"]["inliers"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
