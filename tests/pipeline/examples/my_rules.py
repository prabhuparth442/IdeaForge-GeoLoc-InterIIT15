"""Example replacement: a stricter answer rule (needs 20 inliers instead of 12).
Try:  python tests/pipeline/test_result_thresholding.py --swap result_thresholding.decision.decide=tests/pipeline/examples/my_rules.py:decide
(test_decide_threshold is then expected to FAIL, which shows the swap took effect.)
"""


def decide(inliers):
    assert inliers >= 0
    return bool(inliers >= 20)
