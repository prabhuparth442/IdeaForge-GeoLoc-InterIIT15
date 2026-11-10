"""Block: answer-or-skip from query agreement (no GPU, no plotting).

Several queries of one photo (other headings or scales) each find their own best place. When two of them land on the same place, a false match
is unlikely: a false place collects inliers by chance and two queries rarely make the same chance hit. Measured on 379 official and 500 NUDT
photos (tests/Test_Iteration_3/Decision_Report.md): answer at >= 8 inliers OR >= 2 of the 6 standard queries agreeing gives +3.9 official,
+1.0 / +0.3 NUDT, 0 wrong answers.

Agreement only counts when the queries see clearly different pictures. Queries 2.5 degrees apart (the 27-query escalation) land on the same
false place together, so the escalated decision uses a higher inlier bar instead (see ESCALATION_MIN_INLIERS).

query_best_places   each query's own best candidate: place and inliers
query_agreement     number of queries whose best candidate lies within radius of a place
decide_with_agreement   answer when inliers >= bar or agreement >= min_agree
"""
import math

import numpy as np

AGREE_MIN_QUERIES = 2           # queries that must agree (the chosen query counts itself)
AGREE_RADIUS_SIGMA = 1.0        # agreement radius in scoring sigma
AGREE_QUERY_INLIERS = 3         # a query's best candidate needs this many inliers to count
ESCALATION_MIN_INLIERS = 11     # bar for the 27-query decision: false places reached 10 inliers there, never 11 (12 scores almost the same)


def query_best_places(results, ids):
    """For each query in `ids`: (x, y, inliers) of its candidate with the most inliers (x, y after the fine-stage shift)."""
    assert len(ids) > 0, "need at least one query"
    places = []
    for query_id in ids:
        result = results[query_id]
        inliers = np.asarray(result["stage2"]["inliers"], dtype=float)
        assert len(inliers) == len(result["peaks"]), "one inlier count per candidate"
        best = int(np.argmax(inliers))
        x = float(result["peaks"][best][1] + result["stage2"]["shifts"][best][0])
        y = float(result["peaks"][best][2] + result["stage2"]["shifts"][best][1])
        places.append((x, y, float(inliers[best])))
    return places


def query_agreement(results, ids, x, y, radius_m, min_query_inliers=AGREE_QUERY_INLIERS):
    """Number of queries in `ids` whose own best candidate has >= min_query_inliers inliers and lies within radius_m of (x, y)."""
    assert radius_m > 0, "radius must be above 0 metres, got %s" % radius_m
    count = 0
    for qx, qy, inliers in query_best_places(results, ids):
        if inliers >= min_query_inliers and math.hypot(qx - x, qy - y) <= radius_m:
            count += 1
    return count


def decide_with_agreement(inliers, agreement, min_inliers, min_agree=AGREE_MIN_QUERIES):
    """Answer when the chosen place has >= min_inliers inliers, or when >= min_agree queries agree on it."""
    assert inliers >= 0 and agreement >= 0, "inliers and agreement cannot be negative"
    assert min_agree >= 2, "one query always agrees with itself; min_agree must be >= 2"
    return bool(inliers >= min_inliers or agreement >= min_agree)
