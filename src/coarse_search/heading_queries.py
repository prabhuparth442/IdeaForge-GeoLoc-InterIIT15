"""Block: several query pictures per photo, one for each guess of the heading (and scale).

The photo is drawn north-up on the canvas with the yaw the metadata gives. That yaw is a few degrees off for some photos (Patch v0.5.6: -5 to +10
degrees), and a few degrees are enough to lose the place in the coarse search. So the photo is drawn several times, with the yaw moved by each
heading in the list, and every drawing goes through the pipeline as its own query. The answer is chosen among the candidates of all the queries.

query_list      the queries (heading, scale factor) from two lists
query_geometry  the geometry override that draws one query (settings.GEOMETRY is not changed)
pool_queries    the candidates of the chosen queries as one list, in the format the decision code takes
choose_with_heading_penalty  the same choice as semantic_rerank.decision.choose_candidate, with a cost for every degree the query's heading is away from the metadata
"""
import numpy as np

from header import settings

DEFAULT_HEADINGS = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]       # degrees added to the yaw; step 2.5 is below the 3 degrees the search tolerates


def query_list(headings, scale_factors=(1.0,)):
    """One dict(id, heading, scale_factor) per combination, headings first. Needs the plain query (0 degrees, factor 1) to be in the list."""
    assert len(headings) > 0 and len(scale_factors) > 0, "need at least one heading and one scale factor"
    assert all(factor > 0 for factor in scale_factors), "scale factors must be above 0"
    queries = []
    for factor in scale_factors:
        for heading in headings:
            queries.append(dict(id=len(queries), heading=float(heading), scale_factor=float(factor)))
    plain = [q for q in queries if q["heading"] == 0.0 and q["scale_factor"] == 1.0]
    assert len(plain) == 1, "the list must hold the plain query once (heading 0, scale factor 1), it is the baseline"
    return queries


def query_geometry(heading, scale_factor):
    """Geometry overrides for render_query: the yaw offset moved by `heading` degrees, the scale multiplied by `scale_factor`."""
    base = settings.GEOMETRY
    return dict(yaw_offset=base["yaw_offset"] + heading, scale=base["scale"] * scale_factor)


def pool_queries(results, ids):
    """The candidates of the queries `ids` as one list. results: one dict per query with peaks (M, >= 3), z (M,), stage2 (inliers (M,), shifts (M, 2)).
    Returns peaks, stage2, z, owner (which query each candidate came from), the arrays the decision code takes."""
    assert len(ids) > 0, "need at least one query"
    peaks_parts = []
    z_parts = []
    inlier_parts = []
    shift_parts = []
    owner_parts = []
    for query_id in ids:
        result = results[query_id]
        count = len(result["peaks"])
        assert len(result["z"]) == count and len(result["stage2"]["inliers"]) == count, "one z and one inlier count per candidate"
        peaks_parts.append(np.asarray(result["peaks"], dtype=float))
        z_parts.append(np.asarray(result["z"], dtype=float))
        inlier_parts.append(np.asarray(result["stage2"]["inliers"], dtype=float))
        shift_parts.append(np.asarray(result["stage2"]["shifts"], dtype=float))
        owner_parts.append(np.full(count, query_id, dtype=int))
    peaks = np.concatenate(peaks_parts)
    stage2 = dict(inliers=np.concatenate(inlier_parts), shifts=np.concatenate(shift_parts))
    return peaks, stage2, np.concatenate(z_parts), np.concatenate(owner_parts)


def choose_with_heading_penalty(peaks, stage2, z, owner, headings, gamma, penalty):
    """Like choose_candidate (best  inliers + gamma * z, candidates with 0 inliers are never chosen) but each candidate also pays
    penalty * |heading of its query| (penalty in inliers per degree; 0 gives exactly choose_candidate). headings: the heading of each query, by query id.
    Returns x, y, inliers, index, z of the chosen one. The inliers returned are the real count, so the answer bar is not changed by the penalty."""
    assert penalty >= 0 and gamma >= 0, "penalty and gamma must be >= 0"
    assert len(peaks) == len(z) == len(owner) == len(stage2["inliers"]), "one z, one owner and one inlier count per candidate"
    inliers = stage2["inliers"]
    usable = inliers > 0
    if not usable.any():
        return peaks[0, 1], peaks[0, 2], 0.0, 0, float(z[0])
    score = np.full(len(peaks), -1e9)
    for k in range(len(peaks)):
        if usable[k]:
            score[k] = inliers[k] + gamma * z[k] - penalty * abs(headings[int(owner[k])])
    order = np.lexsort((-peaks[:, 0], -score))
    best = int(order[0])
    x = peaks[best, 1] + stage2["shifts"][best, 0]
    y = peaks[best, 2] + stage2["shifts"][best, 1]
    return x, y, float(inliers[best]), best, float(z[best])
