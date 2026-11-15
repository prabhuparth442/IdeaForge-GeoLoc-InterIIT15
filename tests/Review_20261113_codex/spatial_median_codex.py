"""Optional position refinement; no change to answer/skip decisions."""
import numpy as np


def refine_position(x, y, sigma, queries):
    positions = []
    for query in queries:
        if len(query["inliers"]) == 0:
            continue
        best = int(np.argmax(query["inliers"]))
        if query["inliers"][best] < 3:
            continue
        position = query["peaks"][best, 1:3] + query["shifts"][best]
        if np.linalg.norm(position - [x, y]) <= sigma:
            positions.append(position)
    if len(positions) < 3:
        return x, y, len(positions)
    median = np.median(positions, axis=0)
    # Componentwise medians need not stay in a circular support region.
    if np.linalg.norm(median - [x, y]) > sigma:
        return x, y, len(positions)
    return float(median[0]), float(median[1]), len(positions)


def refine_pipeline_answer(answer):
    """Opt-in adapter for AgreementQueryPipeline.run output. Returns a copied answer."""
    result = dict(answer)
    result["raw_x"], result["raw_y"] = result["x"], result["y"]
    result["refinement_support"] = 0
    if not result["answered"]:
        return result
    queries = []
    for query in result["queries"]:
        queries.append(dict(peaks=query["peaks"], inliers=query["stage2"]["inliers"], shifts=query["stage2"]["shifts"]))
    result["x"], result["y"], result["refinement_support"] = refine_position(
        result["x"], result["y"], result["sigma_m"], queries)
    return result
