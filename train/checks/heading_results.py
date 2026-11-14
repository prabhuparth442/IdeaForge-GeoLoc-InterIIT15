"""Block: turn the queries of one photo into table rows for the heading-query test (no plotting, no GPU).

query_rows      one row per query: did it find the place, with how many inliers, how long it took
variant_rows    one row per variant (a chosen subset of the queries): the answer the pipeline would give from that subset, its points, and the stage flags
                (in_K: a query's K places hold the target, in_M: the kept M places do, chosen_right: the chosen place is right, all before the fine shift)
summarise       a table of variant rows -> score, answered, wrong answers, recall, wrong candidates that reach the bar
"""
import math

import numpy as np

from header.scoring import final_score, points_for_error


def variant_ids(queries, headings, scale_factors):
    """The ids of the queries whose heading and scale factor are in the two lists. Every combination has to exist."""
    ids = []
    for query in queries:
        if query["heading"] in headings and query["scale_factor"] in scale_factors:
            ids.append(query["id"])
    assert len(ids) == len(headings) * len(scale_factors), "headings %s x scale factors %s are not all among the queries of the pipeline" % (headings, scale_factors)
    return ids


def candidate_errors(peaks, shifts, truth, sigma):
    """Distance in sigma of each candidate (after its shift) from the truth."""
    assert sigma > 0, "sigma must be positive"
    errors = np.full(len(peaks), 1e9)
    for k in range(len(peaks)):
        if peaks[k, 1] < 1e8:
            errors[k] = math.hypot(peaks[k, 1] + shifts[k, 0] - truth[0], peaks[k, 2] + shifts[k, 1] - truth[1]) / sigma
    return errors


def peak_errors(peaks, truth, sigma):
    """Distance in sigma of each place the search proposed (before the fine stage moves it) from the truth."""
    return candidate_errors(peaks, np.zeros((len(peaks), 2)), truth, sigma)


def query_rows(found, truth, sigma, found_sigma, bar):
    """One row per query of one photo. right = a candidate within found_sigma of the truth; wrong = any other."""
    rows = []
    for query in found["queries"]:
        errors = candidate_errors(query["peaks"], query["stage2"]["shifts"], truth, sigma)
        inliers = np.asarray(query["stage2"]["inliers"], dtype=float)
        right = errors <= found_sigma
        wrong = ~right
        row = dict(query=query["id"], heading=query["heading"], scale_factor=query["scale_factor"], valid_fraction=query["valid_fraction"])
        row["target_found"] = bool(right.any())
        row["right_inliers"] = float(inliers[right].max()) if right.any() else 0.0
        row["wrong_inliers"] = float(inliers[wrong].max()) if wrong.any() else 0.0
        row["right_reaches_bar"] = bool(row["right_inliers"] >= bar)
        row["seconds"] = sum(query["seconds"].values())
        rows.append(row)
    return rows


def variant_rows(pipeline, found, variants, truth, sigma, height, focal_px, found_sigma, bar, penalties=None):
    """One row per variant for one photo. variants: dict name -> list of query ids. penalties: dict name -> inliers per degree of heading (missing = 0)."""
    penalties = penalties or {}
    rows = []
    for name, ids in variants.items():
        answer = pipeline.answer_for(found["queries"], ids, penalties.get(name, 0.0))
        errors = candidate_errors(answer["peaks"], answer["stage2"]["shifts"], truth, sigma)
        inliers = answer["stage2"]["inliers"]
        error_m = math.hypot(answer["x"] - truth[0], answer["y"] - truth[1])
        earned = points_for_error(error_m, height, focal_px)
        wrong_at_bar = bool(((errors > found_sigma) & (inliers >= bar)).any())
        in_k = False
        for query_id in ids:
            if (peak_errors(found["queries"][query_id]["shortlist"], truth, sigma) <= found_sigma).any():
                in_k = True
        place_errors = peak_errors(answer["peaks"], truth, sigma)
        seconds = 0.0
        for query_id in ids:
            seconds += sum(found["queries"][query_id]["seconds"].values())
        row = dict(variant=name, n_queries=len(ids), x=answer["x"], y=answer["y"], answered=answer["answered"], inliers=answer["inliers"],
                   err_m=error_m, err_sigma=error_m / sigma, would_earn=earned, points=earned if answer["answered"] else 0,
                   target_found=bool((errors <= found_sigma).any()), wrong_candidate_at_bar=wrong_at_bar,
                   in_K=in_k, in_M=bool((place_errors <= found_sigma).any()), chosen_right=bool(place_errors[answer["chosen"]] <= found_sigma),
                   winning_heading=float(found["queries"][int(answer["owner"][answer["chosen"]])]["heading"]),
                   winning_scale_factor=float(found["queries"][int(answer["owner"][answer["chosen"]])]["scale_factor"]), seconds=seconds)
        rows.append(row)
    return rows


def summarise(table, found_sigma):
    """Score and the shares of one variant over a table of variant rows (one row per photo)."""
    assert len(table) > 0, "need at least one photo"
    answered_wrong = table["answered"] & (table["err_sigma"] > found_sigma)
    result = {}
    result["photos"] = len(table)
    result["score (/60)"] = final_score(list(table["points"]))
    result["answered"] = float(table["answered"].mean())
    result["answered right"] = float((table["answered"] & (table["err_sigma"] <= found_sigma)).mean())
    result["wrong answers"] = int(answered_wrong.sum())
    result["target in candidates"] = float(table["target_found"].mean())
    result["wrong candidate reaches bar"] = float(table["wrong_candidate_at_bar"].mean())
    result["seconds per photo"] = float(table["seconds"].mean())
    return result
