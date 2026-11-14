"""Block: the per-photo numbers a decision gate can look at, from one heading-query run (no plotting, no GPU).

gate_row   one dict per photo: the answer, its error, and gate features (inliers, runner-up inliers, coarse z, peak score and sharpness,
           how many queries agree on the place, the height, the heading and scale of the winner)
"""
import math

import numpy as np

from train.checks.heading_results import candidate_errors


def _query_best_places(found):
    """For each query, the place (x, y after its shift) of its candidate with the most inliers, and that inlier count."""
    places = []
    for query in found["queries"]:
        inliers = np.asarray(query["stage2"]["inliers"], dtype=float)
        best = int(np.argmax(inliers))
        peak = query["peaks"][best]
        shift = query["stage2"]["shifts"][best]
        places.append((peak[1] + shift[0], peak[2] + shift[1], float(inliers[best])))
    return places


def gate_row(found, answer, truth, sigma, height):
    """Features and truth of one photo. found: run_queries output; answer: answer_for output over all queries."""
    chosen = answer["chosen"]
    inliers = np.asarray(answer["stage2"]["inliers"], dtype=float)
    peaks = answer["peaks"]
    shifts = answer["stage2"]["shifts"]
    chosen_x = float(answer["x"])
    chosen_y = float(answer["y"])
    # runner-up: best inlier count among candidates more than 1 sigma away from the chosen place
    runner_up = 0.0
    for k in range(len(inliers)):
        distance = math.hypot(peaks[k, 1] + shifts[k, 0] - chosen_x, peaks[k, 2] + shifts[k, 1] - chosen_y)
        if distance > sigma and inliers[k] > runner_up:
            runner_up = float(inliers[k])
    # agreement: queries whose own best candidate (with at least 3 inliers) lies within 1 sigma of the chosen place
    agree = 0
    for x, y, count in _query_best_places(found):
        if count >= 3 and math.hypot(x - chosen_x, y - chosen_y) <= sigma:
            agree += 1
    # the other candidates (more than 1 sigma from the chosen place): how strong is the field the answer has to stand out from
    others = []
    for k in range(len(inliers)):
        distance = math.hypot(peaks[k, 1] + shifts[k, 0] - chosen_x, peaks[k, 2] + shifts[k, 1] - chosen_y)
        if distance > sigma:
            others.append(float(inliers[k]))
    others = np.asarray(others if others else [0.0])
    error_m = math.hypot(chosen_x - truth[0], chosen_y - truth[1])
    owner = found["queries"][int(answer["owner"][chosen])]
    return dict(inliers=int(answer["inliers"]), runner_up_inliers=runner_up, z_chosen=float(answer["z_chosen"]),
                peak_score=float(peaks[chosen, 0]), peak_sharpness=float(peaks[chosen, 3]), agree_queries=agree, n_queries=len(found["queries"]),
                height=float(height), level=int(found["level"]), winner_heading=float(owner["heading"]), winner_scale=float(owner["scale_factor"]),
                others_median=float(np.median(others)), others_p90=float(np.percentile(others, 90)), others_ge5=int((others >= 5).sum()), others_ge8=int((others >= 8).sum()),
                answered=bool(answer["answered"]), error_m=error_m, error_sigma=error_m / sigma, sigma_m=float(sigma))
