"""Downstream check: how well does the existing coarse search find the truth from this canvas? Same function for real and simulated queries."""
import numpy as np
import torch

from train.maps.map_bank import cell_of, compute_maps

EXCLUDE_M = 45.0            # cells closer than this to the truth are not "wrong" cells


def coarse_truth_metrics(canvas, valid, level, truth_xy, prior_xy, families, map_name="edge_zncc"):
    """Score map of one canvas over the search disc around the prior, then:

    peak_error_m   distance of the best cell from the truth
    z_truth        score at the truth in standard deviations of the disc's scores
    truth_percentile  share of disc cells that score higher than the truth cell (0 = the truth is the best cell)
    margin         score at the truth minus the best wrong cell (> 0 = found)
    """
    maps, geometry = compute_maps(canvas, valid, level, prior_xy, families, [map_name])
    score = maps[map_name]
    inside = geometry["inside"]
    found = cell_of(geometry, *truth_xy)
    assert found is not None, "the truth lies outside the search window"
    row, col = found
    truth_score = float(score[row, col])

    x_grid = geometry["x_m"][None, :].expand_as(score)
    y_grid = geometry["y_m"][:, None].expand_as(score)
    distance = torch.hypot(x_grid - float(truth_xy[0]), y_grid - float(truth_xy[1]))
    wrong = inside & (distance > EXCLUDE_M)
    masked = torch.where(inside, score, torch.full_like(score, -9.0))
    peak = int(torch.argmax(masked))
    peak_row, peak_col = peak // score.shape[1], peak % score.shape[1]
    peak_error = float(torch.hypot(geometry["x_m"][peak_col] - float(truth_xy[0]), geometry["y_m"][peak_row] - float(truth_xy[1])))

    values = score[inside]
    best_wrong = float(score[wrong].max())
    return dict(peak_error_m=peak_error, z_truth=(truth_score - float(values.mean())) / (float(values.std()) + 1e-9),
                truth_percentile=float((values > truth_score).float().mean()), margin=truth_score - best_wrong)
