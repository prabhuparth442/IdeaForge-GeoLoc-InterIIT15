"""Tests of the Step 1 blocks on made-up data: no GPU, no dataset, no torch.
Run from this folder:   python tests/test_blocks.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thresholding_blocks.environment_check import missing_files, package_versions
from thresholding_blocks.photo_selection import check_matches_loaded_photos, choose_photos
from thresholding_blocks.probe_blocks import classify_cause, fit_with_reasons, offset_label, offset_positions
from thresholding_blocks.probe_tables import cause_counts, compare_with_pipeline, offset_curve, per_photo_table, truth_shift_table

CANVAS = 224
GSD = 0.6
LIMIT = 12 * GSD


def make_pairs(n_good, n_bad, shift_px=(0.0, 0.0), scale=1.0, angle_deg=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.random((n_good + n_bad, 2)) * 200 + 10).astype(np.float32)
    angle = math.radians(angle_deg)
    rotation = np.array([[scale * math.cos(angle), -scale * math.sin(angle)], [scale * math.sin(angle), scale * math.cos(angle)]])
    b = (a @ rotation.T + np.array(shift_px)).astype(np.float32)
    b[n_good:] = (rng.random((n_bad, 2)) * 200 + 10).astype(np.float32)
    return a, b


def test_fit_ok():
    a, b = make_pairs(30, 10, shift_px=(3.0, -2.0))
    r = fit_with_reasons(a, b, 3.0, CANVAS, GSD, LIMIT)
    assert r["reason"] == "ok" and r["inliers"] == r["raw_inliers"] and r["raw_inliers"] >= 30, r
    assert abs(r["scale"] - 1) < 0.01 and abs(r["angle_deg"]) < 0.5


def test_fit_shift_gate_keeps_raw_count():
    a, b = make_pairs(30, 5, shift_px=(40.0, 0.0))
    r = fit_with_reasons(a, b, 3.0, CANVAS, GSD, LIMIT)
    assert r["reason"] == "shift_gate" and r["inliers"] == 0 and r["raw_inliers"] >= 30, r
    assert r["shift_east_m"] > LIMIT


def test_fit_scale_angle_gate():
    a, b = make_pairs(30, 5, angle_deg=20.0)
    r = fit_with_reasons(a, b, 3.0, CANVAS, GSD, LIMIT)
    assert r["reason"] == "scale_angle_gate" and r["inliers"] == 0 and r["raw_inliers"] >= 25, r


def test_fit_too_few_pairs():
    a, b = make_pairs(3, 0)
    r = fit_with_reasons(a, b, 3.0, CANVAS, GSD, LIMIT)
    assert r["reason"] == "too_few_pairs" and r["raw_inliers"] == 0 and len(r["inlier_mask"]) == 3


def test_fit_random_pairs_give_few_inliers():
    a, b = make_pairs(0, 40, seed=3)
    r = fit_with_reasons(a, b, 3.0, CANVAS, GSD, LIMIT)
    assert r["raw_inliers"] < 12, r["raw_inliers"]


def test_fit_same_count_as_the_src_call():
    import cv2
    a, b = make_pairs(25, 10, shift_px=(2.0, 1.0), seed=5)
    cv2.setRNGSeed(1)
    matrix, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=3000, confidence=0.995)
    cv2.setRNGSeed(1)
    r = fit_with_reasons(a, b, 3.0, CANVAS, GSD, LIMIT)
    assert r["raw_inliers"] == int(mask.sum())


def test_classify():
    assert classify_cause(30, 20, "shift_gate", 5) == "gate_removed_a_good_fit"
    assert classify_cause(30, 20, "ok", 5) == "candidate_is_fine"
    assert classify_cause(2, 1, "too_few_pairs", 5) == "weak_even_at_truth"
    assert classify_cause(30, 1, "ok", 40) == "candidate_too_far_from_truth"
    assert classify_cause(30, 1, "ok", 3) == "weak_at_candidate_near_truth"


def test_offsets():
    assert offset_label(0, 0) == "truth" and offset_label(8, -16) == "truth+8,-16"
    positions = offset_positions(100.0, 200.0, 0.5, [(0, 0), (8, 0), (0, -16)])
    assert positions[0] == ("truth", 100.0, 200.0)
    assert positions[1][1] == 104.0 and positions[2][2] == 192.0
    try:
        offset_positions(0, 0, 1.0, [(8, 0)])
        assert False, "must insist that (0, 0) comes first"
    except AssertionError as error:
        assert "(0, 0)" in str(error)


def make_probes():
    rows = []
    def add(photo, group, label, raw, inl, reason, offset=np.nan, east=0.0, south=0.0):
        rows.append(dict(photo=photo, group=group, label=label, offset_px=offset, raw_inliers=raw, inliers=inl, reason=reason, n_pairs=raw + 5,
                         shift_east_m=east, shift_south_m=south, shift_m=(east ** 2 + south ** 2) ** 0.5, scale=1.0, angle_deg=0.0))
    add(0, "control", "truth", 30, 30, "ok", 0.0, 1.0, 1.0); add(0, "control", "chosen_candidate", 25, 25, "ok"); add(0, "control", "coarse_peak0", 25, 25, "ok")
    add(1, "skipped", "truth", 30, 30, "ok", 0.0, 2.0, 0.0); add(1, "skipped", "chosen_candidate", 20, 0, "shift_gate"); add(1, "skipped", "coarse_peak0", 0, 0, "too_few_pairs")
    add(2, "skipped", "truth", 3, 3, "ok", 0.0); add(2, "skipped", "chosen_candidate", 1, 1, "ok"); add(2, "skipped", "coarse_peak0", 1, 1, "ok")
    add(3, "skipped", "truth", 30, 30, "ok", 0.0); add(3, "skipped", "chosen_candidate", 2, 2, "ok"); add(3, "skipped", "coarse_peak0", 2, 2, "ok")
    add(3, "skipped", "truth+16,+0", 4, 4, "ok", 16.0)
    return pd.DataFrame(rows)


def make_photos():
    return pd.DataFrame(dict(photo=[0, 1, 2, 3], group=["control", "skipped", "skipped", "skipped"], candidate_offset_px=[2.0, 3.0, 5.0, 40.0],
                             sigma_m=[8.0] * 4, pipeline_inliers_of_chosen_candidate=[25.0, 0.0, 1.0, 5.0]))


def test_tables():
    table = per_photo_table(make_probes(), make_photos())
    assert list(table["cause"]) == ["candidate_is_fine", "gate_removed_a_good_fit", "weak_even_at_truth", "candidate_too_far_from_truth"], list(table["cause"])
    assert cause_counts(table).loc["skipped"].sum() == 3
    curve = offset_curve(make_probes())
    row = curve[(curve["group"] == "skipped") & (curve["offset_px"] == 0.0)].iloc[0]
    assert row["photos"] == 3 and abs(row["share_at_least_good"] - 2 / 3) < 1e-9
    shift = truth_shift_table(make_probes(), make_photos())
    assert sorted(shift["photo"]) == [0, 1, 3] and abs(shift[shift["photo"] == 1]["shift_sigma"].iloc[0] - 0.25) < 1e-9
    compared = compare_with_pipeline(make_probes(), make_photos())
    assert list(compared["difference"]) == [0.0, 0.0, 0.0, 3.0], list(compared["difference"])


def test_choose_photos():
    csv = pd.DataFrame(dict(photo=range(10), answered=[True, False, True, False, True, True, False, True, True, True],
                            points=[10, 0, 6, 0, -20, 3, 0, 10, 0, 10], true_lat=np.arange(10) * 0.001, true_lon=np.arange(10) * 0.002))
    chosen = choose_photos(csv, n_controls=2, max_skipped=2, seed=0)
    assert [p for p, g in chosen if g == "skipped"] == [1, 3]
    controls = [p for p, g in chosen if g == "control"]
    assert len(controls) == 2 and all(csv.loc[p, "answered"] and csv.loc[p, "points"] > 0 for p in controls)
    assert chosen == choose_photos(csv, n_controls=2, max_skipped=2, seed=0), "must be repeatable"
    assert len([1 for p, g in choose_photos(csv, 0) if g == "skipped"]) == 3
    items = [dict(lat=i * 0.001, lon=i * 0.002) for i in range(10)]
    assert check_matches_loaded_photos(csv, items) == []
    items[5]["lat"] += 0.5
    assert len(check_matches_loaded_photos(csv, items)) >= 1
    assert len(check_matches_loaded_photos(csv, items[:5])) == 1


def test_environment_check():
    found = package_versions(["numpy", "no_such_package_xyz"])
    assert found["numpy"] is not None and found["no_such_package_xyz"] is None
    assert missing_files([Path(__file__), "/no/such/file"]) == ["/no/such/file"]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok  ", name)
    print("all tests passed")
