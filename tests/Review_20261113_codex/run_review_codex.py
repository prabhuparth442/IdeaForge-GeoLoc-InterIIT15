"""Run fixed CPU replays on existing real-image dumps; writes only beside this file."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from policies_codex import early_twelve, eligible_rescue, headless_policy
from train.checks import replay
from train.checks.rich_dump import load_rich
from fine_matching.loftr_matcher import fit_matches
from image_processing.map_levels import level_gsd
from header import settings

RESULTS = HERE / "results"
DUMPS = ROOT / "tests/Test_Iteration_7/results/rich_dump"


def refit_valid(photo):
    counts, shifts = [], []
    for query in photo["queries"]:
        count = np.zeros(len(query["inliers"]))
        shift = np.zeros((len(count), 2))
        limit = (settings.MAX_SHIFT_CELLS * settings.CELL +
                 (settings.CONTEXT_MARGIN + query["margin"]) // 2) * level_gsd(photo["level"])
        cv2.setRNGSeed(settings.SEED)
        for rank, match in enumerate(query["matches"]):
            if match is None:
                continue
            fitted = fit_matches(match["points_a"], match["points_b"], match["confidence"],
                                 query["inside"], photo["level"], 0.22)
            if np.linalg.norm(fitted["shift"]) <= limit:
                count[rank] = fitted["n_inliers"]
                shift[rank] = fitted["shift"]
        counts.append(count)
        shifts.append(shift)
    return counts, shifts


def run_photo(photo):
    results = {
        "agreement_baseline": replay.decide_current(photo),
        "eligible_rescue": eligible_rescue(photo),
        "inlier_first": replay.decide_current(photo, gamma=0.0),
        "early12": early_twelve(photo),
        "headless_full18": headless_policy(photo),
        "headless_cap10": headless_policy(photo, cap=10),
        "headless_cap10_stop_second": headless_policy(photo, cap=10, stop_second=True),
    }
    control_counts, control_shifts = replay.refit_queries(photo)
    results["refit_control"] = replay.decide_current(photo, control_counts, control_shifts)
    current_counts, current_shifts = replay.refit_queries(photo, min_confidence=0.21)
    results["current_conf21"] = replay.decide_current(photo, current_counts, current_shifts)
    current = dict(photo)
    current["queries"] = []
    for query, count, shift in zip(photo["queries"], current_counts, current_shifts):
        current["queries"].append(dict(query, inliers=count, shifts=shift))
    results["headless_conf21_cap10"] = headless_policy(current, cap=10)
    results["headless_conf21_full18"] = headless_policy(current)
    valid_counts, valid_shifts = refit_valid(photo)
    results["valid_mask_refit"] = replay.decide_current(photo, valid_counts, valid_shifts)
    return results


def summarise(table):
    rows = []
    for (data, policy), group in table.groupby(["data", "policy"], sort=False):
        reference = "agreement_baseline"
        if policy.startswith("headless"):
            reference = "headless_cap10"
        if policy == "valid_mask_refit":
            reference = "refit_control"
        base = table[(table.data == data) & (table.policy == reference)].set_index("photo")
        current = group.set_index("photo").reindex(base.index)
        gain, low, high = replay.paired_bootstrap(base.points, current.points)
        rows.append(dict(data=data, policy=policy, n=len(group), score=replay.score(group.points),
                         wrong=int((group.points < 0).sum()), answered=int(group.answered.sum()),
                         mean_queries=group.n_queries.mean(), reference=reference, gain=gain,
                         ci_low=low, ci_high=high,
                         changed_points=int((base.points != current.points).sum())))
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    rows, provenance, headroom = [], [], []
    for data in ("official150", "nudt_fresh707"):
        files = sorted((DUMPS / data).glob("photo_*.npz"))
        if len(files) != 150:
            raise ValueError("Expected 150 complete photos in " + data)
        for number, path in enumerate(files):
            photo_id = int(path.stem.split("_")[1])
            photo = load_rich(path)
            provenance.append(dict(data=data, photo=photo_id, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   truth_x=photo["truth"][0], truth_y=photo["truth"][1], height=photo["height"]))
            results = run_photo(photo)
            for policy, result in results.items():
                rows.append(dict(data=data, photo=photo_id, policy=policy,
                                 **{key: result[key] for key in ("points", "answered", "err_sigma", "inliers", "stage", "n_queries", "x", "y")}))
            diagnostic = replay.headroom(photo, replay.query_ids(photo, replay.WIDE))
            headroom.append(dict(data=data, photo=photo_id, baseline_stage=results["agreement_baseline"]["stage"], **diagnostic))
            if (number + 1) % 25 == 0:
                print(data, number + 1, "done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "per_photo_codex.csv", index=False)
    pd.DataFrame(provenance).to_csv(RESULTS / "dump_manifest_codex.csv", index=False)
    pd.DataFrame(headroom).to_csv(RESULTS / "headroom_codex.csv", index=False)
    summary = summarise(table)
    summary.to_csv(RESULTS / "summary_codex.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
