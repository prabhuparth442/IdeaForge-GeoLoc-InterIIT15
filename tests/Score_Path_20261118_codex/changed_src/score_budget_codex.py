"""Summarize saved development outputs; return tables and counterfactual bounds.

All paths and band boundaries come from the notebook. Stored points are kept;
rounded output errors are not used to recompute scoring thresholds.
"""
import hashlib

import numpy as np
import pandas as pd


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development(details_path, metadata, split, band_edges, band_labels):
    frame = pd.read_csv(details_path)
    assert frame.filename.is_unique
    assert set(frame.filename) == set(metadata.filename)
    assert split.set_index("filename").loc[frame.filename, "split"].eq("dev").all()
    frame = frame.merge(metadata[["filename", "alt", "cam_fx"]], on="filename", validate="one_to_one")
    frame["band"] = pd.cut(frame.alt, band_edges, labels=band_labels, right=False)
    assert frame.band.notna().all()
    frame["answered"] = frame.status.eq("match")
    frame["wrong"] = frame.answered & frame.points.eq(-20)
    frame["precision_lost"] = np.where(frame.answered & ~frame.wrong, 10 - frame.points, 0)
    frame["wrong_lost"] = np.where(frame.wrong, 30, 0)
    frame["skip_lost"] = np.where(frame.answered, 0, 10)
    frame["error_sigma_rounded"] = frame.error_m / frame.sigma_m
    assert frame.points[~frame.answered].eq(0).all()
    assert np.all(frame.points + frame.precision_lost + frame.wrong_lost + frame.skip_lost == 10)
    return frame


def summarize_group(frame, total_count):
    answered = frame[frame.answered]
    return {
        "n": len(frame),
        "answered": int(frame.answered.sum()),
        "skipped": int((~frame.answered).sum()),
        "wrong_beyond8sigma": int(frame.wrong.sum()),
        "full_points": int(frame.points.eq(10).sum()),
        "six_points": int(frame.points.eq(6).sum()),
        "three_points": int(frame.points.eq(3).sum()),
        "points_per_photo": float(frame.points.mean()),
        "score60_within_group": float(6 * frame.points.mean()),
        "score60_contribution": float(6 * frame.points.sum() / total_count),
        "precision_headroom60": float(6 * frame.precision_lost.sum() / total_count),
        "skip_headroom60": float(6 * frame.skip_lost.sum() / total_count),
        "wrong_headroom60": float(6 * frame.wrong_lost.sum() / total_count),
        "precision_lost_raw": int(frame.precision_lost.sum()),
        "skip_lost_raw": int(frame.skip_lost.sum()),
        "accepted_median_error_sigma_rounded": float(answered.error_sigma_rounded.median()),
        "mean_seconds": float(frame.total_seconds.mean()),
        "p95_seconds": float(frame.total_seconds.quantile(0.95)),
        "max_seconds": float(frame.total_seconds.max()),
    }


def grouped_budget(frame, group_column):
    rows = []
    for name, group in frame.groupby(group_column, observed=True):
        row = summarize_group(group, len(frame))
        row[group_column] = name
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_variants(frames):
    rows = []
    for name, frame in frames.items():
        row = summarize_group(frame, len(frame))
        row["variant"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_bounds(frames, current_name, target_score):
    current = frames[current_name]
    count = len(current)
    score = float(6 * current.points.mean())
    all_points = {}
    for name, frame in frames.items():
        all_points[name] = frame.set_index("filename").points
    points = pd.DataFrame(all_points)
    best_points = points.max(axis=1)
    return {
        "current_score60": score,
        "current_wrong_beyond8sigma": int(current.wrong.sum()),
        "target_score60": target_score,
        "raw_points_needed_for_target": target_score * count / 6 - float(current.points.sum()),
        "perfect_skips_current_answers_unchanged": score + 6 * float(current.skip_lost.sum()) / count,
        "perfect_precision_existing_answers_only": score + 6 * float(current.precision_lost.sum()) / count,
        "truth_picked_best_of_four_score60": float(6 * best_points.mean()),
        "never_answered_by_any_variant": int(points.eq(0).all(axis=1).sum()),
        "has_full_point_option_in_any_variant": int(best_points.eq(10).sum()),
        "illustration_24_perfect_skips_plus_84_six_to_ten": score + 6 * (24 * 10 + 84 * 4) / count,
    }


def summarize_stage_diagnostics(path, band_edges, band_labels):
    frame = pd.read_csv(path)
    frame["band"] = pd.cut(frame.height, band_edges, labels=band_labels, right=False)
    rows = []
    for (variant, band), group in frame.groupby(["variant", "band"], observed=True):
        rows.append({
            "variant": variant,
            "band": band,
            "n": len(group),
            "target_in_K_fraction": float(group.in_K.mean()),
            "target_in_M_fraction": float(group.in_M.mean()),
            "answered_fraction": float(group.answered.mean()),
            "points_per_photo": float(group.points.mean()),
            "wrong_beyond8sigma": int((group.answered & group.points.eq(-20)).sum()),
        })
    return pd.DataFrame(rows)
