"""Tables for Step 1 (pandas only, testable without torch). The notebook shows what these return; the logic lives here.

probes : one row per (photo, tile): photo, group, label, offset_px, n_pairs, raw_inliers, inliers, reason, shift_east_m, shift_south_m, ...
photos : one row per photo: photo, group, sigma_m, candidate_offset_px (chosen candidate to truth, map pixels), pipeline_inliers_of_chosen_candidate
"""
import pandas as pd

from thresholding_blocks.probe_blocks import classify_cause


def per_photo_table(probes, photos, good_count=8, near_px=12.0):
    """One row per photo: what the tile at the truth gave, what the chosen candidate gave, what coarse peak 0 gave, and the cause label."""
    for column in ("photo", "label", "n_pairs", "raw_inliers", "inliers", "reason"):
        assert column in probes.columns, "probes needs the column %s" % column
    assert "candidate_offset_px" in photos.columns and "group" in photos.columns, "photos needs group and candidate_offset_px"
    rows = []
    for _, info in photos.iterrows():
        mine = probes[probes["photo"] == info["photo"]].set_index("label")
        assert "truth" in mine.index and "chosen_candidate" in mine.index, "photo %s lacks a truth or chosen_candidate probe" % info["photo"]
        truth, chosen = mine.loc["truth"], mine.loc["chosen_candidate"]
        row = dict(photo=info["photo"], group=info["group"], candidate_offset_px=info["candidate_offset_px"],
                   truth_raw_inliers=int(truth["raw_inliers"]), truth_pairs=int(truth["n_pairs"]), truth_reason=truth["reason"],
                   candidate_raw_inliers=int(chosen["raw_inliers"]), candidate_inliers=int(chosen["inliers"]),
                   candidate_pairs=int(chosen["n_pairs"]), candidate_reason=chosen["reason"])
        if "coarse_peak0" in mine.index:
            row["peak0_raw_inliers"] = int(mine.loc["coarse_peak0"]["raw_inliers"])
        row["cause"] = classify_cause(row["truth_raw_inliers"], row["candidate_raw_inliers"], row["candidate_reason"], row["candidate_offset_px"], good_count, near_px)
        rows.append(row)
    return pd.DataFrame(rows)


def cause_counts(table):
    """Photos per cause, split by group."""
    return table.groupby(["group", "cause"]).size().unstack(fill_value=0)


def offset_curve(probes, good_count=8):
    """For the tiles around the truth: per group, the median raw inliers and the share of photos with at least good_count, by offset in pixels.
    Rows of other labels (chosen_candidate, coarse_peak0) have offset_px NaN and are skipped."""
    for column in ("group", "offset_px", "raw_inliers"):
        assert column in probes.columns, "probes needs the column %s" % column
    around = probes[probes["offset_px"].notna()].copy()
    around["enough"] = (around["raw_inliers"] >= good_count).astype(float)
    grouped = around.groupby(["group", "offset_px"])
    table = pd.DataFrame({"median_raw_inliers": grouped["raw_inliers"].median(), "share_at_least_good": grouped["enough"].mean(), "photos": grouped.size()})
    return table.reset_index()


def truth_shift_table(probes, photos, good_count=8):
    """Signed shift of the canvas centre on the tile at the true position, for photos that matched there (raw inliers >= good_count).
    A shift that is not near zero is a bias between the photo's centre and the GPS truth (or a map offset). Also in sigma units."""
    truth = probes[(probes["label"] == "truth") & (probes["raw_inliers"] >= good_count)]
    merged = truth.merge(photos[["photo", "sigma_m"]], on="photo", how="left")
    merged["shift_sigma"] = merged["shift_m"] / merged["sigma_m"]
    return merged[["photo", "group", "raw_inliers", "shift_east_m", "shift_south_m", "shift_m", "shift_sigma", "scale", "angle_deg"]]


def compare_with_pipeline(probes, photos):
    """The probe on the chosen candidate must reproduce what the pipeline itself got for that candidate (same inliers, up to RANSAC randomness)."""
    chosen = probes[probes["label"] == "chosen_candidate"].merge(photos[["photo", "pipeline_inliers_of_chosen_candidate"]], on="photo")
    chosen["difference"] = (chosen["inliers"] - chosen["pipeline_inliers_of_chosen_candidate"]).abs()
    return chosen[["photo", "inliers", "pipeline_inliers_of_chosen_candidate", "difference"]]
