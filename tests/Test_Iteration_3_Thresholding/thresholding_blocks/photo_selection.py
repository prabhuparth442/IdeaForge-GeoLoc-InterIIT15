"""Which photos to probe, and a check that the data set loaded now is the one a saved run was made from. Pandas only."""
import pandas as pd


def choose_photos(per_photo, n_controls, max_skipped=None, seed=0):
    """per_photo: the table of a Single_Image_Run CSV (columns photo, answered, points).
    Returns a list of (photo_index, group): every skipped photo ('skipped'; only the first max_skipped when given) and n_controls
    answered photos that earned points ('control', random but repeatable) for comparison."""
    for column in ("photo", "answered", "points"):
        assert column in per_photo.columns, "the CSV needs the column %s" % column
    assert n_controls >= 0, "n_controls must be >= 0"
    skipped = per_photo[~per_photo["answered"].astype(bool)]
    if max_skipped is not None:
        skipped = skipped.head(max_skipped)
    good = per_photo[per_photo["answered"].astype(bool) & (per_photo["points"] > 0)]
    controls = good.sample(n=min(n_controls, len(good)), random_state=seed) if n_controls > 0 else good.head(0)
    return [(int(p), "skipped") for p in skipped["photo"]] + [(int(p), "control") for p in controls["photo"]]


def check_matches_loaded_photos(per_photo, items, tolerance=1e-6):
    """Compare the true latitude and longitude of three photos in the CSV with the photos loaded now.
    Returns a list of problems (empty list = they match)."""
    for column in ("photo", "true_lat", "true_lon"):
        assert column in per_photo.columns, "the CSV needs the column %s" % column
    problems = []
    if len(per_photo) > len(items):
        problems.append("the CSV has %d photos but only %d are loaded" % (len(per_photo), len(items)))
        return problems
    for position in sorted({0, len(per_photo) // 2, len(per_photo) - 1}):
        row = per_photo.iloc[position]
        item = items[int(row["photo"])]
        if abs(row["true_lat"] - item["lat"]) > tolerance or abs(row["true_lon"] - item["lon"]) > tolerance:
            problems.append("photo %d: the CSV says (%.6f, %.6f), the loaded photo is at (%.6f, %.6f)" % (int(row["photo"]), row["true_lat"], row["true_lon"], item["lat"], item["lon"]))
    return problems
