"""One fixed exploratory position refinement on the existing regression dumps."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from spatial_median_codex import refine_position
from train.checks import replay
from train.checks.rich_dump import load_rich


def main():
    rows, summaries = [], []
    for data in ("official150", "nudt_fresh707"):
        folder = ROOT / "tests/Test_Iteration_7/results/rich_dump" / data
        for path in sorted(folder.glob("photo_*.npz")):
            photo = load_rich(path, with_matches=False)
            baseline = replay.decide_current(photo)
            x, y, support = baseline["x"], baseline["y"], 0
            if baseline["answered"]:
                keys = replay.STANDARD if baseline["n_queries"] == 6 else replay.WIDE
                queries = [photo["queries"][index] for index in replay.query_ids(photo, keys)]
                x, y, support = refine_position(x, y, photo["sigma"], queries)
            error = replay.error_sigma(photo, x, y)
            points = replay.points(error) if baseline["answered"] else 0
            rows.append(dict(data=data, photo=int(path.stem.split("_")[1]), base_points=baseline["points"],
                             points=points, base_err_sigma=baseline["err_sigma"], err_sigma=error,
                             answered=baseline["answered"], support=support, x=x, y=y))
        group = pd.DataFrame(rows)
        group = group[group["data"] == data]
        gain, low, high = replay.paired_bootstrap(group.base_points, group.points)
        summaries.append(dict(data=data, score=replay.score(group.points), wrong=int((group.points < 0).sum()),
                              gain=gain, ci_low=low, ci_high=high, changed_points=int((group.points != group.base_points).sum()),
                              mean_error_answered=group.loc[group.answered, "err_sigma"].mean(),
                              base_mean_error_answered=group.loc[group.answered, "base_err_sigma"].mean()))
    pd.DataFrame(rows).to_csv(HERE / "results/precision_photos_codex.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "results/precision_summary_codex.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
