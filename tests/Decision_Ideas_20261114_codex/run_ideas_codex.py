import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from decision_ideas_codex import decide_photo
from train.checks import candidate_judge as judge
from train.checks import replay
from train.checks.rich_dump import load_rich


def main():
    source = ROOT / "tests/Test_Iteration_7/results"
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    model = joblib.load(source / "judge_t7.joblib")
    rows = []
    for name, folder in (("official150", "official150"), ("nudt707", "nudt_fresh707")):
        raw = pd.read_csv(source / ("signals_ours_" + name + ".csv.gz"))
        table = judge.add_context(judge.add_features(raw))
        table["p"] = judge.judge_new(model, table)
        frozen = judge.decide_frozen(table, table["p"].to_numpy(), 0.9, "pc").set_index("photo")
        for number, (photo_id, group) in enumerate(table.groupby("photo", sort=True)):
            photo = load_rich(source / "rich_dump" / folder / ("photo_%d.npz" % photo_id))
            original = replay.decide_current(photo)
            rows.append(dict(data=name, photo=photo_id, policy="agreement22", points=original["points"],
                             answered=original["answered"], n_queries=original["n_queries"], stage=original["stage"]))
            f = frozen.loc[photo_id]
            rows.append(dict(data=name, photo=photo_id, policy="frozen_judge", points=f.pts,
                             answered=f.stage != "skip", n_queries=f.n_queries, stage=f.stage))
            for policy in ("strict_judge", "unique_location", "heldout_geometry", "unique_geometry"):
                decision = decide_photo(group, photo, policy)
                points = 0
                if decision["answered"]:
                    points = replay.points(replay.error_sigma(photo, decision["x"], decision["y"]))
                rows.append(dict(data=name, photo=photo_id, policy=policy, points=points, **decision))
            if (number + 1) % 50 == 0:
                print(name, number + 1, "done", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "per_photo_codex.csv", index=False)
    summary = []
    for (data, policy), group in results.groupby(["data", "policy"], sort=False):
        base = results[(results.data == data) & (results.policy == "agreement22")].set_index("photo")
        candidate = group.set_index("photo").reindex(base.index)
        gain, low, high = replay.paired_bootstrap(base.points, candidate.points)
        summary.append(dict(data=data, policy=policy, score=replay.score(group.points), wrong=int((group.points < 0).sum()),
                            answers=int(group.answered.sum()), n=len(group), queries=group.n_queries.mean(),
                            gain=gain, ci_low=low, ci_high=high))
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary_codex.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
