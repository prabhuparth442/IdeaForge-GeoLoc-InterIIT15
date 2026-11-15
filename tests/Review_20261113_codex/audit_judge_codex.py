"""Reproduce frozen judge and measure the M=10 to M=5 feature-availability mismatch."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evaluation_fixes_codex import available_candidate_features, select_certified_tau
from train.checks import candidate_judge as judge


def main():
    source = ROOT / "tests/Test_Iteration_7/results"
    model = joblib.load(source / "judge_t7.joblib")
    parts, changed = [], []
    for data in ("official150", "nudt707"):
        raw = pd.read_csv(source / ("signals_ours_" + data + ".csv.gz"))
        old = judge.add_context(judge.add_features(raw))
        available = available_candidate_features(raw, 5)
        old_small = old[old["rank"] < 5].reset_index(drop=True)
        p_old = judge.judge_new(model, old_small)
        p_new = judge.judge_new(model, available)
        changed.append(dict(data=data, candidates=len(available),
                            context_changed=int(((np.abs(old_small.zgap - available.zgap) > 1e-9) |
                                                 (np.abs(old_small.ligap - available.ligap) > 1e-9)).sum()),
                            probability_changed=int((np.abs(p_old - p_new) > 1e-9).sum()),
                            max_probability_change=float(np.max(np.abs(p_old - p_new)))))
        for name, table, probabilities in (("full_context", old_small, p_old),
                                            ("available_context", available, p_new)):
            for variant in ("jetson", "jetson_early"):
                result = judge.decide_frozen(table, probabilities, 0.9, variant)
                result["variant"] = name + "_" + variant
                parts.append(result)
    results = pd.concat(parts, ignore_index=True)
    results.to_csv(HERE / "results/judge_context_photos_codex.csv", index=False)
    summaries = []
    for variant, group in results.groupby("variant"):
        summary = judge.summary(group).reset_index()
        summary["variant"] = variant
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(HERE / "results/judge_context_summary_codex.csv", index=False)
    pd.DataFrame(changed).to_csv(HERE / "results/judge_context_changes_codex.csv", index=False)
    failure = pd.DataFrame(dict(tau=[0.9, 0.99], data="calibration", wrong=[10, 10], n_answered=[100, 100]))
    audit = dict(existing_ltt=judge.ltt_tau(failure, ["calibration"]),
                 isolated_fix=select_certified_tau(failure, ["calibration"]),
                 existing_zero_wrong=judge.smallest_safe_tau(failure, ["calibration"]))
    (HERE / "results/calibration_failure_codex.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(changed).to_string(index=False))
    print(audit)


if __name__ == "__main__":
    main()
