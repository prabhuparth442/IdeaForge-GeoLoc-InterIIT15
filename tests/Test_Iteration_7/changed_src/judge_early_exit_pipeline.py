"""The agreement pipeline with a judge-based early exit (Test_Iteration_7 of palash/dev, checked on our dumps in Ours_Judge_Check).

The standard queries are run one at a time. After each of the first five, every fine-matched candidate gets P(right) from the frozen T7 judge
(train/checks/candidate_judge.py, results/judge_t7.joblib). The choice so far (most inliers + z among candidates with >= 3 inliers) is answered at once
(stage "early") when its P >= early_tau (0.98). If no query reaches that, the photo continues exactly as in AgreementQueryPipeline: the queries already
run are reused, then bar 8 / agreement, then the 27 wide queries at bar 11.

Replay on our dumps (150 official + 150 fresh NUDT): queries per photo 8.4 -> 3.8 (official) and 7.8 -> 3.5 (NUDT); score +0.20 [-1.64, +2.20] and
-0.16 [-0.48, 0.00]; no new wrong answers. Nothing else in the decision changes.

run(item, prior) returns the keys of AgreementQueryPipeline.run, with stage "early" added, and `judge_p` (the P of the choice that exited, else nan).
Needs the LoFTR matcher (the signals read the raw matches of every candidate).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from agreement_query_pipeline import AgreementQueryPipeline
from header.settings import DEVICE
from header.scoring import sigma_m
from image_processing.map_levels import level_for_height
from train.checks.candidate_judge import FEATURES_CTX, add_context, add_features, judge_new
from train.checks.decision_signals import photo_signal_table
from train.checks.rich_dump import RichDumpPipeline

ROOT = Path(__file__).resolve().parents[3]
JUDGE_DIR = ROOT / "tests/Test_Iteration_7/results"
EARLY_TAU = 0.98
MIN_CHOICE = 3                  # the judge has no opinion under 3 inliers (P = 0), and such a candidate is never chosen here
STANDARD_COUNT = 6


def live_query(result):
    """A query result of run_query_list in the layout photo_signal_table reads from a dump."""
    stored = result["matches"]
    return dict(peaks=result["peaks"], z=result["z"], inliers=result["stage2"]["inliers"], shifts=result["stage2"]["shifts"],
                heading=result["heading"], scale=result["scale_factor"], valid_fraction=result["valid_fraction"],
                inside=stored["inside"], matches=stored["items"])


def query_table(result, query_id, level, height, sigma):
    """One row per fine-matched candidate of one query, with every signal and the context features (zgap, ligap), ready for the judge."""
    photo = dict(truth=np.zeros(2), sigma=sigma, height=height, level=level, queries=[live_query(result)])
    rows = photo_signal_table(photo, 0, "live", with_lo=False)
    for row in rows:
        row["query"] = query_id
    return pd.DataFrame(rows)


def pick(table):
    """The pipeline's choice among candidates with >= MIN_CHOICE inliers: most inliers + z, ties by the coarse score."""
    usable = table[table["inliers"] >= MIN_CHOICE]
    if not len(usable):
        return None
    return usable.iloc[int(np.lexsort((-usable["coarse"].to_numpy(), -(usable["inliers"] + usable["z"]).to_numpy()))[0])]


class JudgeEarlyExitPipeline(AgreementQueryPipeline):
    # the dump pipeline's version of candidates_for_query keeps the raw LoFTR matches of every candidate (what the signals need)
    candidates_for_query = RichDumpPipeline.candidates_for_query

    def __init__(self, *args, early_tau=EARLY_TAU, judge_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.5 < early_tau <= 1.0, "early_tau is a probability above 0.5"
        self.early_tau = early_tau
        self.judge = joblib.load(judge_path or JUDGE_DIR / "judge_t7.joblib")
        self._done = []

    def describe(self):
        lines = super().describe()
        lines.append("early exit: answer after any of the first %d standard queries when the T7 judge gives P >= %.2f to the choice" % (STANDARD_COUNT - 1, self.early_tau))
        return lines

    def run_queries(self, item, prior_m):
        # the first step of AgreementQueryPipeline.run: queries already run by the early exit loop are reused, the rest are run
        level = level_for_height(item["rel_height"], self.camera)
        return dict(level=level, queries=self.run_query_list(item, prior_m, level, self.queries, self._done))

    def early_answer(self, item, prior_m):
        """Runs the standard queries one at a time. Returns (result or None, queries run so far)."""
        level = level_for_height(item["rel_height"], self.camera)
        sigma = float(sigma_m(item["rel_height"], self.camera.fx))
        tables, done, start = [], [], time.perf_counter()
        for k in range(1, STANDARD_COUNT):
            done = self.run_query_list(item, prior_m, level, self.queries[:k], done)
            tables.append(query_table(done[-1], k - 1, level, float(item["rel_height"]), sigma))
            table = add_context(add_features(pd.concat(tables, ignore_index=True)))
            table["p"] = judge_new(self.judge, table)
            choice = pick(table)
            if choice is not None and choice["p"] >= self.early_tau:
                return dict(x=float(choice["x"]), y=float(choice["y"]), inliers=float(choice["inliers"]), p=float(choice["p"]),
                            level=int(level), sigma=sigma, queries=done, seconds_judge=time.perf_counter() - start), done
        return None, done

    def run(self, item, prior_m):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        early, self._done = self.early_answer(item, prior_m)
        if early is None:
            answer = super().run(item, prior_m)
            answer["judge_p"] = float("nan")
            answer["total"] = time.perf_counter() - start
            self._done = []
            return answer
        queries = early["queries"]
        answer = self.answer_for(queries, [q["id"] for q in queries])
        answer.update(x=early["x"], y=early["y"], inliers=early["inliers"])
        decision_seconds = early["seconds_judge"] - sum(sum(q["seconds"].values()) for q in queries)
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0.0
        answer.update(seconds=self.seconds_per_stage(queries, max(decision_seconds, 0.0)), gpu_peak_mb=gpu_peak_mb, answered=True, agreement=0,
                      escalated=False, stage="early", sigma_m=early["sigma"], level=early["level"], queries=queries, n_queries_run=len(queries),
                      judge_p=early["p"], total=time.perf_counter() - start)
        self._done = []
        return answer
