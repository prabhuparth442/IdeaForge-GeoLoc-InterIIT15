"""Block: evaluation bookkeeping for Test_Iteration_10. No GPU, no plotting.

manifest(...)                 one dict with every setting a result depends on (search, matcher, queries, decision, judge training data);
                              save_manifest writes it next to a result file (<result>.manifest.json)
HEADLESS                      the submitted headless pipeline's settings as documented on the team's Notion page "Main pipeline
                              (Main_Pipeline_Headless) submitted state" (2026-11-11). The code itself is not on this machine, so every value
                              here is as documented, not read from the code
compare_with_headless(m)      every setting where a research result differs from the submission (or is not comparable)
failure_log(table, result, variant)
                              per photo: answered_right (+10/+6/+3) / answered_off (0 points, 4-8 sigma) / answered_wrong (-20) /
                              retrieved_but_unmatched (skip; a candidate within 4 sigma was among those this rule looked at, but fine
                              matching gave it < 3 inliers) / retrieved_but_rejected (skip; such a candidate had >= 3 inliers and the rule
                              did not answer) / never_retrieved (skip; no candidate within 4 sigma, by its coarse peak or its fitted place,
                              in any query this rule ran, at its M)
"""
import json

import numpy as np
import pandas as pd

from train.checks.candidate_judge import SECOND_PASS, STANDARD, WIDE

HEADLESS = dict(
    source="Notion: Main pipeline (Main_Pipeline_Headless) submitted state, 2026-11-11 (code not on this machine)",
    coarse="edge-direction cells 4 px, 8 bins, FFT masked ZNCC, disc 5500 m", k_peaks=50, dino_weight=100, m_fine=5,
    matcher="LoFTR (kornia) + MINIMA weights", loftr_confidence=0.21, ransac_px=3, gamma=1.0, seed=7,
    queries_first="headings -5, 0, +5 x scales 1.0, 1.05 (6)", queries_second="9 headings x 2 scales (18 in all), only if no answer",
    query_cap=None, time_budget_s=22.0, decision="pooled candidates, winner by inliers + DINO z, answer at >= 8 inliers (both passes)",
    override=None, min_inliers_choice=1, judge=None, early_exit=None, attitude_correction=None,
    hint="dev hint = truth in the Jetson run (no 5 km offset)")


def manifest(variant, k_peaks=50, m_fine=None, queries=None, cap=None, cut=None, wide_cut=None, override=None, min_choice=3, early_exit=None,
             bars=None, judge_features=None, judge_trained_on=None, calibration=None, matcher_confidence=0.22, loftr_coarse_threshold=0.21,
             hint="5 km random hint (evaluation.testset.random_prior), search disc 5500 m", data=None, note=None):
    """variant: 'pc' (6 standard queries, then all 27 at once) or 'jetson' (6 standard, then SECOND_PASS one at a time up to cap)."""
    if queries is None:
        queries = dict(first=[list(q) for q in STANDARD],
                       second=[list(q) for q in (WIDE if variant == "pc" else SECOND_PASS) if q not in STANDARD])
    return dict(variant=variant, k_peaks=k_peaks, m_fine=m_fine if m_fine is not None else (10 if variant == "pc" else 5), queries=queries,
                cap=cap if cap is not None else (27 if variant == "pc" else 10), cut=cut, wide_cut=wide_cut, override=override, min_choice=min_choice,
                early_exit=early_exit, bars=bars, judge_features=judge_features, judge_trained_on=judge_trained_on, calibration=calibration,
                matcher="LoFTR (kornia) + MINIMA weights", matcher_confidence=matcher_confidence, loftr_coarse_threshold=loftr_coarse_threshold,
                ransac_px=3, gamma=1.0, dino_weight=100, coarse="edge ZNCC, cells 4 px, 8 bins", hint=hint, data=data, note=note)


def save_manifest(result_path, m):
    with open(str(result_path) + ".manifest.json", "w") as f:
        json.dump(m, f, indent=1, default=str)


def compare_with_headless(m):
    """Rows (setting, research, headless, note) for every difference."""
    rows = []

    def diff(name, mine, theirs, note=""):
        if mine != theirs:
            rows.append(dict(setting=name, research=str(mine), headless=str(theirs), note=note))
    diff("M (candidates fine-matched per query)", m["m_fine"], HEADLESS["m_fine"])
    diff("K (coarse peaks)", m["k_peaks"], HEADLESS["k_peaks"])
    n2 = len(m["queries"]["second"])
    diff("second-pass queries", "%d extra (%s)" % (n2, "all at once" if m["variant"] == "pc" else "one at a time"), "12 extra, as one pass",
         "PC replay uses 27 queries incl. scale 0.95; the headless second pass has no 0.95 scale")
    diff("query cap", m["cap"], "none (22 s time budget)", "the research Jetson replay models the budget as a 10-query cap")
    diff("matcher confidence cut", m["matcher_confidence"], HEADLESS["loftr_confidence"],
         "research: matches >= 0.22 kept, LoFTR coarse threshold 0.21; the Notion page lists one value (0.21): check which one it is")
    if m["cut"] is not None:
        diff("decision", "judge P >= %s (extra queries %s), min %s inliers" % (m["cut"], m["wide_cut"], m["min_choice"]), HEADLESS["decision"])
    elif m["bars"] is not None:
        diff("decision", "bars %s, min %s inliers" % (m["bars"], m["min_choice"]), HEADLESS["decision"])
    diff("override (answer at >= n inliers whatever P)", m["override"], HEADLESS["override"])
    diff("early exit", m["early_exit"], HEADLESS["early_exit"])
    diff("judge training data", m["judge_trained_on"], None)
    diff("hint", m["hint"], HEADLESS["hint"], "the Jetson submission run used the truth as hint; research uses a 5 km random hint")
    return pd.DataFrame(rows)


def failure_log(table, result, variant):
    """result: candidate_judge.decide / baseline output on `table`. The candidates a rule looked at: rank < M and, per photo, the queries it ran
    (PC: all 27 when it got past the standard 6; Jetson: the standard 6 then SECOND_PASS in order, n_queries in all; early exit: the standard
    queries in order, n_queries of them)."""
    m = 10 if variant == "pc" else 5
    std = [(round(h, 2), round(s, 2)) for h, s in STANDARD]
    extra = [(round(h, 2), round(s, 2)) for h, s in (WIDE if variant == "pc" else SECOND_PASS) if (round(h, 2), round(s, 2)) not in std]
    order = std + extra
    t = table[table["rank"] < m]
    near = t[(t["err_sigma"] <= 4) | (t["peak_err_sigma"] <= 4)]
    right_queries = {k: set(g) for k, g in near.groupby(["data", "photo"])["qkey"]}
    matched_queries = {k: set(g) for k, g in near[near["inliers"] >= 3].groupby(["data", "photo"])["qkey"]}
    rows = []
    for _, r in result.iterrows():
        key = (r["data"], r["photo"])
        ran = set(order[:int(r["n_queries"])]) if r["n_queries"] > 0 else set()
        found = bool(right_queries.get(key, set()) & ran)
        anywhere = key in right_queries
        if r["stage"] != "skip":
            kind = "answered_right" if r["pts"] > 0 else ("answered_wrong" if r["pts"] < 0 else "answered_off")
        else:
            matched = bool(matched_queries.get(key, set()) & ran)
            kind = "never_retrieved" if not found else ("retrieved_but_rejected" if matched else "retrieved_but_unmatched")
        rows.append(dict(data=r["data"], photo=r["photo"], outcome=kind, right_in_any_query=anywhere, n_queries=r["n_queries"]))
    return pd.DataFrame(rows)
