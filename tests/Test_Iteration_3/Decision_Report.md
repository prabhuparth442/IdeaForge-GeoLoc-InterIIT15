# Decision step report: where the marks go and what helps

Date: 2026-11-09. Scope: the answer-or-skip decision at the end of the geo-localisation pipeline. Nothing in `src/` was changed and nothing was committed. Every number below comes from notebooks in `tests/Test_Iteration_3/notebooks/` and CSVs in `tests/Test_Iteration_3/results/`.

## 1. Summary

- On the official set the current rule (answer if inliers ≥ 8) leaves about 8 marks on the table. Almost all of that loss is **right-but-skipped** photos, not wrong answers: 0 wrong answers on the official 500 sweep.
- A gate on raw inlier count looked good on official (54–56) but **does not carry over to NUDT** (loses 2–4 marks). Do not use it.
- Two changes helped on both datasets with 0 wrong answers:
  1. **Rescue rule**: also answer at 4–7 inliers when coarse z ≥ 2 and ≥ 2 of the 6 queries agree. Official +2.31 [1.27, 3.58], NUDT +0.92 [0.20, 1.80].
  2. **Wide-query escalation** for photos that end with 0–2 inliers: re-run with 9 headings × 3 scales. Official +0.47, NUDT +0.8, about 4% of photos trigger.
- Together they are worth roughly +3 marks of a ceiling near 57. The rescue-rule parameters were chosen after seeing both sets, so treat the size of the gain as an upper estimate until it is confirmed on photos outside the dev sets.

## 2. Setup

Pipeline: thermal photo → north-up canvas → edge-direction cells → FFT correlation over the 5 km disc → K = 50 candidates per query → DINO layer → LoFTR-MINIMA fine match with RANSAC (M = 10 per query). The answer is the candidate with the most inliers (plus gamma·z). It is answered if inliers ≥ bar 8, otherwise skipped.

Standard query set: 3 headings (−5, 0, +5) × 2 scales (1.0, 1.05) = 6 queries.

Scoring per photo from the error in σ: ≤1σ +10, ≤2σ +6, ≤4σ +3, ≤8σ 0, worse −20, skip 0. Score = points / (10·n) · 60.

Datasets:

| Set | Description | n |
|---|---|---|
| Official | Cartosat / new map, flight heights 100–351 m, photos inside the map window | 379 (also a 500-photo sweep from Member 1, a different pool) |
| NUDT | Changsha, heights ≥ 205 m | 300 (seeded random sample, `default_rng(101)`) |

Reference scores (6 queries, conf 0.22):

| Set | Bar 8 | Always answer | Ceiling (answer iff right) |
|---|---|---|---|
| Official 379 | 49.01 | 54.17 | 57.02 |
| NUDT 300 | 53.00 | 44.68 | 54.68 |

The two sets want opposite things. Official loses by skipping, NUDT loses by answering wrong when told to always answer. Any gate has to behave on both.

## 3. Where the marks are lost

Official 500, conf 0.22, bar 8, 49.0 / 60:

- 510 of 5000 points lost to photos skipped but right.
- 87 points lost to imprecise answers.
- 0 wrong answers.
- 28 photos are far off; 22 of those are at ≤ 125 m flight height.

So the decision step is mostly too cautious, but only where a safe signal exists.

## 4. What was tried in the decision

Per-photo features logged by `train/checks/gate_features.py`: inliers, runner-up inliers, coarse z of the chosen place, peak score and sharpness, number of queries agreeing with the chosen place (`agree_queries`), height, winning heading and scale, and the inlier counts of other candidates (median, p90, count ≥ 5, count ≥ 8).

Evaluation used leave-one-source-out and grouped folds (`grouped_cv_gate` in `train/checks/decision_gate.py`), plus paired bootstrap intervals. AUC is how well a feature separates right from wrong among the low-inlier answers:

| Feature | AUC official | AUC NUDT |
|---|---|---|
| Raw inliers | 0.51 | 0.82 |
| Coarse z | 0.95 | 0.90 |
| Query agreement | 0.77 | 0.93 |

Raw inliers do not separate right from wrong on official, which is why a gate on them fails to transfer. Z and agreement hold on both, and each has a logical reason: a strongly peaked coarse match and several independent queries pointing to the same place are evidence not tied to one matcher count.

### 4.1 Rescue rule

`rescue_flags(table, bar=8, low_bar=4, z_min=2.0, agree_min=2)`: answer if inliers ≥ 8, or if inliers ≥ 4 and z ≥ 2 and agree ≥ 2.

| Variant | Official | NUDT |
|---|---|---|
| Full rule | +2.31 [1.27, 3.58], 0 wrong | +0.92 [0.20, 1.80], 0 wrong |
| Drop agreement | +4.24 | +0.52, 1 wrong |
| Drop both z and agreement | — | −4.0, 13 wrong |

Both conditions are needed. Dropping agreement buys more on official but starts admitting wrong answers on NUDT.

### 4.2 Wide-query escalation

For photos with 0–2 inliers after the standard set, re-run with 9 headings × scales (0.95, 1.0, 1.05). The winning headings are mostly −10° or +7.5°: at low altitude the metadata yaw is often off by more than the ±5° the standard set covers.

| Set | Photos triggered | Recovered | Gain | Wrong |
|---|---|---|---|---|
| Official | 14 | 3 (all right) | +0.47 | 0 |
| NUDT | 12 | 4 (all right) | +0.8 | 0 |

The trigger was fixed before the outcomes were known.

## 5. Tried and rejected

| Idea | Layer | Result |
|---|---|---|
| Gate on raw inliers | Decision | Official +4, NUDT −2 to −4. Not used. |
| Competitor-strength features | Decision | No transferable gain |
| Fused six-map coarse on hard photos | Coarse | Same 3 photos answerable as edge ZNCC |
| M = 50 fine candidates | Fine | No gain, so it is not an M problem |
| Base scale 0.95–1.15 | Image processing | Flat from 1.0 to 1.05 at every height |

High-altitude precision (300–400 m, 20–25% of answers beyond 1σ) is unaffected by scale. Its cause is still unknown.

## 6. Caveats

1. The rescue-rule thresholds (4, 2.0, 2) were chosen after seeing both sets. The bootstrap intervals do not account for that choice.
2. Everything is on the same dev photos. Nothing is confirmed on held-out Cartosat photos.
3. `agree_queries` is computed from the saved per-query results in the notebooks. The real decision step does not compute it yet, so the rule needs that wired in before it can run in the pipeline.
4. Nine low-altitude photos (100–150 m; for example 11, 39, 94, 137, 286, 303) still fail every variant.
5. Not tested yet: LoFTR confidence by height, RoMa or LightGlue on skipped photos, yaw-offset recheck on Cartosat, DINO layer feeding the decision.

## 7. Recommendation

1. Wire the rescue rule and escalation into the decision step on a branch, behind a flag, computing `agree_queries` there.
2. Before adopting, check both on a set of photos not used here (held-out Cartosat), and on a second NUDT seed. Adopt only if the gain holds with 0 wrong answers.
3. Keep the bar at 8 for the default path.

## 8. Files

- Code blocks: `train/checks/decision_gate.py`, `train/checks/gate_features.py`.
- Notebooks (`tests/Test_Iteration_3/notebooks/`, each with an executed copy): `Decision_Gate_Check`, `Gate_Features_Run` and `_v2`, `Gate_Fresh_Eval`, `Rescue_Rule_Check`, `Fine_M50_Rescue`, `Escalate_Wide_Queries`, `Escalate_Fused_Coarse`, `Scale_Sweep`.
- Results: `tests/Test_Iteration_3/results/fresh_run/` (per-photo CSVs and run logs) and `results/input/` (Member 1's sweeps).
- Notion: "Loss hunt: where the marks go and the decision gate (2026-11-09)", https://app.notion.com/p/4130a439e48381ac9e1cd3c8e01e0aab
