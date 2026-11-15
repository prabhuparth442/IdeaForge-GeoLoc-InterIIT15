# Palash's T7 judge vs our agreement + escalation, on our own dumps (2026-11-13)

Notebook: `notebooks/Ours_Judge_Check_executed.ipynb`. Dumps: `results/rich_dump/{official150,nudt_fresh707}` (27 queries, raw LoFTR matches, RTX 3050).
official150 = 150 random official photos (**in-sample for his judge**); nudt707 = 150 fresh NUDT photos no T6/T7 set used (**clean test**).
Judge and rule are his frozen ones, nothing refitted. Scores /60, w = wrong answers.

| Rule | official150 | nudt707 | queries/photo (off / nudt) |
|---|---|---|---|
| bars only (no agreement) | 50.00 / 0w | 53.04 / 1w | 9.4 / 8.1 |
| **ours: agreement + escalation (current best)** | 52.80 / 0w | **53.84 / 1w** | 8.4 / 7.8 |
| ours + DINO-z veto | 52.80 / 0w | 53.44 / 1w | 8.2 / 7.8 |
| judge (PC) | **56.80 / 0w** (in-sample) | 53.44 / 1w | 7.3 / 8.1 |
| judge (PC) + early exit | 55.40 / 0w | 53.28 / 1w | 3.2 / 3.8 |
| judge (Jetson, M=5) | 56.80 / 0w | 53.44 / 1w | 6.2 / 6.4 |
| judge (Jetson) + early exit | 55.40 / 0w | 53.28 / 1w | 2.1 / 2.0 |
| bars only, Jetson | 51.20 / 0w | 53.04 / 1w | 6.6 / 6.4 |
| ours OR judge (our answer, else the judge's) | 57.20 / 0w | 54.24 / 1w | 8.4 / 7.8 |
| ours AND judge (judge as veto) | 52.40 / 0w | 53.04 / 1w | 8.4 / 7.8 |

## Reading
- The judge's official gain (+4.0 over ours) is in-sample, so not evidence. On clean NUDT the judge is -0.40 vs our rule: it gains photo 1636 and loses photos 2066 and 2181, which our agreement rule answers right.
- Early exit is the robust piece: about 8 to 3.5 queries (PC), 6.4 to 2.0 (Jetson), for about -0.6 on NUDT. At 2.2 s per query that is about 14 s to 4.5 s per photo on the Jetson (modelled).
- One wrong answer on NUDT under every rule; the judge does not fix it (a recall problem per T6).
- Our DINO-z veto cost -0.40 on nudt707 (within noise at n = 150), so it did not help here.
- Judge as an OR rescue on top of ours: +0.40 NUDT, no new wrong answers; as a veto (AND) it loses points.

## Recommendation
1. Keep agreement + escalation; do not replace it with the judge.
2. Add early exit if the Jetson budget binds (about half a mark).
3. Judge as an OR rescue is the only add-on that removes nothing; confirm on a larger fresh NUDT set.
4. Nothing removed from the old rule is supported by this check.

## Early exit added (changed_src)

`changed_src/judge_early_exit_pipeline.py`: `JudgeEarlyExitPipeline(AgreementQueryPipeline)`. Standard queries run one at a time; after each of the first five, the frozen T7 judge scores
every candidate and the choice is answered at once (stage `early`) when P >= 0.98. Otherwise the photo continues exactly as in the agreement pipeline (queries already run are reused).
Needs `train/checks/{candidate_judge,decision_signals,rich_dump}.py` and `results/judge_t7.joblib`; it adds a new file only, nothing in `src/` or the root pipelines is edited.

- Replay on our dumps (ours + early exit vs ours): queries 8.4 -> 3.8 (official150), 7.8 -> 3.5 (nudt707); score +0.20 [-1.64, +2.20] and -0.16 [-0.48, 0.00]; 0 new wrong answers (132 and 134 of 150 photos exit early).
- Live check (`notebooks/Early_Exit_Live_Check_executed.ipynb`, 15 official photos, GPU): same stage, same place and same query count as the replay on 15 of 15. Early photos took 0.6-2.0 s (PC) vs about 6 s for a photo that runs all 6 standard queries; escalated photos about 14-26 s.
- Not measured: Jetson time (modelled 2.2 s per query), a larger fresh NUDT set, the judge's CPU cost on the Orin (the signals are numpy/OpenCV on about 5 candidates per query).

## Ours vs the T10 rules, override off, inflation on everything (2026-11-15)

`notebooks/Ours_vs_T10_Rules_executed.ipynb`, results `ours_vs_t10_scales.csv`, `ours_vs_t10_v5small.csv`. Frozen T7 judge (trained with official + NUDT; official150 is in-sample), cuts from T10: J0 0.9 + override 12, J1 0.9 no override, J2 0.98 no override. Every inlier count x f for every rule, ours included. Score /60 / wrong.

nudt707 (clean for the T7 judge; 46 photos are also in his 707, 46 in his 808):

| rule | x1 | x1.25 | x1.5 | x2 |
|---|---|---|---|---|
| bars (no agreement) | 53.04 / 1w | 53.84 / 1w | 51.44 / 4w | 49.84 / 6w |
| ours (agreement + escalation) | 53.84 / 1w | 53.84 / 1w | 51.44 / 4w | 49.84 / 6w |
| J0 judge + override 12 | 53.44 / 1w | 53.44 / 1w | 53.44 / 1w | 53.44 / 2w |
| J1 judge, no override | 53.44 / 1w | 53.44 / 1w | 53.44 / 1w | 53.44 / 1w |
| J2 judge 0.98, no override | 54.24 / 0w | 54.24 / 0w | 54.24 / 0w | 54.24 / 0w |
| ours OR J1 | 54.24 / 1w | 54.24 / 1w | 51.44 / 4w | 49.84 / 6w |

official150 (in-sample for the judge): ours 52.80 / 54.40 / 54.80 / 56.40 with 0 wrong at every scale; J1 56.80-57.20; J2 54.00-55.84; all 0 wrong.

- Our rule has the same inflation weakness as the bars: 4 wrong at x1.5 and 6 at x2 on NUDT (official stays 0 wrong, answering more of a safe set).
- The one wrong answer at x1 (photo 1181) is made by bars, ours, J0, J1; only J2 (cut 0.98) avoids it, at no loss on NUDT707 (54.24). J2's cut was picked in T9/T10 after 707 was seen, so this is not independent evidence.
- V5-small (relative + augmentation, trained on 150 photos of the other set): 51.68 / 3w on nudt707, 54.00 / 0w on official150; too little training data to judge V5.
- Not tested: his V5 / S1z as trained (design signal tables and satellite-made photos are not in git); a fresh NUDT draw.

## Fresh NUDT 1010 (133 photos nobody has seen; 2026-11-15)

Draw: seed 1010, 150 requested, 133 left in the pool after excluding 1,717 photos (his 404-909 draws, our 707, T7-seen, the 500-pool). Dumped on the RTX 3050 under `tools/run_capped.sh` (45 min, 0 failures; `scripts/run_fresh1010.sh`).
Rules and cuts fixed before the dump (frozen T7 judge; J0 0.9 + override 12, J1 0.9, J2 0.98). Notebook `notebooks/Ours_vs_T10_Fresh1010_executed.ipynb`, results `ours_vs_t10_fresh1010.csv`. Score /60 / wrong:

| rule | x1 | x1.25 | x1.5 | x2 |
|---|---|---|---|---|
| bars (no agreement) | 53.50 / 1w | 54.14 / 1w | 54.14 / 2w | 51.11 / 5w |
| ours (agreement + escalation) | 54.09 / 1w | 54.72 / 1w | 54.27 / 2w | 51.11 / 5w |
| J0 judge + override 12 | 55.04 / 0w | 55.49 / 0w | 54.72 / 1w | 54.27 / 2w |
| J1 judge, no override (0.9) | 55.04 / 0w | 55.49 / 0w | 55.62 / 0w | 56.08 / 0w |
| J2 judge, no override (0.98) | 55.04 / 0w | 55.04 / 0w | 55.04 / 0w | 55.04 / 0w |
| ours OR J1 | 54.54 / 1w | 54.72 / 1w | 54.27 / 2w | 51.11 / 5w |

Paired gain over ours at x1: J0, J1, J2 +0.95 [-0.68, +3.20] (all three answer the same photos); at x2 J1 +4.96 [+0.90, +9.92]. Our one wrong answer at x1 is photo 111 (also bars); the judges skip it.
Reading: same pattern as 707 on unseen photos. At x1 the judge is +0.95 over ours (not proven at n = 133), and it is the only family with 0 wrong at x1.5 and x2. Ours fails like the bars (2 wrong at x1.5, 5 at x2). The frozen judge is trained with official + NUDT (not satellite-only), so this shows what the decision could be, not what can ship.

## Inflation guard for our own rule (2026-11-15) - promising, not adopted

`notebooks/Ours_Inflation_Guard_executed.ipynb`, `ours_inflation_guard.csv`. Label-free guard: running mean inliers of the far (> 4 sigma) candidates, prior = larger of official150 and nudt707 levels; the three inlier numbers (bars 8 / 11, agreement floor 3) are multiplied by max(1, level / prior). Nothing fitted to labels. Score /60 / wrong, unguarded -> guarded:

| set | x1 | x1.25 | x1.5 | x2 |
|---|---|---|---|---|
| official150 | 52.80 / 0w, unchanged | 54.40 / 0w, unchanged | 54.80 / 0w, unchanged | 56.40 / 0w, unchanged |
| nudt707 | 53.84 / 1w, unchanged | 53.84 / 1w, unchanged | 51.44 / 4w -> 52.24 / 3w | 49.84 / 6w -> 52.24 / 3w |
| nudt1010 (fresh) | 54.09 / 1w -> 53.37 / 1w | 54.72 / 1w -> 54.09 / 1w | 54.27 / 2w -> 53.64 / 1w | 51.11 / 5w -> 54.27 / 1w |

Gain at x2: nudt707 +2.40 [0, +5.60], nudt1010 +3.16 [0, +7.22]. But on clean nudt1010 (x1) the guard costs -0.72 [-1.89, 0.00] because that set's far level is above the prior, so it raises the bars without any inflation. Not adopted: it needs a dead zone (act only above a factor the dev sets never reach, e.g. 1.3-1.5), chosen from the dev spread of far levels (0.19 official, 0.24 NUDT) and pre-registered, then checked on one more fresh draw. Choosing it now from nudt1010 would be tuning on the test set.

## Ideas-planner tests (2026-11-15)

**Idea 7, disjoint-half agreement (`notebooks/Idea7_Halves_Agreement_executed.ipynb`, `idea7_halves.csv`): dead.** Rule fixed before looking: both halves (left/right or top/bottom) keep >= 3 inliers and their fits land within 1 sigma of each other. Rescue of the photos ours skips: 0 of 33 (one false pass, wrong). As a veto on ours: removes 0-2 wrong answers per set but loses 13-40 right ones (official: 0 wrong removed, 40 right lost). AUC on NUDT 0.93-0.95, no more than the inlier count itself.

**Idea 0, oracle test at low altitude (`notebooks/Idea0_Oracle_Low_Altitude_executed.ipynb`, `idea0_oracle*.csv`).** True place injected as the only candidate; LoFTR over heading -15..+15 (step 5) x scale 0.80..1.25 (49 poses) on 12 official150 photos at <= 150 m that ours skips, and 8 controls ours answers right. Best inliers at the truth: skipped photos mean 4.4 / median 4 / max 14, only 1 of 12 reaches 8 (photo 172, at heading +10, 14 inliers), 1 more is at 7; controls mean 21 (several 7-9 at 100 m). The matches that exist sit at the right place (shift 0.2-1.15 sigma). So for about 10 of 12 the loss is LoFTR matching at the true place, not retrieval, and a wider heading / scale grid does not fix it (the grid already covers +-15 degrees and 0.8-1.25). n = 12 + 8 from one area.

**Inflation guard, pre-registered confirmation (`results/preregistered_guard.json`, `notebooks/Ours_Inflation_Guard_Valid1212_executed.ipynb`).** NUDT valid split (other flights), seed 1212, 150 photos, dead zone 1.5, level from all 27 queries. Ours unguarded -> guarded, score / wrong at x1, 1.25, 1.5, 2: 51.60 / 3w unchanged; 51.60 / 4w -> 52.40 / 2w; 48.80 / 7w -> 52.00 / 2w; 45.60 / 11w -> 52.00 / 2w. Paired gain +0.80 [-1.60, +3.60], +3.20 [-0.01, +7.20], +6.40 [+2.00, +11.60]. All three pre-registered criteria met: adopted. Note: ours makes 3 wrong answers at x1 on this valid split (vs 0-1 on test-split sets).
Deployable form (`inflation_guard_pipeline.py`, repo root): the level can only use the 6 standard queries, prior 0.169 (same recipe, 6 queries). Replay wrong answers at x1 / 1.25 / 1.5 / 2: valid1212 3/3/4/3 (unguarded 3/4/7/11), nudt707 1/1/3/3 (1/1/4/6), nudt1010 1/1/2/2 (1/1/2/5), official150 unchanged; clean answers identical on all four sets. Weaker than the pre-registered all-queries form on valid1212 at x1.5-2 (4 and 3 wrong, not 2 and 2).
Live check of the deployable class (`notebooks/Inflation_Guard_Live_Check_executed.ipynb`, 10 official photos covering bar, agreement, escalation, skip; unit tests `tests/pipeline/test_inflation_guard.py`, 4 pass): stage, place, query count and the far-candidate level equal the replay on 10 of 10; guard factor 1.0 on every photo (clean data).

## Ideas 9, 5 and 1 of the ideas planner (2026-11-15)

**Premises first.** Camera 320 x 256, fx 200 px: a photo at 100-150 m is 0.5-0.75 m/px, the map 0.3 m/px, our lowest level 3 is 0.85 m/px, and `make_query` already blurs the photo to the canvas sharpness. So idea 9's premise (photo finer than the map) is inverted and the blur is built in; the real variant is a FINER canvas. Idea 5's aligned token comparison is what `dino_rerank.py` already does (window cut at the peak, token-to-token cosine, layer 8); only the depth (K = 200) is new.

**Idea 9, finer level / extra blur at the true place (`notebooks/Ideas_9_1_Finer_Level_Pose_Feedback_executed.ipynb`, `idea9_variants*.csv`): dead.** 12 skipped + 8 controls, 25 poses, best inliers (mean): level 3 as today skipped 4.6 / control 16.0; level 3 + 1.5 px blur 0.0 / 2.5; level 2 (0.60 m/px) 2.5 / 6.8. Both variants are worse than today on both groups. Level 1 was dropped (map build exceeded the 6 GB cap).

**Idea 1, pose feedback at the true place (`idea1_pose_feedback*.csv`): inconclusive, small.** A finer grid around each photo's best pose lifts the skipped photos' best from 4.6 to 8.0 mean and 4 of 12 reach 8 (2 of 12 before), but the inlier count is RANSAC-noisy: photo 200 gave 4 in the Idea 0 run and 14 here at the same pose. A maximum over more poses of a noisy count is biased upward, so this does not show a gain; it would need repeated RANSAC seeds before any claim.

**Idea 5, deeper shortlist (`notebooks/Idea5_Deep_Shortlist_DINO_executed.ipynb`, `idea5_deep_shortlist*.csv`): dead.** NUDT valid-split photos never retrieved (9; no peak within 4 sigma in any of 27 queries) and 10 retrieved controls, 6 standard queries, K = 200. The true place is in the 200 coarse peaks for 1 of 9 (rank 71) and reaches the fused top 10 for 0 of 9, at depth 50 and at depth 200. Controls: 10 of 10 stay in the top 10 at both depths. Falsifier was fewer than a third reaching the top 10. n = 9, one split, 6 queries.

**Idea 1, scale coverage against height metadata +20 % (`notebooks/Idea1_Scale_Coverage_Dump*`, `Idea1_Scale_Coverage_Analysis_executed.ipynb`, `idea1_scale_coverage.csv`): partial.** 60 official photos, 15 queries (headings -5/0/+5 x scales 0.80, 0.85, 0.90, 1.00, 1.05). Today's 6 standard queries: clean 54.8, height +20 % 18.8. All 15 queries at bar 8: 47.8 / 44.5 (0 wrong). Standard 6 then escalate to the 15 at bar 11: 54.8 / 43.6, 0 wrong, 6.6 queries on clean photos (answers identical to today on clean). It recovers about 70 % of the loss at no clean cost, but is 11 points under clean, so it misses Fable's "within about 3 points" test. Test_Iteration_9's numbers for the same photo list (48.8 / 31.8, 27 queries) are not comparable with these (my 6-query clean baseline is 54.8); the cause was not traced.

## Ideas 6 and 8 of the ideas planner (2026-11-15)

**Idea 6, map self-similarity (`notebooks/Idea6_Map_Twins_executed_{official,nudt}.ipynb`, `idea6_twins*.csv`): dead; time cost measured.** The satellite tile at a candidate place is correlated against the 5.5 km disc around it by the pipeline's own coarse search. Right candidate (best-inlier within 1 sigma, >= 3 inliers) vs hard-wrong candidate (best-inlier beyond 4 sigma, >= 3 inliers): official150 145 / 90, NUDT valid1212 + fresh1010 + fresh707 404 / 417. Twin count (other places within 0.05 of the own score) is 0 for every candidate (own score about 0.72-0.82, best other place 0.06-0.12). AUC of own-minus-best-other score, wrong vs right: official 0.64, NUDT 0.64 (0.62-0.66 per draw), 0.55 / 0.67 restricted to 3-11 inliers; the inlier count itself gives 0.96 / 0.98. Planned bar was AUC >= 0.7 on both domains; missed. The wrong candidates are LoFTR false matches, not twin places. Caveat: the template was the full tile, not the photo's valid mask. Time: one extra correlation per photo costs 0.18 s mean (median 0.11, p90 0.47) on official and 0.20 s on NUDT on this PC (RTX 3050, cuFFT), against about 0.75 s per query in the dumps. Jetson not measured (without cuFFT the 6-to-18-query photo takes about 13 s, so one more correlation is an estimate of 1-2 s).

**Idea 8, input canonicalisation (`notebooks/Idea8_Estimators*.ipynb`, `idea8_estimators.csv`): estimators fail, end-to-end not run.** Facts found first: the coarse edge bins are already unsigned (angle mod pi), so a polarity flip cannot hurt the coarse stage; the loss of an inverted photo sits in DINO / LoFTR. Estimators on the 60 official photos (clean / invert / blur), read at the true place and at the top peak (the top peak is within 4 sigma for 52 of 60): signed-gradient cosine (polarity) has median 0.03-0.05 only, negative for 20 % (truth) / 13 % (peak) of CLEAN photos and 80 % / 85 % of inverted ones (AUC 0.85), so a rule "flip if negative" would flip 13-20 % of clean photos. Gradient-magnitude ratio photo / map (blur): median 1.99 clean vs 1.60 blurred (AUC 0.76); no photo falls below 0.75 even when blurred, the clean spread overlaps, so no absolute threshold separates them. Because the planned falsifier was a cost on clean data, and the estimators misfire on clean photos, the end-to-end test was not run. Trying both polarities for every photo would double the queries and is a different idea.
