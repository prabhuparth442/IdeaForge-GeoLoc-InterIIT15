# Test_Iteration_7: the decision after fine matching, without agreement

**Brief.** Improve the answer-or-skip decision using only what exists after fine matching, without multi-query agreement (the team ruled it out
as too risky). Two routes were explored: (a) make right places produce **more lines**; (b) use **all the signals** up to that point to judge the
decision. Gains count only when they hold on held-out sets with 0 wrong answers, with paired-bootstrap intervals, and the Jetson time is reported
in the team's classes (**< 4 s good, 4-6 s viable, >= 6 s slow**; read here as the time a method adds to or removes from a photo).

Full write-up: [REPORT.md](REPORT.md).

## Result in one table

Held-out scores out of 60 (leave one set out: judge and cut fitted on the other four sets) and the blind set (NUDT fresh 606, 300 photos, scored
once after the rules were frozen). `w` = wrong answers. Baselines are agreement-free: PC = bar 8 on the 6 standard queries, else bar 11 on all 27;
Jetson today = bar 8, second pass, cap 10, M = 5.

| | official | NUDT 101 | NUDT 202 | NUDT 404 | NUDT 505 | **blind 606** | LoFTR queries | Jetson s / photo (model) |
|---|---|---|---|---|---|---|---|---|
| PC baseline | 49.96 | 54.84 | 57.06 | 54.24 / 2w | 54.24 / 1w | 55.08 / 2w | 7.5-9.5 | — |
| **PC judge** | 54.84 | 55.72 | 57.06 | 56.70 | 56.76 | 55.80 / 2w | 7.5-8.3 | — |
| Jetson today | 50.28 | 54.24 | 55.26 | 55.44 | 54.68 / 1w | 54.42 / 2w | 6.3-6.6 | 14.1 |
| **Jetson judge** | 53.16 | 54.92 | 55.26 | 57.00 | 55.56 | 54.82 / 1w | 6.2-6.4 | 14.0 |
| **Jetson judge + early exit** | 52.50 | 54.84 | 54.78 | 57.36 | 55.32 | 54.50 / 1w | **1.7-2.5** | **4.5** |

Paired gains over the baseline (95 % bootstrap): PC judge official +4.88 [+3.26, +6.62], 101 +0.88 [+0.08, +1.80], 404 +2.46 [+0.48, +4.98],
505 +2.52 [+1.20, +4.04], 202 0; blind 606 +0.72 [-0.56, +2.12]. Jetson judge: official +2.88 [+1.52, +4.31], 404 +1.56 [+0.48, +2.94];
blind 606 +0.40 [-0.60, +1.60].

## What it means

1. **More lines for right places: no.** Re-matching the stored pairs (guided matching, LO-RANSAC, translation consensus) adds ~0 lines; a second
   LoFTR look with the tile re-centred on the fitted place adds +0.6 lines on average (re-aiming heading/scale loses lines). The second look is a
   repeatability test instead (wrong places drop from median 4.5 to 0 lines), but as a judge input it *lowered* official by 1.08: dropped.
2. **Judge all the signals: yes on held-out sets, not proven on the blind set.** A monotone gradient-boosted judge trained on every fine-matched
   candidate (9,600 wrong and 23,500 right labelled candidates with >= 3 inliers) removes all the fresh-NUDT wrong answers of the bars and gains
   +0.9 to +4.9 on four of five held-out sets, 0 wrong everywhere. On the blind set it is +0.7 (PC) / +0.4 (Jetson) with intervals through 0; it
   removes one wrong answer but adds another (a **3-inlier** answer with high DINO z). Wrong answers on the blind set: PC 2 (baseline 2),
   Jetson 1 (baseline 2). By the brief's rule this is **not yet an adoptable gain**.
3. **Early exit: the robust result.** Running the standard queries one at a time and answering as soon as the choice has judge P >= 0.98 cuts LoFTR
   queries on the Jetson from ~6.3 to ~2 per photo: **14 s -> ~4.5 s per photo** (blind 606: 13.9 -> 4.2 s), at the same score as today on the
   blind set (54.50 vs 54.42) and fewer wrong answers (1 vs 2). Time class of the whole photo goes from *slow* to *viable / good*; the judge itself
   adds ~0.1 s (good).
4. **Next blind set candidate:** the same rule with a minimum of 4 inliers for an answer (It6: -0.16 on official, 0 elsewhere, 0 wrong; rules out
   the 2071 failure). It cannot be scored on 606 any more.

## Notebooks (each has an `_executed*` copy)

| Notebook | What |
|---|---|
| `It1_Signal_Table` | per-candidate signals after fine matching (block `train/checks/decision_signals.py`), all sets; `ONLY=nudt606` built the blind table after the freeze |
| `It2_Candidate_Judge` | the judge: feature groups, PC and Jetson leave-one-set-out, changed photos, importance |
| `Second_Look_Run` (GPU) / `It3_Second_Look` | second LoFTR look at a candidate: lines before/after, as a judge signal |
| `It4_Decision_Rule` | cut rules (zero-wrong margins, Learn-then-Test), Jetson floor and extra-query cut, early exit, time model, **freeze** |
| `It5_Blind_606` | the frozen rules applied once to NUDT fresh 606 |
| `It6_After_Blind` | diagnosis of the blind wrong answers; minimum-inlier rule on the seen sets only |

Blocks: `train/checks/decision_signals.py`, `train/checks/candidate_judge.py`, `train/checks/second_look.py`; tests:
`tests/pipeline/test_iteration7_blocks.py`. `second_look.py` imports `src/coarse_search/heading_queries.py` from `prabhuparth/dev` (as the T6 dump does). Frozen rule: `results/frozen_rules_t7.json`, model `results/judge_t7.joblib`. `src/` is unchanged.
