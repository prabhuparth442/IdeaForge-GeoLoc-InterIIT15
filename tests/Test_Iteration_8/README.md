# Test_Iteration_8: honest hold-outs — what was fitted, what really gains

**Brief (team review, 2026-11-14).** The T7 judge was trained on the data it was meant to improve, so it fits that data, and the final data will
not be the same. Check every earlier method the same way: is it fitted to the data, and does it really improve the score or only answer more
with some wrong answers? Build new methods on NUDT, and split official into a TRY half and a TEST half so official is not fitted again.

Full write-up: [REPORT.md](REPORT.md). No multi-query agreement anywhere. `src/` is unchanged.

## Protocol

| Data | Photos | Role |
|---|---|---|
| NUDT 101, 202, 404, 505, 606 | 1,300 | **design**: every rule and model is built here |
| official TRY | 185 | check while building (12 places split by k-means on the true position, heights balanced; `results/official_split.csv`) |
| official TEST | 194 | **final**, other places; loaded only by It4, once |
| NUDT fresh 707 | 300 | **final**, blind: dumped (GPU) before any T8 rule, scored once |

Every finalist was written to `results/preregistered_t8.json` (01:47) before It4 loaded the final sets. Disclosure: one exploration printout
showed official TEST scores of three NUDT-only judge feature sets before the half was locked; nothing in T8 was chosen from it.

## Result: final sets, scored once

Gains are paired bootstrap 95 % against today's rule without agreement on the same machine (PC: bar 8 on the 6 standard queries, else 11 on all 27;
Jetson today: bar 8, second pass, cap 10, M = 5). `w` = wrong answers. "vs bar" = score minus the plain inlier bar that answers as many photos.

| Method | official TEST | gain | vs bar | NUDT 707 (blind) | gain | vs bar | queries | verdict |
|---|---|---|---|---|---|---|---|---|
| PC base | 48.62 / 0w | — | — | 54.04 / **3w** | — | — | 7.8-10 | — |
| **PC: T7 judge trained on NUDT only** | **53.75 / 0w** | **+5.13 [+2.97, +7.61]** | +0.31 | **55.44 / 0w** | **+1.40 [+0.20, +3.00]** | +1.40 | 8.0-8.5 | **gain** |
| PC: two-tier rule (new) | 52.39 / 0w | +3.77 [+1.11, +6.62] | +1.61 | 54.64 / 1w | +0.60 [-1.00, +2.40] | +0.60 | 8.1-9.1 | gain (weaker) |
| PC: two-tier, a_low 4 | 53.01 / 0w | +4.39 [+1.55, +7.30] | -0.43 | 53.84 / 3w | -0.20 | -0.20 | 8.0-8.7 | no |
| Jetson today | 49.24 / 0w | — | — | 54.04 / **2w** | — | — | 6.3-6.7 | — |
| **Jetson: judge, NUDT only** | **52.82 / 0w** | **+3.59 [+1.48, +6.06]** | +1.11 | **54.84 / 0w** | +0.80 [+0.00, +2.00] | +1.40 | 6.4-6.5 | **gain** |
| **Jetson: judge, NUDT only + early exit** | **52.70 / 0w** | **+3.46 [+1.24, +6.00]** | +0.99 | **54.84 / 0w** | +0.80 [+0.00, +2.00] | +1.40 | **2.0-2.8** | **gain, ~60 % less matching time** |
| Jetson: two-tier | 51.16 / 0w | +1.92 [-0.31, +4.27] | -0.56 | 53.84 / 2w | -0.20 | -0.20 | 6.3-6.5 | no |
| Jetson: two-tier + early exit (20 inliers) | 50.78 / 0w | +1.55 [-0.80, +4.02] | -0.93 | 53.84 / 2w | -0.20 | -0.20 | 2.5-3.7 | no |
| Attitude correction (fitted on TRY) on any rule | +0.12 to +0.62, intervals through 0 | | | (official camera only) | | | 0 | not proven |

Reference, final sets: a plain bar of 4 scores 55.8 / 1 wrong on official TEST but 49.0 / **15 wrong** on NUDT 707; bar 6: 53.4 / 1w and 53.8 / 4w.

## What it means

1. **The review is right about fitting.** The frozen T7 judge (trained with all official photos) scores 56.95 on official TRY; the same model
   trained on NUDT only scores 54.03. **About 3 points of T7's official numbers were the judge remembering official.**
2. **But the architecture is not the problem: trained on NUDT only, it is the best method on both unseen sets.** It gains +5.1 on official places
   it never saw and +1.4 on a fresh NUDT draw, with **0 wrong answers on both** (it removes all 3 of the bar's wrong answers on 707). The earlier
   "trained on NUDT, poor on official" result was T6's verifier (`it2_verifier_transfer.csv`: NUDT 101+202 -> official, GBM replace mode -1.46);
   the T7 candidate-level judge with a NUDT-chosen cut transfers.
3. **Where its gain comes from.** On official, answering more is nearly free (bar 4: 0 wrong on TRY, 1 on TEST), so most of the judge's official
   gain is *answering more*; it sits +0.2 (TRY) / +0.3 (TEST) above the bar that answers as many photos, i.e. it answers more **without** the
   wrong answer that bar makes. On NUDT, where low bars make 4-15 wrong answers, its gain is *judging*: +1.2 to +3 above the bar curve, all
   from removing wrong answers.
4. **The new two-tier rule** (3 numbers, no learned model) is safer than the bars on NUDT design data (0 wrong on all five) and gains on official
   TEST, but on the blind draw it keeps 1 wrong answer (PC) / 2 (Jetson): **simpler is not more robust here**; the judge's extra signals (inlier
   hull, coverage, fit plausibility) are what remove the city look-alikes.
5. **Precision is not where the points are:** today's similarity-fit position is the best estimate of every alternative tried, and the attitude
   correction (2 numbers, 1.03 deg forward, fitted on TRY) adds +0.1 to +0.6 on TEST with intervals through 0 (TEST answers were already 95.6 %
   within 1 sigma). It should matter more at 1-3 km heights, which no data here covers.

## Recommendation

- **Use the T7 judge trained on NUDT only** (never on official), cut by Learn-then-Test on NUDT (PC 0.9, Jetson 0.925), extra-query cut 0.95,
  override 12, min 3 inliers. On the Jetson **with the early exit** (P >= 0.98): +3.5 on official TEST, +0.8 on 707, 0 wrong, 2-3 queries
  per photo (~5-6 s instead of ~14 s modelled). Keep official only for checking, never for training or choosing cuts.
- Do not adopt the two-tier rule or the attitude correction on this evidence.
- Next: more official-like photos (a second official area) are the scarcest data; heights above 351 m are still untested.

## Notebooks (each has an `_executed` copy)

| Notebook | What |
|---|---|
| `Rich_Dump_Run_executed_nudt_fresh707` | GPU dump of the blind draw (`run_blind_dump.sh`) |
| `It1_Blind_Signal_Table` | signal table of the blind draw (no scores) |
| `It2_Audit` | bar curve; T7 judge with vs without official; early exit, min 4, z-veto; null-calibrated cut (dropped); position estimates |
| `It3_Two_Tier_Rule` | grid on NUDT, selection, TRY check, early-exit level, attitude cross-check, **pre-registration** |
| `It4_Final_Test` | the pre-registered finalists on official TEST and NUDT 707, once |

Blocks: `train/checks/holdout.py` (split, locked loader, bar curve, answer-change split, attitude helpers), `train/checks/two_tier.py`;
`train/checks/candidate_judge.py` now also returns the answer position (x, y). Tests: `tests/pipeline/test_iteration8_blocks.py` (3 pass; T7's 10 still pass).
