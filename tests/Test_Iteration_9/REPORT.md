# Test_Iteration_9 — Full Report: red team of the decision

Branch `palash/dev`. `src/` unchanged. Question from the team: *rigorous checking, like a red team, that the current method is invariant to
the input drone images; can you confirm this?* **It cannot be confirmed, and the evidence says it is not invariant.** This report lists what
was attacked, what breaks, and how badly.

## 1. What is under test

The T8 recommendation: the T7 candidate judge (monotone gradient boosting over 19 signals after fine matching) **trained on NUDT 101-606
only**, cut 0.9 (PC) / 0.925 (Jetson) from Learn-then-Test on NUDT, extra-query cut 0.95, override at 12 inliers, min 3 inliers; Jetson with
early exit at P >= 0.98. Baselines: today's inlier bars without agreement. Official (379) and NUDT 707 are outside the judge's training data
(both were scored in T8, so they are seen data now: this iteration attacks, it does not claim a gain).

## 2. Offline attacks (It1, It2)

| Attack | Finding |
|---|---|
| Can a classifier tell NUDT candidates from official ones? | yes, AUC 0.988. Largest shifts: coarse ZNCC score (0.83), inlier ratio, spread, sharpness, DINO z. The judge extrapolates on any new map. |
| Is P(right) the same on new data? | no. Wrong rate among candidates at P 0.9-0.95: 15 % NUDT (out-of-fold), 12 % NUDT 707, 1 % official; at P 0.8-0.9: 43 % on NUDT 707. Only P >= 0.98 is near 0 everywhere. |
| Height bands (official 100-351 m) | 0 wrong in every band; gains in all but 175-275 m. Nothing above 351 m exists to test. |
| Margin of the cut | NUDT 707: a cut of 0.8 already makes 1 wrong answer (0.85+ clean). Official: clean down to 0.5. |
| One signal shifted by +-1 SD | 14 of 76 shifts create wrong answers on NUDT 707 (inliers +1 SD: 3 wrong; hull +1 SD: 3; DINO top-place gap: 1-2); none on official. |
| Hardening by feature choice (drop coarse/sharp; monotone gaps; relative-only) | does not remove the failures (relative-only is worse: up to 8 wrong). |
| Every candidate's inliers x1.25 / x1.5 / x2 | judge with override: 2-3 / 2-5 / 6-11 wrong on NUDT 707; **today's bars 5-6 / 7-9 / 14-18**; judge without override 1-2 / 1-2 / 1-3; **judge without override at cut 0.98: 0 / 0 / 0** (clean official 51.8 vs 53.9). The override at 12 inliers causes most of the judge's failures; removing it costs nothing on clean data. |

## 3. Real pipeline runs on altered photos (It3)

Block `train/checks/perturb.py` writes altered copies of a photo (the pipeline reads an absolute path, so `src/` is untouched) or alters its
metadata; the T6 rich-dump pipeline runs unchanged (27 queries, K = 50, M = 10, raw matches stored). 60 official photos (both halves, heights
balanced) + 60 NUDT 707 photos x 7 alterations = 840 runs, 0 failures (~10 s per photo on the RTX 4070 laptop). Smoke check: polarity flip
51 -> 37 inliers on one photo, heading +15 deg 51 -> 3.

Alterations: polarity flip (hot <-> cold), Gaussian blur 2 px, Gaussian noise 4 % of the 1-99 % range, half resolution (down and up), tone
curve x^0.5, heading metadata +15 deg, height metadata +20 %. Truth and sigma are never altered.

Results: see the table in [README.md](README.md) (`results/it3_altered.csv`, per photo `it3_altered_photos.csv`). Totals over the 14 altered runs:

| Rule | wrong answers | mean score | clean score |
|---|---|---|---|
| PC bar 8 / 11 | 12 | 40.7 | 49.2 |
| PC judge 0.9 + override (T8) | 4 | 45.4 | 53.2 |
| PC judge 0.9, no override | 3 | 43.9 | 53.2 |
| PC judge 0.98, no override | 1 | 40.3 | 51.2 |
| PC V3 scale augmentation (T10) | 2 | 43.4 | 54.2 |
| PC V5 relative + augmentation (T10) | 3 | 42.9 | 54.2 |
| Jetson today | 8 | 38.9 | 50.2 |
| Jetson judge + override + early exit (T8) | 1 | 41.8 | 52.6 |
| Jetson judge, no override, early exit | 1 | 41.7 | 52.6 |
| **Jetson judge 0.98, no override, early exit** | **0** | 40.2 | 51.1 |
| Jetson V5 relative + augmentation | 2 | 42.1 | 54.2 |

## 4. Conclusions

1. **Not input-invariant.** Wrong height metadata (+20 %), blur, a heading error beyond +-10 deg and a polarity flip each cost 9-30 points
   for every rule; the inlier-count signals carry the camera / matcher / map scale; P(right) is calibrated only near 1.
2. **The judge fails more gracefully than the bars:** 4 vs 12 wrong (PC), 1 vs 8 (Jetson) under the same alterations, at higher scores.
3. **Remove the 12-inlier override** (no cost on clean data, fewer wrong answers under inflation and alterations).
4. A **strict cut (0.98, no override)** is the safest rule found (Jetson: 0 wrong in every run) at -3 points on clean official; worth
   pre-registering against the plain cut on the fresh NUDT 808 draw (T10).
5. The biggest input risks are upstream of the decision: **height and heading metadata** outside the query grid (scale 0.95-1.05, heading
   +-10 deg), and **blur / resolution** (the real evaluation reaches 3000 m, where resolution per metre is far lower). Wider query grids, a
   height-scale search and training / checking on blurred or high-altitude photos are the levers, not the judge.

## 5. Problems hit

- RAM: 12 parallel sweep processes (~1.4 GB each) next to a GPU dump ran the 14 GB laptop out of memory twice; the kernel killed the Claude app
  and every job. Jobs now run as `systemd-run --user` services with a memory cap (an over-limit job is killed inside its own cap). It3 itself
  needed float32 tables and freed copies to fit in 6 GB.
- A stuck system crash reporter (apport) used one CPU core for 10 h; stopped by the user.
- Earlier hard freeze (01:48) after CPU machine-check errors (known CPU 10/11 fault).
