# New evidence on the decision layer and precision: think it through for us

You wrote Patch v0.5.5 (the breakthrough candidates) and the experiments report. Since then we ran the full single-image pipeline (fused coarse, DINO semantic layer, K=50 to M=25, fine stage, decision) and logged per-photo results. The numbers below change the priorities in v0.5.5. I want your thinking, not code: what the evidence means, what is theoretically worth doing, what it should improve and by how much, and in what order.

Attached: three per-photo CSVs (100 photos each): official set with LightGlue, official set with MINIMA (same 100 photos), and NUDT with MINIMA. Columns: `photo, height, level, sigma_m, error_m, error_sigma, answered, inliers, chosen, z_chosen, points, t_*`. `error_sigma` is the error of the chosen candidate even when the photo was skipped; for 0-inlier photos it appears to be the coarse peak's error (identical across matchers). Please verify that reading against the decision code in the repo if you can.

## 1. What the run says

**The DINO layer did not move the score.** Coarse only / with semantic layer / layer + DINO decision all end at 0.78 answered and 41.49 out of 60, although coarse top-1 went from 0.905 to 0.960. Confirms the objective-mismatch hypothesis from your report (the fine stage tests all 25 candidates independently and picks by inliers, so order never matters).

**Where the points go** (per 100 photos, max 1000):

| file | points | answered | <=1σ | <=2σ | wrong (-20) | lost to skipped-but-answerable | lost to imprecise (2σ not 1σ) | lost to wrong answers |
|---|---|---|---|---|---|---|---|---|
| official + LightGlue | 656 | 75 | 58 | 16 | 1 | 148 | 64 | 20 |
| official + MINIMA | 710 | 79 | 59 | 20 | 0 | 155 | 80 | 0 |
| NUDT + MINIMA | 716 | 75 | 73 | 1 | 1 | 59 | 4 | 20 |

So the largest loss is **skipping photos we would have answered correctly**, about twice the precision loss on the official set. A perfect answer/skip gate would reach 824 / 865 / 795. This reverses the v0.5.5 ordering (precision gap bigger than recall gap): the decision gap is now the biggest.

**The 12-inlier rule cuts good answers.** Official + MINIMA, skipped photos with 5-11 inliers (14 photos): errors 0.37, 0.62, 0.21, 0.22, 0.73, 0.48, 2.37, 0.53, 0.45, 0.64, 0.36, 0.56, 1.00, 0.86 sigma, i.e. all points-earning, 12 of 14 within 1σ. All four wrong skips there had 0-3 inliers. But the same inlier count is not equally trustworthy across matchers: with LightGlue, 3-6 inliers includes three -20 answers (photos 21, 88, 97). Threshold sweep (points, LightGlue / MINIMA / NUDT):

| answer if inliers >= | LightGlue | MINIMA | NUDT |
|---|---|---|---|
| 12 (now) | 656 | 710 | 716 |
| 8 | 662 | 770 | 728 |
| 6 | 642 | 823 | 738 |
| 5 | 622 | 843 | 738 |

**The coarse z (`z_chosen`) rescues many 0-5 inlier photos.** Among photos with under 4 inliers, every photo with z >= 2.9 earned 0 or more (nearest -20 is NUDT photo 17 at z = 2.76); below about z = 1.2 every one was wrong. A rule "answer if inliers >= 6, or coarse z >= 3.0 (use the coarse position)" gave 721 / 849 / 747 against 656 / 710 / 716 (sum +235 over 300 photos), with 0-2 wrong answers per file. Thresholds are in-sample on 100 photos each.

**Two wrong answers get through today:** NUDT photo 34 (21 inliers, 452σ, z 1.46) and official LightGlue photo 74 (14 inliers, 256σ, z 0.96, 350 m).

**Repeated near-misses at 19-21σ** (official photos 38 and 82, same error under both matchers; NUDT photo 1 similar) all have z of 2.1-2.2: the coarse peak sits a fixed ~150 m from the truth. Cause unknown (look-alike block, wrong yaw/scale hypothesis, map offset?).

**Altitude (official set, answered fraction / median inliers / median error σ, LightGlue | MINIMA):**
<=125 m: 0.59/24/0.42 | 0.68/29.5/0.40; 125-200 m: 0.80/36/0.56 | 0.65/21.5/0.54; 200-300 m: 1.00/138/0.94 | 1.00/129/0.92; 300-400 m: 0.70/61/1.19 | 0.87/55/0.88. Low altitude has the lowest answered rate but the best precision; 200-300 m is always answered but sits at the 1σ boundary. On NUDT, 200-300 m is only 40 % answered (median 4 inliers, 15 photos) while 300-400 m is 81 %.

**A ~1σ systematic floor on the official set only.** Among answered photos with >= 40 inliers, median error is 0.76σ (LightGlue) and 0.86σ (MINIMA) and 26-35 % exceed 1σ; on NUDT the median is 0.45σ and 0 % exceed 1σ. More inliers do not help (100-220 inlier answers still sit at 0.9-1.0σ). It looks like bias (our official-set heading offset of 180° and scale ×1.04 found by trial, the nadir/tilt conversion, or a map offset), not noise. The CSVs only hold unsigned error, so I cannot separate these.

**Time:** 1.3-2.1 s mean per photo of the 30 s limit (development GPU), so there is room for rescue passes on the grey-zone photos.

**Caveats:** 100 photos per file; every threshold above is in-sample; no data above 350 m; final evaluation is panchromatic 11-bit.

## 2. What I want you to think through

1. **Verdict on my reading.** Where is the analysis above wrong, over-fitted, or confounded (matcher, altitude, dataset, the 0-inlier error assumption)?
2. **The decision rule.** What is the right theoretical form of the answer/skip rule given the +10/+6/+3/0/-20 payoffs? Break-even is P(wrong) < m/(m+20) (about 23-29 %). Should it be a calibrated expected-points model on fine-stage features (your R1/R2), and which features carry the signal? Consider inlier count versus ratio, inlier spread, residual, scale/yaw versus the IMU prior, margin to the second-best candidate, number of candidates with real support, coarse z, DINO scores, agreement among top candidates. Which are worth logging, and what is the cleanest way to handle the matcher difference (MINIMA vs LightGlue counts)?
3. **Low-inlier photos.** Rank the options by expected gain: rescue re-match (more keypoints, second scale, yaw ± a few degrees, margin > 0) with agreement of two weak matches; coarse-only answer when z is high; consensus of the top 2-3 candidates; dense matcher on the grey zone. What does each theoretically recover, and what does each risk?
4. **Blocking confident wrong answers** (photos 34 and 74): what independent check would catch them (pose plausibility, DINO agreement, margin, MI)?
5. **Precision and the 1σ floor.** Re-rank P1, P2, P3, P4 in light of this: how much of the official-set floor can they remove, at 100-350 m versus 1-3 km? How do we tell scale/yaw/tilt/nadir-conversion bias from a map offset, and exactly what signed quantities should we log to do it? Is the nadir-versus-image-centre conversion a plausible contributor?
6. **Anything new** beyond v0.5.5 that could move many 2σ answers to 1σ or turn skipped photos into answers within the Jetson budget. Do not repeat items already in v0.5.5 unless the new evidence changes their value.

## 3. Output

- A re-ranked plan: for each item, the theoretical reason, the expected gain (points per photo and out of 60) with an honest uncertainty, cost, and risk.
- The exact extra columns to log in the next run so these questions become answerable.
- A validation design that avoids fooling ourselves: paired bootstrap, fit on one dataset and test on the other, split-half on the official set, per-altitude reporting, and how to emulate altitudes above 350 m.
- Clearly label what is measured, derived, and guessed.

No code needed. Be direct; say where you are unsure.
