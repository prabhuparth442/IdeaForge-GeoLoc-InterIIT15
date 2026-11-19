# Red-team audit of PV-X: selection, leakage, overfitting

Offline only, from the existing per-photo CSVs (runs/, native_fine/ for the bases, the scratchpad copy of the tilt-fit matches). No server job, no repo source file changed.
Scripts and raw outputs next to this file: `common.py`, `p1_gain.py` ... `p6_misc.py`, `out_p1.txt` ... `out_p6.txt`, `offline_prior_curve_points.csv`. (Other files in this folder, `analysis/` and `sensitivity_synth*`, are not from this audit.)
Scores are /60 with the repo's own rule (score.py). "Cluster" = the dataset's own spatial cell for the new set (69 dev cells, 27 test cells, no cell is in both), k-means on metres for old official (25 clusters, the photos lie in 1.7 x 1.5 km) and NUDT (15 clusters). Cluster bootstrap = resample clusters with replacement, 10 000 draws.

## 1. Verdict

The claim that PV-X is better than the base is **not an artefact**: on the new set it holds on unseen cells (locked 168: +11.32, cluster 95 % CI +5.8 to +18.2), the pooled new-set gain (500 photos, 96 cells) is +11.10 [+7.99, +14.54], and no ground truth is read inside the pipeline. What **is** inflated is the *size* of the numbers on the two older sets and the *wrong-answer protection*:

* Old official 52.69 -> 60.00 rests entirely on choices made while looking at old official (the same code at the a-priori bar 20 scored 52.08, i.e. -0.61 against base). 60.00 is the ceiling (500/500 within 1 sigma), so that set cannot show over- or under-fitting and gives no held-out information.
* Bar 12 sits at the lowest value ever run and exactly where wrong answers begin: pooled over sets, 5 of the 20 answers with 12-15 inliers are wrong; all 4 wrong answers of the new set (dev seed 0, dev seed 1 twice, test) have 12-13 inliers.
* The tuned steps after the a-priori configuration are worth +4.6 on the new set, mostly not individually significant; selection among them is stable in leave-one-cluster-out (below), but that only protects against within-region noise, not against a different region or camera.

## 2. Garden of forking paths (what was chosen, on what, how many looks)

| decision | value | chosen on | evidence in numbers | verdict |
|---|---|---|---|---|
| photo view at all | on | true-place diagnostic, 130 dev photos >= 1000 m (truth defines "right" and "wrong" places) | pipeline step base -> PV bar 20: +2.33, cluster CI [-0.13, +5.57] | concept is not truth leaked into the pipeline, but chosen on dev truth |
| pose fit | on | 125 dev photos with >= 20 inliers (survivors), candidate = truth + 30-60 m random offset (simulated, not a real coarse peak) | pipeline step +4.03 [+1.83, +6.42] | solid, a-priori priors |
| prior widths 1/1/2/2 % | audit "assumed noise" | synthetic audit tuned so that noisy metadata reproduces the real dev error vs truth (0.87 vs 0.86 sigma, CHANGELOG iteration 41) | - | truth-derived calibration, not "not tuned" |
| prior scale x0.4 | 0.4 | odd/even CV on the same 125 dev photos, 8 scales | offline curve peaks at 0.4 (8.54 vs 7.94 points/photo at x1, cluster CI of the gain [0.21, 1.07] per photo); real pipeline: +0.70 [-0.63, +2.26] | stable but small and in-sample for the 125 |
| match noise 1.5 px, robust scale 2 | grid of 3 x 3 | same 125 photos, odd/even | not re-run here | 9 more looks on the same photos |
| gate 1000 -> 800 -> 600 -> 0 (photo view) | 0 | dev, 4 runs | +0.04, +0.47 [-0.29, +1.28], +1.43 [-0.85, +4.13] (none significant); LOCO picks 0 in 69/69 cells | direction consistent, size unproven |
| XoFTR for all photos | 0 | dev + official + NUDT, after PV-X's other parts were fixed | dev +1.79 [+0.98, +2.52], official +2.76 [+1.14, +4.94], NUDT -0.20 [-1.21, +0.61] | the only late step with a significant gain on two sets; not tested separately on the locked split |
| bar | 12 | official (bar 20 -> 12 gave +4.49 [+2.55, +6.72]); NUDT 0.00 (wrong 2 -> 5); no pair on the new set | the lowest value ever run, never tried below 12 | boundary solution, evidence from one set |
| agreement rule | on | official: off -2.14 [-3.87, -0.69] vs bar12 on 56.57 | NUDT +/-0.20 | one set |
| fit max move 3 sigma, rounds 1 | 3, 1 | no evidence of tuning found; rounds 2 killed (+0.07) | the 3-sigma rule dropped 10 of 1277 answered photos (9 new set, 1 locked) | low risk |
| hint-jitter seed | 0 | every selection on seed 0; seed 1 only after the freeze | dev 50.69 -> 50.26, NUDT 52.08 -> 51.68, official identical | small seed optimism about -0.4 |
| inherited calibration | GEOMETRY scale 1.04, yaw +180, signs (dev set), NUDT scale 0.8, attitude off | truth, on the sets themselves | shared by base and PV-X | both columns, but PV draws the map with these conventions |

Looks (counted from runs/, the scratch run folders, CHANGELOG): new set 19 pipeline runs in the PV phase (b16, b20, b24, fit20, g800, g600, g0, g0p04, g600p04, r2, b12na, b16na, b12a, x0, v0, x2s0, x2s1, x3s0, s1) on top of about 19 earlier iterations scored on the same 332; old official 12 PV runs; NUDT 16; locked split: Fable's coarse rule, XoFTR-from-800 (twice, 40.61 then 40.75), PV-X = at least 3 looks, so it is a validation set only in the sense that PV-X itself was frozen before it. The freeze text cannot be verified by version control (nothing committed; FREEZE_PV-X.md mtime is after the test result).

## 3. Gain with honest intervals (PV-X seed 0 minus base, per photo, then x6)

| set | n / clusters | base | PV-X | gain | photo bootstrap 95 % | cluster bootstrap 95 % | share of photos better / worse |
|---|---|---|---|---|---|---|---|
| new set dev | 332 / 69 | 39.70 | 50.69 | +10.99 | [+8.78, +13.18] | [+7.31, +15.22] | 41 % / 5 % |
| locked test | 168 / 27 | 38.00 | 49.32 | +11.32 | [+8.00, +14.68] | [+5.79, +18.16] | 39 % / 4 % |
| new set pooled | 500 / 96 | 39.13 | 50.23 | +11.10 | - | [+7.99, +14.54] (PV-X level [47.04, 52.79]) | - |
| old official | 500 / 25 | 52.69 | 60.00 | +7.31 | [+5.63, +9.07] | [+3.74, +12.13] | 15 % / 0 % |
| NUDT | 300 / 15 | 48.52 | 52.08 | +3.56 | [+0.40, +7.04] | [+0.29, +7.18] | 5 % / 2 % |

Photo-level CIs are too narrow by 1.3-2x because neighbours are correlated. NUDT's lower bound is +0.3.

Where the gain comes from (points of the 60 scale):
* new set: skipped -> answered +4.75 (45 photos), more precise on photos both answer +6.23; locked: +5.82 and +5.50. Spread over many photos: the top 10 % of photos give 48-49 % of the net gain, the largest cluster 14-24 %; removing the best cluster leaves +10.2 (new set) and +10.5 (locked).
* by height: gain per photo-band is large from 1300 m (new set 1300-1600: 21.9 -> 50.7, locked 1600+: 13.2 -> 38.1) and small below 351 m (+5.2 dev, +2.2 locked). A different height mix changes the headline: the dev per-band means reweighted to the locked set's height mix give 39.82 -> 50.53; observed locked 38.00 -> 49.32, i.e. -1.21 below the dev prediction (cluster CI [-7.2, +4.2]): no sign of overfit, but the test cannot detect a shortfall smaller than about 6.
* old official: concentrated. 29 photos (6 %) give half of the net gain (the 46 skipped -> answered photos, +5.52); 83 % comes from the top 50.
* NUDT: concentrated. Net gain +3.56 = wrong-answer change +3.20 (12 -> 6 wrong) + precision +2.24 + answer -> skip +1.20 (3 photos) + skip -> answer +0.12; 3 photos carry half of the net gain.
* The test split is not far from the dev photos: 80 % of test photos lie within 300 m of a dev photo, median 172 m (cell-blocked, but same flight region and camera).

## 4. Does the selection survive leave-one-cluster-out?

* Gate (4 real runs on the new set): LOCO selects g0 in all 69 held-out cells; held-out 48.00 = in-sample 48.00. No optimism within this set.
* Prior scale (offline, 125 photos, 8 scales, LOCO by cell): selects 0.4 in all 49 held-out cells; held-out 8.536 = in-sample 8.536 points/photo; cluster-bootstrap argmax is 0.4 in 98 % of draws; odd/even both halves choose 0.4. The curve is sharp (x0.4 +0.59 over x1.0, x2.0 is 1.0 worse), so the choice itself is not noise. Caveat: these are 125 survivors of the matching, candidates simulated near truth. In the real pipeline on all 332 photos the effect is only +0.70 (n.s.), about half of what the offline number would predict (+1.3).
* Bar: **cannot be LOCO-selected**. Only two bar levels exist as real runs (12 and 20, and only on official and NUDT; new set only 16 vs 20 without fit). Post-hoc emulation by thresholding the inlier column is wrong (from the real bar-12 run, "bar 20" emulates to 43.32 on official versus the real 52.08, and 48.84 vs 52.28 on NUDT, because the second pass and agreement rule rescue photos), so no bar curve is reported. Leave-official-out: bar 12 vs 20 on NUDT is a tie (52.28 / 52.28) with wrong 2 -> 5, so without official the bar is not selectable.
* Ladder over the four real configs available on all three dev sets (bar 20 x1 priors, bar 12 agree-off, bar 12 agree-on LoFTR below 800, PV-X): leave-one-set-out always selects PV-X. The grid is coarse (4 points) and its lower edge (12) is the chosen value, so the test is weak.

## 5. Wrong answers and the bar

PV-X wrong answers: new set 1 of 329 (seed 0), 2 of 330 (seed 1); locked 1 of 166; NUDT 6 of 282 (seed 0), 8 of 286 (seed 1); official 0 of 500. Clopper-Pearson 95 % of the wrong rate: new set pooled 2/495 = 0.4 % [0.05, 1.45 %]; NUDT 2.1 % [0.8, 4.6 %].
* All new-set wrong answers have 12-13 inliers (dev seed 0: 12 at 1078 m; locked: 13 at 1056 m, second pass; dev seed 1: 12 and 12). NUDT seed-0 wrongs: 14, 15, 15, 18, 24, 24; seed 1 adds 5 and 6 (answers via the agreement rule below 12).
* Answers with 12-15 inliers (the band the bar admits): new set 7 answered, 2 wrong; NUDT 6, 3 wrong; official 7, 0 wrong. Pooled 5 of 20 wrong (25 %), mean points of the band negative on the new set (-4.7, -8.5).
* A bar of 16, in hindsight and in-sample: new set +0.25, locked +0.61, NUDT +0.68, official -0.84. So the new set and NUDT would have preferred a higher bar, official the lower one; the chosen bar is the one that official prefers, and it is the bar where the wrong answers appear. Whether a few 12-15 answers per 100 photos are wrong in a new area is a coin flip: expected cost of one wrong answer is 0.35 (official size) to 1.04 (locked size) of 60.
* Protection by agreement/bar is not independent of the places seen: bar and agreement settings were picked by counting wrong places the sets contain (truth defines "wrong").

## 6. Ground truth in the pipeline or in the tuning

* Inside the pipeline: none. `grep` of geoloc/, presets/, settings.py finds truth only in cli.py: scoring (error_m, sigma_m) and `move_hint` (development only, jitters the hint from truth; the real hint is separate). The pose-fit prior is centred on the metadata pose (zero), the start is the median vote; the 3-sigma drop rule uses sigma from height and focal only. PASS.
* In the tuning (not at test time, but it shapes numbers): (a) audit noise calibrated to dev truth error (0.87 vs 0.86 sigma); (b) x0.4 and the noise/robust grid chosen on the 125 dev photos scored against truth, then the same photos sit inside the 332 reported; (c) bar and "wrong place" definitions come from truth; (d) GEOMETRY scale/yaw/signs (and NUDT 0.8, attitude off) fitted to each set's truth, shared with base; (e) hint seed 0 reused for all selections; (f) offline CV candidates are truth + 30-60 m: the matches were drawn at a near-truth place, easier than a real coarse peak.
* Not checked: whether the base columns (native_fine/*.csv) were run with the same jitter seed as PV-X; seed-to-seed spread (0 to 0.4) bounds the effect.

## 7. Risks ranked

1. **Old official +7.31 and NUDT's wrong-answer story are in-sample choices** (high). Official: a-priori PV (bar 20) -0.61; bar 12 +4.49, agree on +2.81, XoFTR-all +2.76 all found on official. Expected gain on a fresh low-altitude set: +0 to +4, not +7.3; 60.00 should be read as a ceiling.
2. **Bar 12 at the edge of the wrong-answer cliff** (high for wrong-answer claims): lowest value run, 25 % of 12-15-inlier answers wrong, all new-set wrongs at 12-13. A new region with more look-alikes (NUDT-like 2 %) costs about 2-3 points per 300 photos.
3. **Concentration on official and NUDT** (medium): 6 % and 1 % of photos carry half of the net gain; NUDT cluster CI starts at +0.3.
4. **Tuned steps after the a-priori configuration are +4.6 on the new set** (medium): gate 0 (+1.9 over gate 1000), x0.4 (+0.7), XoFTR-all (+1.8); only the last is significant; LOCO-stable within region.
5. **Locked split is a third look, same region and camera** (medium): +11.32 and -1.21 below the dev prediction are consistent with no overfit but cannot exclude a 4-6 point shortfall.
6. **Seed 0 selection** (low): -0.43 (new set) and -0.40 (NUDT) on seed 1.
7. **Inherited truth-fitted geometry** (low to medium, not quantified): PV draws the map with scale 1.04, yaw +180, sign conventions found on dev; a camera or pose convention change hurts it at least as much as the base.

## 8. Expected score on unseen data (judgement from the numbers above)

| unseen data like | PV-X in-sample | expected | rationale |
|---|---|---|---|
| new set, new cells, same camera/flight | 50.23 pooled (dev 50.69, test 49.32) | **about 49 (80 % interval 45-52)**, gain about +10 over base | pooled cluster CI 47.0-52.8, minus 0.4 seed, minus about 0.8 for tuned-step optimism (half of the -1.2 test shortfall) |
| low-altitude set like old official | 60.00 | **55-58**, gain +2 to +5 | a-priori 52.08; bar/agree/XoFTR-all effects shown on one set only |
| set with look-alikes like NUDT | 52.08 | **50-52**, gain +1.5 to +3.5, 3-8 wrong | gain CI +0.3 to +7.2; wrong rate 2 % [0.8, 4.6] |
| different camera / region / pose conventions | - | unknown, lower; base would also fall | no data |

Largest numbers the claim can honestly keep: new set +11 (supported on unseen cells), NUDT +3.6 (weak), official +7.3 (in-sample). What would settle it: one fresh set of cells (or flights) run once with nothing changed, wrong answers counted separately, jitter seed not 0.
