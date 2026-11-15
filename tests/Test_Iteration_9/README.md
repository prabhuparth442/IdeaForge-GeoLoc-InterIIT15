# Test_Iteration_9: red team — is the decision independent of the input photos?

**Answer: no.** The recommended decision (the T7 judge trained on NUDT only, T8) is safe and useful on photos like the dev photos, but its
score collapses and some wrong answers appear when the input changes in ways a new drone, camera, palette or bad metadata could bring.
It is still **much safer than today's inlier bars** under the same changes. Full write-up: [REPORT.md](REPORT.md).

## Real pipeline runs on altered photos (`It3_Altered_Photos`)

60 official photos (both halves, all heights) and 60 NUDT 707 photos, each run through the unchanged pipeline once clean and once per alteration
(block `train/checks/perturb.py`, dumps by `Rich_Dump_Perturbed` / `run_perturbed_dumps.sh`; 840 altered photo-runs, 0 failures).
Score /60 and **wrong answers**; judges trained on NUDT 101-606 only; Jetson rules with the T10 context fix (5 matched candidates).

| Rule | clean | polarity flip | blur 2 px | noise 4 % | half resolution | tone curve | heading +15 deg | height +20 % | **wrong, 14 altered runs** |
|---|---|---|---|---|---|---|---|---|---|
| PC bar 8 / 11 (today, no agreement) | 48.8 / 49.6 | 35.4 / 41.6 | 19.6 / 49.6 | 50.2 / 49.2 | 38.6 / 49.6 | 50.2 / 52.2 | 37.2 / 39.9 | 31.8 / 24.2 | **12** |
| PC judge 0.9 + override 12 (T8 recommendation) | 54.8 / 51.6 | 45.5 / 44.2 | 32.0 / 52.2 | 54.2 / 51.8 | 46.6 / 52.6 | 56.8 / 49.6 | 37.2 / 39.9 | 36.4 / 36.6 | **4** |
| PC judge 0.9, no override | 54.8 / 51.6 | 45.5 / 42.2 | 32.0 / 51.2 | 54.2 / 50.8 | 46.6 / 51.6 | 55.8 / 47.6 | 34.2 / 37.9 | 34.4 / 30.6 | **3** |
| PC judge 0.98, no override | 50.8 / 51.6 | 37.4 / 42.6 | 27.2 / 49.2 | 49.2 / 48.8 | 42.6 / 49.6 | 49.2 / 45.6 | 29.2 / 35.9 | 32.8 / 24.6 | **1** |
| Jetson today | 48.8 / 51.6 | 35.4 / 43.6 | 23.6 / 51.6 | 51.2 / 52.8 | 38.2 / 51.6 | 50.2 / 52.0 | 33.8 / 35.9 | 25.1 / -0.8 | **8** |
| Jetson judge 0.925 + override + early exit (T8) | 53.6 / 51.6 | 42.5 / 46.8 | 29.5 / 52.0 | 52.0 / 51.8 | 47.2 / 51.6 | 51.2 / 52.0 | 37.8 / 38.9 | 24.5 / 7.6 | **1** |
| **Jetson judge 0.98, no override, early exit** | 50.6 / 51.6 | 36.4 / 46.8 | 26.9 / 52.0 | 48.4 / 51.8 | 40.2 / 51.6 | 50.2 / 50.0 | 36.8 / 38.9 | 24.2 / 8.6 | **0** |

Each cell: official / NUDT 707 (60 photos each). Not shown: the T10 variants V3 (scale augmentation) and V5 (relative inliers + augmentation):
2-3 wrong on the PC, 2 on the Jetson — no better than the plain judge on real alterations (`results/it3_altered.csv`).

**What breaks it:**
1. **Wrong height metadata (+20 %)**: the worst case. Scores halve; NUDT 707 falls to -0.8 (Jetson today, 5 wrong) and 6 wrong for the PC bars;
   the judge keeps 0-1 wrong but scores 8-37. The queries cover scale 0.95-1.05 only.
2. **Blur** (official): -20 to -29 points for every rule; official matches depend on fine thermal edges.
3. **Heading off by 15 deg** (outside the +-10 deg queries): -12 to -17 points, wrong answers for the bars.
4. **Polarity flip** (hot <-> cold): -9 to -13 points, 1 wrong for several rules.
5. **Inlier scale** (offline, It2): if every candidate gets more inliers (sharper camera / looser matcher), the 12-inlier override answers
   look-alikes; without it, and at cut 0.98, 0 wrong up to x2 inflation.
6. Offline (It1): NUDT and official candidates are almost perfectly separable (AUC 0.988), P(right) is not calibrated across sets (wrong rate
   at P 0.9-0.95: 15 % NUDT, 1 % official), and on NUDT 707 a cut of 0.8 already makes a wrong answer.

**Safest rule found:** Jetson judge, cut 0.98, no override, early exit: **0 wrong in all 16 runs** (clean + 14 altered + clean), at -3.0
points on clean official (50.6 vs 53.6, 60 photos) and 0 on clean NUDT 707 vs the T8 rule. Not a final claim (official and NUDT 707 are seen data); T10 tests it on the fresh NUDT 808 draw.

## Notebooks

| Notebook | What |
|---|---|
| `It1_Offline_Red_Team` | input shift (AUC), calibration by P band, height bands, cut margin, one-feature +-1 SD shifts |
| `It2_Hardening` | feature-hardening attempts; inlier inflation x1.25-x3 for the judge, the bars and the judge without override |
| `Rich_Dump_Perturbed` (+ `run_perturbed_dumps.sh`) | GPU runs of the altered photos |
| `It3_Altered_Photos` | every rule on the altered photos |
