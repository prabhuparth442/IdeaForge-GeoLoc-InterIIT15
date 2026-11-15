# Decision experiments, 14 November 2026

Exploratory regression on official150 and nudt707, both previously examined. Frozen T7 judge, no retraining and no threshold sweep. Official150 overlaps judge training; neither set establishes new-flight performance. Raw matches and features use confidence 0.22, not the active notebook's 0.21. Baselines must use the same 0.22 data. No production files are edited.

Fix these policies before scoring:

1. Frozen PC judge, tau .9, wide tau .95, override 12, min choice 3: reproduction.
2. Strict judge: same geometric ranking, minimum 4 inliers, judge >= .98 in both stages, no override. Isolates conservative abstention.
3. Unique location: choose the highest judge probability among >=4-inlier candidates; require >=.98 and no other >=4-inlier candidate with probability >=.9 farther than two scoring sigmas. The two-sigma separation is twice the existing one-sigma agreement radius. Repeat detections near the winning coordinate do not count as independent probability evidence.
4. Held-out geometry: existing geometric ranking among >=4-inlier candidates, judge >=.9 at standard stage / .95 wide, no override; require reciprocal held-out spatial correspondence verification. Split raw >=.22 pairs by checkerboard spatial cells before fitting, fit on each half, require >=4 fitting inliers and >=4 withheld matches within the existing 3-pixel RANSAC tolerance in each direction; fitted canvas centres must agree within 3 pixels. Keep existing scale, angle and centre-shift gates. This is held-out correspondence evidence, not an independent image.
5. Unique location plus held-out geometry: combine policies 3 and 4, keeping .98 confidence.

All policies first inspect the six standard queries, then all 27 only on abstention. All photos stay in denominator. Record score, wrong answers, coverage, query cost, and paired-bootstrap differences. These checks share a model and data, so outcomes are development evidence; do not promote a winner as a zero-error guarantee. Retain all results, including negative ones.
