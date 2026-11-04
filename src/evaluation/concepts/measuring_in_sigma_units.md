# Measuring in σ units and reading the score tables

**Code:** [`../run_sample.py`](../run_sample.py), [`../report.py`](../report.py), `src/header/scoring.py`.
**You need first:** `header/concepts/scoring_rule.md` (what σ is and what the tiers are).

---

## 1. Why we never judge an error in plain metres

The same 20 m error is a very different result at different heights, because the allowed error grows with height. The scoring rule measures every error in units of

$$\sigma=\sqrt{8^2+(3\cdot\mathrm{GSD})^2},\qquad \mathrm{GSD}=\frac{h}{f}\ \ (\text{ground metres per pixel}).$$

With a 200-pixel focal length:

| height | GSD (m/pixel) | σ |
|---|---|---|
| 100 m | 0.5 | 8.14 m |
| 200 m | 1.0 | 8.54 m |
| 350 m | 1.75 | 9.57 m |
| 1000 m | 5.0 | 17.0 m |

(`sigma_m` also uses a minimum height of 100 m, so lower photos get the 100 m value.)

A 20 m error is therefore 2.09σ at 350 m (points: +3) but only 1.18σ at 1000 m (points: +6). Everything in the analysis is expressed as

$$e_\sigma=\frac{\text{error in metres}}{\sigma_{\text{this photo}}}.$$

## 2. From error to points

`points_for_error(error_m, altitude, focal)` computes the ratio and applies the tier table:

| `e_σ` | points |
|---|---|
| ≤ 1 | +10 |
| ≤ 2 | +6 |
| ≤ 4 | +3 |
| ≤ 8 | 0 |
| > 8 | −20 |

Examples with real numbers from the code, at 100 m height (σ = 8.14 m):

| error | 5 m | 10 m | 20 m | 40 m | 80 m |
|---|---|---|---|---|---|
| `e_σ` | 0.61 | 1.23 | 2.46 | 4.91 | 9.83 |
| points | +10 | +6 | +3 | 0 | −20 |

The final score is `final_score(points) = 60 × max(Σ points / (10·N), 0)`: the sum of points divided by the best possible total, times 60, never below 0. **Skipped photos count in `N` with 0 points.** `check_scoring()` in `header/scoring.py` reproduces the problem statement's worked example (28.8) as a self-test.

## 3. What `run_sample.py` does

The evaluation is split into two functions so that the slow fine stage can be re-run without redoing the search:

* `run_correlation(testset, store, variants, dino=None)`: for each sampled photo, render the canvas, run `search_frame` against the photo's simulated prior, keep the 25 peaks. Returns `(disc, seconds)` where `disc[i] = {"peaks": {variant: (25,6) array}, "cutoff": …}`.
* `run_minima(testset, store, matcher, disc)`: for each photo run `second_stage` on the stored peaks and `final_answer`. Returns `(stage2, final, seconds_per_photo)` with `final[i] = (x, y, inliers, chosen)`.

Neither calls `localise_photo`; that is the single-photo version used for the independence check (see `result_thresholding/concepts/one_photo_at_a_time.md`).

## 4. Three answers compared, side by side

`collect_answers` builds three answer types for every photo from the same stored data:

| name | meaning |
|---|---|
| `correlation` | the top correlation peak only |
| `+MINIMA re-rank` | the candidate with most inliers (if at least 12), at its coarse position |
| `+MINIMA re-rank + refine` | the same candidate plus its RANSAC shift (the real pipeline's answer) |

and a few diagnostics: `closest` (the distance from the truth to the nearest of the 25 candidates), `ratios` (coarse ratio) and `inlier_counts`.

## 5. Reading `print_results`

Columns are labelled **a / l / h**: all photos, low photos (height ≤ 200 m), high photos.

* `<= 1 sigma`, `<= 4 sigma`: share of photos with that accuracy for each answer type;
* `always`: the score (out of 60) if **every** photo is answered;
* **ceiling**: the share of photos where *any* of the 25 candidates is within 8σ of the truth. The fine stage can only pick among candidates, so it cannot beat this number. A low ceiling means the coarse search is the bottleneck; a high ceiling with a low final accuracy means the selection is the bottleneck;
* **AUC** of the correlation ratio and of the inlier count: how well each signal separates right from wrong answers (see [`auc_and_confidence_signals.md`](auc_and_confidence_signals.md)).

## 6. Reading `print_final_scores`

It prints one column per answer type, with these rows:

* share within 4σ (all, low, high), within 1σ, within 2σ, and worse than 8σ;
* the **median error in σ**;
* the score if every photo is answered (overall, low only, high only);
* the score with a confidence gate: **optimistic** (the best threshold chosen on the same photos) and **honest** (split-half, see [`honest_threshold_selection.md`](honest_threshold_selection.md)), each with the share of photos answered;
* the **point tiers**: how many photos earned +10, +6, +3, 0, −20;
* a table over **K**, the number of candidates considered (1, 3, 5, 25): accuracy, score and ceiling. It tells you whether checking more candidates is worth the time;
* a table of **fixed rules** "answer if inliers ≥ T" for T = 0 … 40 with the number of −20 answers.

## 7. Reading these tables with the sample size in mind

A run uses about 12 to 60 photos per table (the stratified subset). With 20 photos in a row, one photo moves a percentage by 5 points. Use the overall numbers and the AUC for decisions, and treat differences below a few photos as noise.

## 8. Common mistakes

* Judging in metres across heights.
* Dropping skipped photos from `N`.
* Comparing a gated score (optimistic) with an ungated one.
* Over-reading the low/high split with a handful of photos each.

## 9. Check yourself

1. At 350 m height, which error in metres is exactly 1σ? (9.57 m.)
2. A configuration has ceiling 70 % and final accuracy 40 %. Where is the bottleneck?
3. 30 photos: 20 earn +10, 4 earn +6, 3 earn 0, 3 earn −20. What is the score? (`(200 + 24 − 60) / 300 × 60 = 32.8`.)
