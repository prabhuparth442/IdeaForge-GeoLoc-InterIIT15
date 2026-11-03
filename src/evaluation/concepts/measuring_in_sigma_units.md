# Measuring in σ units and building the score tables

Code: `src/evaluation/run_sample.py`, `src/evaluation/report.py`, `src/header/scoring.py`.

---

## 1. Why not metres

The same 20 m error is about 2.1σ at 350 m height (σ = 9.57 m) but only 1.2σ at 1000 m (σ = 17.0 m). The score depends on the error **relative to σ** (see `header/concepts/scoring_rule.md`). So all analysis uses

$$e_\sigma=\frac{\text{error in metres}}{\sigma_{\text{photo}}},\qquad \sigma=\sqrt{8^2+(3\cdot\mathrm{GSD})^2},\ \ \mathrm{GSD}=h/f.$$

`σ` grows with height (8.14 m at 100 m, 9.57 m at 350 m, 17.0 m at 1000 m), so a height-independent standard (e.g. "within 15 m") would treat the heights unfairly.

## 2. From error to points

`header/scoring.py::points_for_error` maps `e_σ` to the tier, and `final_score(points)` is `60·max(Σ/(10N), 0)`: the sum of points divided by the maximum possible, scaled to 60 and floored at 0. Skipped photos carry 0 points. Tier boundaries are the ones in the problem statement; the module's `check_scoring()` re-derives the worked example (28.8) as a self-test.

## 3. What `run_sample.py` does

* `run_correlation(testset, store, variants, dino=None)`: for each sampled photo renders the canvas, runs `search_frame` against the simulated prior and keeps the peaks. Returns `(disc, seconds)`: `disc[i] = {"peaks": {variant: (25,6) array}, "cutoff": ...}` and the time per photo.
* `run_minima(testset, store, matcher, disc)`: for each photo runs `second_stage` on the stored peaks and `final_answer`. Returns `(stage2, final, seconds_per_photo)`.

Splitting the two lets you re-run the slow fine stage without redoing the search. Neither calls `localise_photo`; that function is the single-photo version used by `independence_check.py` (see `result_thresholding/concepts/one_photo_at_a_time.md`).

## 4. The ceiling and the diagnostics

`report.py::print_results` prints, for the photos overall / low (≤200 m) / high (the `a/l/h` columns): share within 1σ and 4σ for each answer type, the **ceiling** (any of the 25 candidates within 8σ of the truth, which the fine stage cannot exceed), and the AUC of the correlation ratio and of the inlier count.
`print_final_scores` prints: the share within 4/1/2σ and worse than 8σ, the median error in σ, scores for every photo answered (overall, low, high), the confident-only score (optimistic and honest split-half), the point-tier counts (+10 / +6 / +3 / 0 / −20), a table over the number of candidates K = 1, 3, 5, 25 (accuracy, score, ceiling), and a table of fixed rules "answer if inliers ≥ T" for T = 0 … 40 with the number of −20 answers.
Compared answer types: `correlation`, `+MINIMA re-rank`, `+MINIMA re-rank + refine`.

## 5. Reading the tables


Read tables with the sample size in mind: with about 20 photos per row, one photo moves a percentage by 5 points. Prefer the AUC and the overall number for decisions.
