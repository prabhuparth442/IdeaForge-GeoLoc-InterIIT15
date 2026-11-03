# AUC: how well does a confidence number separate right from wrong?

Code: `auc` in `src/evaluation/metrics.py`.

---

## 1. The question

We have a score per photo (inliers, correlation ratio...) and a label (the answer was within tolerance or not). A good score is **higher for right answers than for wrong ones**. AUC (area under the ROC curve) measures exactly this, independent of any threshold.

## 2. Definition by pairs

Take a random right photo `R` and a random wrong photo `W`:

$$\mathrm{AUC}=P(s_R>s_W)+\tfrac12P(s_R=s_W).$$

* 0.5 = no information (coin toss), 1.0 = perfect separation, below 0.5 = inverted.

## 3. Rank formula (Mann–Whitney U)

Counting all `n_R · n_W` pairs is slow; use ranks. Sort all values, give rank 1 to the smallest. The sum of ranks of the right ones `S_R` satisfies: each right item's rank = (number of items below it) + 1. The number of pairs it wins is its rank − 1 minus the number of *right* items below it. Summing and using `Σ(over right items) (right below) = n_R(n_R−1)/2`:

$$U=S_R-\frac{n_R(n_R+1)}{2},\qquad\mathrm{AUC}=\frac{U}{n_Rn_W}.$$

Code:

```python
ranks = values.argsort().argsort() + 1
return (ranks[is_right].sum() - n_right * (n_right + 1) / 2) / (n_right * n_wrong)
```

`argsort().argsort()` turns values into ranks (the first `argsort` gives the order, the second inverts it). Ties are given arbitrary distinct ranks here; the correct tie handling averages the ranks. For integer inlier counts with many ties (lots of zeros) this makes the value slightly noisy; use `scipy.stats.rankdata` if exactness matters. If either class is empty the function returns `nan`.

## 4. Worked example

values `[1, 2, 3, 4]`, right = `[F, T, F, T]`. Ranks 1,2,3,4; `S_R = 2+4 = 6`; `n_R = 2`, `n_W = 2`; `U = 6 − 3 = 3`; AUC = 3/4. Pairs: (R=2 vs W=1) win, (2 vs 3) lose, (4 vs 1) win, (4 vs 3) win → 3/4. ✓.

## 5. What AUC does not tell you

* Calibration: a score can have AUC 0.95 and still have no threshold that clears the break-even (see `result_thresholding/concepts/answer_or_skip_expected_value.md`).
* Class imbalance does not change AUC, but does change the best threshold.

## 6. Gated curves

`gated_curve(signal, points)` answers a more practical question: for thresholds at 40 quantiles of the signal (0 to 95 %), what share of photos is answered and what is the score? The best point of that curve (`best_gate`) is the highest score achievable by gating on that signal, if the threshold could be tuned on the very same data (optimistic: see the next file).
