# AUC: how well does a confidence number separate right from wrong?

**Code:** [`../metrics.py`](../metrics.py) (`average_ranks`, `auc`, `gated_curve`).
**You need first:** nothing beyond the idea that each photo ends up "right" or "wrong".

---

## 1. The question

For every photo we have a number that is supposed to tell us how much to trust the answer: the number of inliers, the ratio of the best to the second-best correlation peak, and so on. We also know afterwards whether the answer was right (within tolerance) or wrong. A useful confidence number is **higher for right answers than for wrong ones**. AUC (area under the ROC curve) measures exactly that, with no threshold needed.

## 2. Definition by pairs: a coin-toss picture

Take one random right photo `R` and one random wrong photo `W`. Look at their confidence numbers `s_R` and `s_W`.

$$\mathrm{AUC}=P(s_R>s_W)+\tfrac12\,P(s_R=s_W).$$

In words: *the chance that the right one has the higher number, counting ties as half.*

* **0.5** means no information: the number is as good as a coin toss;
* **1.0** means perfect separation: every right photo scores higher than every wrong one;
* **below 0.5** means inverted: the number is higher for wrong answers (usually a bug or a flipped sign).

### A worked example by pairs

Four photos with confidence `[1, 2, 3, 4]`; the second and fourth were right, i.e. `[False, True, False, True]`. List every (right, wrong) pair:

| right (value) | wrong (value) | right wins? |
|---|---|---|
| 2 | 1 | yes |
| 2 | 3 | no |
| 4 | 1 | yes |
| 4 | 3 | yes |

Three out of four, so AUC = **0.75**.

## 3. The rank shortcut

Counting all `n_R × n_W` pairs gets slow with many photos. There is an equivalent formula from ranks (the Mann–Whitney U statistic). Sort all the values and give rank 1 to the smallest, rank 2 to the next, and so on. Let `S_R` be the sum of the ranks of the right photos. Then

$$U=S_R-\frac{n_R(n_R+1)}{2},\qquad \mathrm{AUC}=\frac{U}{n_R\,n_W}.$$

Why it works: a right photo with rank `k` has `k − 1` photos below it. Some of those are other right photos, the rest are wrong photos it "beats". Adding up over all right photos, the number of right-beats-right pairs counted this way is always `n_R(n_R − 1)/2`, which the formula subtracts, leaving only right-beats-wrong pairs.

On the example: ranks are 1, 2, 3, 4; `S_R = 2 + 4 = 6`; `n_R = n_W = 2`; `U = 6 − 3 = 3`; AUC = 3 / 4 = 0.75. ✓

## 4. Ties: the part that is easy to get wrong

Inlier counts are integers with **lots of ties** (many photos have 0 or a small number). The definition says ties count as half. In the rank formula that means tied values must all get the **average** of the ranks they share.

Example: values `[0, 0, 0, 5]`, right = `[False, False, True, True]`. By pairs: the right photo with 0 ties with both wrong photos (half a point each) and the right photo with 5 beats both:

$$\frac{0.5+0.5+1+1}{4}=0.75.$$

With ranks, the three zeros share ranks 1, 2, 3, so each gets the average 2. Then the ranks are `[2, 2, 2, 4]`, `S_R = 2 + 4 = 6`, `U = 6 − 3 = 3`, AUC = 0.75. ✓

An earlier version of this code used `values.argsort().argsort() + 1`, which hands out *distinct* ranks to tied values in an arbitrary order. On this example it returned **1.0**: a perfect score from a signal that is wrong half the time at the low end. The repo now uses `average_ranks`:

```python
def average_ranks(values):
    order = values.argsort(kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks
```

Read it as: sort; walk along the sorted list; whenever a run of equal values from position `start` to `end` is found, give each of them the rank `(start + end)/2 + 1` (the average of the 1-based positions `start+1 … end+1`); continue after the run. For `[3, 1, 3, 2]` the result is `[3.5, 1, 3.5, 2]`. It was checked against the brute-force pair counting on random tied data to 9 decimals.

If either class is empty (all right or all wrong) the function returns `nan` because there are no pairs.

## 5. Reading an AUC from the report

| AUC | meaning |
|---|---|
| 0.5 | useless |
| 0.6 to 0.7 | weak |
| 0.8 to 0.9 | useful |
| above 0.95 | very strong |

The report prints the AUC of the **correlation ratio** (coarse stage) and of the **inlier count** (fine stage). The inlier count is the signal the decision uses, and it should be clearly higher.

## 6. What AUC does not tell you

* **Calibration.** A signal with AUC 0.95 may still have no threshold that clears the break-even of about 70 to 77 % precision. AUC says "the order is good", not "the values are good".
* **Imbalance.** AUC does not change if there are 10 times more right than wrong photos, but the best threshold does.
* **Sample size.** With 20 photos, one photo can move the AUC by several points.

## 7. From AUC to action: gated curves

`gated_curve(signal, points)` answers the practical question. For 40 thresholds at the quantiles 0 % to 95 % of the signal, it answers only the photos whose signal is at least that high and computes the score (skipped photos give 0 but stay in `N`). It returns the answered share and the score at each threshold.

```python
thresholds = np.quantile(signal, np.linspace(0, 0.95, 40))
...
answered = signal >= t
scores.append(final_score(list(np.where(answered, points, 0))))
```

The first threshold (quantile 0) answers everything; the last answers about 5 %. The maximum of the curve is what `best_gate` reports. It is **optimistic**, because the threshold is tuned on the same photos that are scored; see [`honest_threshold_selection.md`](honest_threshold_selection.md).

## 8. Common mistakes

* Using an AUC with ties computed by plain `argsort` (distinct ranks).
* Reading AUC as an accuracy.
* Taking the best point of the gated curve as the expected score on new photos.

## 9. Check yourself

1. Values `[1, 1, 2, 2]`, right = `[F, T, F, T]`. What is the AUC? (Pairs: right 1 vs wrong 1 = tie 0.5, right 1 vs wrong 2 = lose 0, right 2 vs wrong 1 = win 1, right 2 vs wrong 2 = tie 0.5 → 2/4 = 0.5.)
2. Why would an AUC of 0.2 be useful information?
3. What is the share answered at the last threshold of the gated curve, roughly?
