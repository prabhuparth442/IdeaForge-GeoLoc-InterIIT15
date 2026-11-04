# Honest threshold selection (split-half)

**Code:** [`../metrics.py`](../metrics.py) (`best_gate`, `honest_gate`).
**You need first:** [`auc_and_confidence_signals.md`](auc_and_confidence_signals.md) (gated curves) and `result_thresholding/concepts/answer_or_skip_expected_value.md` (why a threshold exists).

---

## 1. The trap in one sentence

If you pick the threshold that gives the best score on a set of photos and then report *that* score, the number is too good, because the threshold was chosen to fit the particular lucky and unlucky photos in this set.

Think of it as exam practice: choosing the answers that happen to match the answer key and then bragging about your marks.

## 2. A demonstration with pure noise

The clearest way to see the bias is to give the threshold something that carries **no information at all**. Take the points of 60 photos (a realistic mixture of +10, +6, +3, 0 and −20) and attach a **random number** to each photo as its "confidence". Then:

| measure | score (of 60) |
|---|---|
| answer everything | 2.9 |
| best threshold, tuned on the same photos (`best_gate`) | **5.9** |
| honest split-half (`honest_gate`) | **1.9** (± 2.8) |

A random number cannot predict anything, yet tuning on the same photos improves the score from 2.9 to 5.9. That gain is pure fitting of noise. The honest estimate (1.9) is, correctly, no better than answering everything. (Numbers produced with the repo's functions on a fixed seed.)

With a signal that really carries information (right answers have around 14 inliers, wrong ones around 6), the same 60 photos give: tuned 26.6 (answering 53 % of the photos) versus honest 23.2 ± 6.2. The honest score is lower by about 3.4 points; that is the optimism removed. And the ± 6.2 says the final score on new photos could easily be several points higher or lower.

## 3. The cure: tune on one half, score on the other

`honest_gate` repeats the following 300 times:

1. shuffle the photo indices and split them in two halves, **dev** and **test**;
2. on dev, scan 40 candidate thresholds (quantiles of the signal) with `best_gate` and keep the best one, `t`;
3. apply that `t` to **test**: answer the photos whose signal is at least `t`, count the others as 0, and compute the test half's score.

```python
order = rng.permutation(len(points))
dev, test = order[: len(points)//2], order[len(points)//2:]
_, t, _ = best_gate(signal[dev], points[dev])
answered = signal[test] >= t
scores.append(final_score(list(np.where(answered, points[test], 0))))
```

It returns the **mean** score over the 300 splits, its **standard deviation**, and the mean share answered.

Notes on the code:

* `rng = np.random.default_rng(SEED + 19)`: fixed, so the report is reproducible;
* the test half's score uses `points[test]` with skipped photos as 0 **but still counted in the denominator**, exactly as the contest does. Dropping skipped photos from `N` would flatter the result;
* it asserts at least 4 photos.

## 4. How to read the three numbers

* **Mean honest score below the tuned score.** The gap is the optimism you removed.
* **Large standard deviation.** The score depends strongly on which photos landed in which half; the threshold is unstable and the true score is uncertain by about that much.
* **Answered share swinging a lot between splits.** The signal's distribution is not well-behaved; consider a coarse, round threshold (like 12 inliers) instead of a fitted one.

## 5. Limits: what this check cannot do

The 300 splits are re-shuffles of the **same** photos, so they are not independent, and the standard deviation understates the true uncertainty. It is an honesty check, not a confidence interval. The real test is a fresh set of photos or a fresh flight.

Also, each half contains only half the photos, so each threshold is tuned on fewer photos than the final one would be. That makes the honest number slightly *pessimistic*. The truth lies between the honest and the tuned scores, closer to the honest.

## 6. A rule of thumb

Fix as few knobs as possible. One threshold fitted on 60 photos costs a few points of optimism; ten knobs fitted together could cost much more. When a setting is not clearly supported (a difference of one or two photos), keep the simple default.

## 7. Common mistakes

* Reporting `best_gate` as the expected score.
* Tuning the threshold on the whole dataset, then testing the system on that same dataset and calling it validation.
* Using the honest score with a handful of photos: with 12 photos, the halves have 6 each and the result is mostly noise.

## 8. Check yourself

1. A signal made of random numbers gets a tuned score of 5.9 and an honest score of 1.9. What does the 4.0 difference represent?
2. Why are skipped photos kept in the denominator?
3. Why can the honest score be a little too pessimistic?
