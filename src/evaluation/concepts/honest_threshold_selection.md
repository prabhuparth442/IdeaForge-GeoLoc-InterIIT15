# Honest threshold selection (split-half)

Code: `best_gate`, `honest_gate` in `src/evaluation/metrics.py`.

---

## 1. The trap

Choose the threshold `T` that maximises the score on the evaluation photos, then report that score. The score is biased upward: `T` was tuned to fit the particular noise of these photos. With a single threshold the bias is small but real; with several knobs it can be large.

## 2. The cure: tune on one set, report on another

`honest_gate` repeats 300 times:

1. shuffle photo indices randomly, split into two halves (dev, test);
2. `best_gate` on dev: scan 40 threshold candidates (quantiles of the signal) and keep the one with the highest score;
3. apply that `T` to test: `answered = signal[test] >= t`, compute the score of test alone.

```python
order = rng.permutation(len(points))
dev, test = order[: len(points)//2], order[len(points)//2:]
_, t, _ = best_gate(signal[dev], points[dev])
answered = signal[test] >= t
```

It returns mean score, its standard deviation across the 300 splits, and the mean answered share.

## 3. How to read it

* Mean of honest scores < tuned (single-set) score: the gap is the optimism you removed.
* Std large (e.g. several points): the threshold is unstable; the score is uncertain by that much on new data.
* Answered share: if it swings a lot between splits, the signal's distribution is not well-behaved.

The thresholds picked by the splits should be similar; if they jump around, use a fixed round number (like 12 inliers) rather than a fitted one.

## 4. Limits

The halves overlap across repeats (they are reshuffles of the same data), so the 300 values are not independent and the std understates uncertainty. This is an honesty check, not a confidence interval. The real test is a fresh set of photos or a fresh flight.

## 5. Why scores use `final_score(list(np.where(answered, points, 0)))`

Skipped photos count 0 points but **stay in the denominator** `N`, just as in the contest. Dropping them from `N` would make the score look better than it is.
