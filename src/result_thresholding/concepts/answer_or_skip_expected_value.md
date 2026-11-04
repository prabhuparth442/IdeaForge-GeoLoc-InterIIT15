# Answer or skip: where is the break-even?

**Code:** `decide` in [`../decision.py`](../decision.py); the points in `src/header/scoring.py`; the threshold is tuned honestly in `src/evaluation/metrics.py`.
**You need first:** `header/concepts/scoring_rule.md` (the five point tiers).

---

## 1. The payoff, in one line

Each photo earns **+10, +6, +3, 0 or −20** points depending on how far the answer is from the truth (in units of σ), and a skipped photo earns **0**. A far-off answer costs twice what the best answer earns. So there are photos for which the best action is to say nothing.

## 2. Expected value of answering

Let

* `p` = probability that our answer lands in a positive tier,
* `r` = the average reward when it does (at most 10, since the tiers are 10, 6 and 3),
* a miss costs 20.

Then on average, answering earns

$$E[\text{answer}]=p\,r-(1-p)\cdot20,\qquad E[\text{skip}]=0.$$

Answer exactly when `p·r > (1−p)·20`. Solving for `p`:

$$p\,r>20-20p\ \Longrightarrow\ p(r+20)>20\ \Longrightarrow\ \boxed{p>\frac{20}{20+r}}.$$

### The break-even for different average rewards

| average reward when right `r` | break-even `p` |
|---|---|
| 10 (every right answer is top tier) | 0.667 |
| 8 | 0.714 |
| 7 | 0.741 |
| 6 | 0.769 |

Real answers are a mix of 10, 6 and 3, so `r` is below 10 and the true break-even is **above two-thirds**, more likely about 0.7 to 0.77. In words: *answer only when you are at least about 70 to 77 % sure of landing in a positive tier.*

### Two quick examples

* `p = 0.8`, `r = 8`: `0.8·8 − 0.2·20 = 6.4 − 4 = +2.4`. Answer.
* `p = 0.7`, `r = 8`: `5.6 − 6 = −0.4`. Skip (barely).

## 3. From "probability" to something we can measure

We never know `p` for a new photo. What we have is a number that moves with it: the **inlier count** `n`. More inliers means more landmark pairs agree, which is stronger evidence (see the chance-inlier calculation in `fine_matching/concepts/similarity_transform_and_ransac.md`). So we define the rule:

> answer if and only if `n ≥ T`.

The best `T` is where "the probability of being right, given `n`" crosses the break-even above. The project uses a single value `T = ANSWER_MIN_INLIERS = 12`.

## 4. Why one fixed `T` and not "answer the best 70 %"

A rule such as "answer the 70 % of photos with the highest confidence" needs the scores of **all** photos at once. But each photo is localised independently (see [`one_photo_at_a_time.md`](one_photo_at_a_time.md)); the gate for photo 17 cannot depend on photo 300. So the threshold has to be an absolute value of the photo's own evidence, and it has to be fixed **before** the test photos are seen.

## 5. A worked example of choosing `T`

Ten photos, each with its inlier count and the points it would earn if answered:

| photo | inliers | points |
|---|---|---|
| 1 | 30 | +10 |
| 2 | 25 | +10 |
| 3 | 22 | +6 |
| 4 | 18 | +10 |
| 5 | 15 | +3 |
| 6 | 14 | −20 |
| 7 | 12 | +6 |
| 8 | 10 | −20 |
| 9 | 8 | 0 |
| 10 | 5 | −20 |

For each `T`, answer the photos with `inliers ≥ T`, add the points of those, and compute the score `60 × max(Σ/(10·N), 0)` with `N = 10` (skipped photos still count in `N`, they just give 0):

| `T` | photos answered | sum of points | score |
|---|---|---|---|
| 30 | 1 | 10 | 6.0 |
| 25 | 2 | 20 | 12.0 |
| 22 | 3 | 26 | 15.6 |
| 18 | 4 | 36 | 21.6 |
| **15** | 5 | 39 | **23.4** |
| 14 | 6 | 19 | 11.4 |
| 12 | 7 | 25 | 15.0 |
| 10 | 8 | 5 | 3.0 |
| 5 (answer all) | 10 | −15 | 0 |

Reading it: a very strict `T` throws away good photos; a lax `T` lets in a −20 that wipes out several +10s. The best `T` here is 15, where adding photo 6 (one wrong answer, −20) would undo the gain of two correct ones. Also notice **answering everything is worth 0** (the negative sum is floored at 0 by the scoring formula) while a good gate gives 23.4. The same effect, with real data, is the "gated" curve in the evaluation report.

## 6. Choosing `T` honestly

In the table above `T = 15` is best *on these ten photos*. But we picked it on the same photos that score it, which is optimistic: part of the gain is fitted noise (maybe photo 6 just happened to be wrong). To measure how much, `evaluation/metrics.py::honest_gate` splits the photos in half, picks `T` on one half, scores it on the other, swaps, and averages. If that honest number is close to the tuned one, the threshold generalises. If it is much lower, the tuned number was flattering. See `evaluation/concepts/honest_threshold_selection.md`.

## 7. Reading the numbers in the report

* Skip everything: score 0.
* Answer everything: `Σ points / N`, which can be negative (shown as 0 after the floor).
* The gated curve shows the score for each `T`. The best `T` sits where an extra, less confident photo stops paying for its occasional −20.

## 8. Common mistakes

* Using `p > 0.5` as the break-even (it would be right if wrong and right answers were worth the same).
* Counting skipped photos out of `N`. They count, with 0 points.
* Tuning `T` and reporting the score from the same photos without the honest split.

## 9. Check yourself

1. If the average reward when right were 5, what is the break-even? (`20/25 = 0.8`.)
2. In the table above, what happens to the score if photo 6 had been correct (+10)? Which `T` would be best then? (Then the sum at `T=14` is 49 (score 29.4) but at `T=12` it is 55 (score 33.0), so `T = 12` wins: one different photo moves the best threshold, which is why the honest split of §6 matters.)
3. Why can `T` not be set per photo?
