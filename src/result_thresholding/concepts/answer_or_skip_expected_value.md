# Answer or skip: the break-even confidence

Code: `decide` in `src/result_thresholding/decision.py`; scoring in `src/header/scoring.py`; the threshold is tuned in `src/evaluation/metrics.py`.

---

## 1. The payoff (see `header/concepts/scoring_rule.md`)

Per photo the tiers are +10 / +6 / +3 / 0 / −20 (best tier to the worst, a far-off answer); skipping gives 0. So a wrong answer costs more than a right one earns, and sometimes the best action is to skip.

## 2. Expected value of answering

Let `p` = probability that an answer earns a positive tier, `r` = average reward when it does (at most 10), and a miss costs 20. Then

$$E[\text{answer}]=p\,r-(1-p)\,20,\qquad E[\text{skip}]=0.$$

Answer exactly when `p·r > (1−p)·20`, i.e.

$$p>\frac{20}{20+r}.$$

With `r = 10` (every right answer in the top tier) this is `p > 2/3`. Real rewards are mixed (+10/+6/+3), so `r < 10` and the true break-even is higher than 2/3. The practical threshold is therefore **measured on data**, not computed from this formula.

## 3. From probability to a number we can observe

We do not know `p` for a new photo. We know a score that correlates with it: the inlier count `n`. Define a threshold `T` and rule "answer iff `n ≥ T`". The ideal `T` is where the probability of correctness conditional on `n` crosses the break-even above. A larger `n` is stronger evidence (see the chance-inlier calculation in `fine_matching/concepts/similarity_transform_and_ransac.md`: 12 accidental inliers are astronomically improbable, the main risk is coherent look-alike scenes).

The project fixes one global `T = ANSWER_MIN_INLIERS = 12`: the same for every photo, because the score is the sum over photos and the gate must be known **before** seeing the test photo.

## 4. Why a fixed threshold per photo (not "answer the best 70 %")

A rule like "answer the top 70 % of photos by confidence" needs all photos at once (it uses other photos' scores). The evaluation runs each photo independently, so the gate must be an absolute value of the photo's own confidence (see `one_photo_at_a_time.md`).

## 5. Choosing T honestly

Picking `T` to maximise the score on the same photos that report the score is optimistic (you tune to the noise). `evaluation/metrics.py::honest_gate` splits the photos in half: it picks `T` on half A, scores on half B, and the reverse, and reports the average. If that number is close to the tuned one, the threshold generalises. See `evaluation/concepts/honest_threshold_selection.md`.

## 6. Reading the numbers

* Skipping everything scores 0.
* Answering everything scores `Σ points / N`.
* The gated curve in the report shows the score for each `T`; the best `T` sits where adding lower-confidence photos stops paying for their wrong answers.
