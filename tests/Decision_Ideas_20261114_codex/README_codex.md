# Better candidate decisions: ideas and initial checks

The strongest architectural direction is a **joint judge over competing locations, with an explicit “none of these locations” option**, followed by independently calibrated abstention. The existing candidate classifier is useful evidence, but its probability alone is not a certificate that the chosen answer is safe.

All work here is separate from production. No existing settings, source, weights or notebooks were modified. Remove this directory to revert these additions.

## What the present judge misses

The frozen T7 PC rule usually selects a candidate using `inliers + DINO z`, then checks the judge's probability. It allows an unconditional >=12-inlier override. Thus a strong alternative may never be selected, and the override can bypass a rejection. Merely switching the winner to maximum judge probability was already tried in T7 and was worse; that alone is not the proposed new architecture.

Candidate probabilities are not mutually exclusive probabilities over physical locations. Several nearby queries can refer to the same place, while two distant places can both receive high scores. An average or product of these numbers is not a justified probability of correctness. The existing model is trained on right candidates <=4 sigma and wrong candidates >8 sigma, omitting intermediate 4–8 sigma cases. It does not directly model all score tiers or retrieval failure.

## Fixed exploratory experiment results

I fixed the rules in `PROTOCOL_codex.md` before scoring. No weights were refitted and no thresholds were swept. Replayed all 300 stored photos using their original 0.22 matching cut; do not compare these directly to active 0.21 notebook scores. Both sets have already been examined. Official150 overlaps judge training, and NUDT707 shares development flights/scenes. These are regression checks, not independent validation.

Scores /60; parenthetical value is wrong-answer count:

| Rule | Official150 | NUDT707 | Answers official / NUDT |
|---|---:|---:|---:|
| Agreement baseline at 0.22 | 52.80 (0) | 53.84 (1) | 134 / 138 |
| Frozen PC judge | 56.80 (0) | 53.44 (1) | 144 / 137 |
| Strict judge, no override | **54.00 (0)** | **54.24 (0)** | 137 / 136 |
| Unique plausible location | 54.08 (0) | 53.84 (0) | 138 / 135 |
| Judge plus held-out geometry | 45.36 (0) | 50.00 (0) | 115 / 125 |
| Unique location plus held-out geometry | 42.96 (0) | 50.00 (0) | 109 / 125 |

**Strict judge** keeps geometric ranking, requires >=4 inliers and P>=0.98 at both stages, removes the inlier override, and uses no early exit. It first checks six queries, escalating to 27 on abstention. Those three changes form one fixed ablation; this run does not isolate their individual effects. It skips the known NUDT false answer (photo 1181), rather than recovering its correct location. Relative to agreement, NUDT has two improved and two worsened photo scores, net +0.40; official has six improved and three worsened, net +1.20. Paired photo-bootstrap 95% intervals are [-1.60, +2.80] NUDT and [-1.20, +3.60] official, both including zero. Query means rise to 8.24 / 8.80 respectively. This is a useful conservative candidate for validation, not a proven accuracy gain or runtime gain.

**Unique location** ranks by judge probability, requires P>=0.98 and >=4 inliers, and abstains if another >=4-inlier candidate with P>=0.9 is farther than two scoring sigmas. Nearby candidates do not act as extra independent votes. This fixed heuristic obtains zero wrong answers but is not better than the strict rule here. It is a prototype of ambiguity handling, not a trained location-level judge or a conformal set.

**Held-out geometry** divides raw confidence>=0.22 pairs into spatial checkerboard halves before fitting. Each fit must explain at least four withheld matches within the existing three-pixel tolerance; the two fitted canvas centres must agree within three pixels. Existing scale, angle and shift gates remain. It rejects too many valid weak matches when applied universally. Preserve it as an evidence feature or targeted verifier for ambiguous cases, rather than a required gate for every answer. Withholding correspondences is weaker than collecting another independent image, and repetitive structures can pass both halves.

Seven regression tests pass, including conflicting hypotheses, withheld-transform consistency, no inlier override and label-independent selection. CUDA is still unavailable; no new network inference or target-board timing was performed.

## Five directions worth developing

### 1. Joint location ranking with an explicit null option — highest priority

Group near-duplicate candidates into physical-location hypotheses using a fixed spatial rule. Avoid connected-component chaining that could merge distant places through intermediate candidates. For each hypothesis retain geometric support, support extent, pose plausibility, translation residuals, DINO evidence, its strongest far challenger and the fraction of valid image content. Cap or summarise repeated-query evidence; do not multiply correlated query probabilities.

Train a small set-aware ranker to choose between these locations **and a null hypothesis** meaning none can be trusted. A softmax over only candidate places is unsafe: it must assign all probability to some place even when retrieval missed the truth. Include all-wrong shortlists and difficult repeated-structure competitors in training, not mostly easy negatives. Train/calibrate with complete photos grouped by physical flight/site.

The explicit unmatched option is an established design pattern in feature matching, e.g. SuperGlue's dustbin for unmatched keypoints; applying it to geolocation hypotheses here is a proposed extension, not a published guarantee for this pipeline. [SuperGlue paper](https://arxiv.org/abs/1911.11763).

This can improve selection itself: a five-inlier correct candidate may beat an eleven-inlier repetitive false candidate when the whole candidate set exposes the ambiguity. A verifier applied only after selecting the eleven-inlier candidate cannot recover the alternative unless it reconsiders selection.

### 2. Return a set of plausible locations, then answer only a singleton

Instead of asking whether the best candidate exceeds a cut, ask whether it is the **only** defensible location and whether null can be excluded. If the result is {A,B}, {A,null}, or {null}, abstain or obtain extra evidence. If it is {A}, answer A after calibration of the complete policy.

Conformal prediction provides a framework for calibrating prediction sets, subject to its data assumptions. A dynamic geolocation shortlist needs an explicitly defined location/null label and calibration of the actual retrieval-and-selection system. Marginal set coverage does **not** automatically guarantee a low error rate among singleton answers; validate/calibrate that selective risk separately. [Classification with Valid and Adaptive Coverage](https://arxiv.org/abs/2006.02544).

The tested unique-location heuristic is only an inexpensive first approximation. It retains neither a conformal coverage claim nor a calibrated risk claim.

### 3. A challenger-directed verification step

When A and B are plausible, spend extra compute where they differ: identify map regions whose geometry differs after alignment and check which prediction agrees with the thermal image there. If the candidates differ mainly outside the visible thermal footprint, acknowledge that the photo cannot decide and skip.

This differs from the previously tested generic second look that re-centred the same candidate. It explicitly compares the best location with a strong alternative and seeks discriminating evidence. Use existing raw pairs first; additional crop matching or different features require new inference. Keep a strict budget and apply only to ambiguous cases. Not implemented or measured in this experiment.

### 4. Predict position reliability and score tiers, not just a binary label

Add features for extrapolation to the camera-centre position: leave-region-out centre movement, support around the centre, matrix conditioning and repeated-pattern ambiguity. Tiny residuals on a small patch need not imply an accurate centre estimate. Keep fit uncertainty distinct from absolute geolocation uncertainty: a confidently wrong repeated facade can still have a stable transform.

Train ordered probabilities for errors <=1, <=2, <=4 and >8 scoring sigmas, including the zero-point intermediate band. Expected points are

`10 P(e<=1σ) + 6 P(1σ<e<=2σ) + 3 P(2σ<e<=4σ) - 20 P(e>8σ)`.

Select for expected points only among hypotheses that pass a separately calibrated wrong-answer-risk constraint. The null action has utility zero. This better matches the competition objective than treating all <=4-sigma candidates as equally desirable. Not trained here.

### 5. Model disagreement and domain checks as abstention evidence

Use small judges trained on different flight/site folds and compare their predictions, especially across geometry and appearance feature groups. High disagreement, unseen feature combinations or an unfamiliar camera/terrain should trigger abstention or extra verification. An average of confident models is not a certificate; correlated models can all be wrong. This is most useful as an additional feature of the calibrated selector, not as a replacement for independent testing. Not trained here.

## Recommended combined architecture

`fine candidates → location hypotheses → joint ranker + null → plausible-location set → optional challenger verification → calibrated answer/skip`

Start with a small model and explicit features rather than a large new neural network: only two geographic domains and limited independent acquisition groups are available. Changing the training target and representing competitors/null are more important than model size. Retain the current strict judge as a comparison baseline. Do not add an unconditional inlier bypass back into the safety gate.

Training, calibration and final testing must be separated by physical acquisition/site; maintain a canonical image manifest. Re-score all final policies at the actual 0.21 matching cut after regenerating their features, and account for missing candidates and all skipped photos. Define calibration risk as wrong accepted photos divided by accepted photos, measure coverage and points alongside it, and make a failed certification disable answers rather than fall back to an inlier bar.

## What “no wrong answers” can mean

Zero observed mistakes is an acceptance target for a frozen test. It is not a guarantee on arbitrary unseen inputs. For illustration, even with independent observations and a rule fixed in advance, zero mistakes in 136 accepted photos has a one-sided 95% binomial upper error bound of about **2.18%**. Roughly **2995 independent accepted photos with zero mistakes** would be needed for that bound to fall below 0.1%. These exploratory, spatially correlated samples do not meet those assumptions.

If two locations are indistinguishable in the available image, reliable behavior is to abstain or obtain genuinely new evidence. Any claimed perfect selector that always answers should be treated sceptically.

## Reproduction

From the IdeaForge repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ../venv-geoloc/bin/python tests/Decision_Ideas_20261114_codex/test_ideas_codex.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ../venv-geoloc/bin/python tests/Decision_Ideas_20261114_codex/run_ideas_codex.py
```

`results/summary_codex.csv` contains all fixed comparisons, `per_photo_codex.csv` records every outcome, and `strict_changed_photos_codex.csv` identifies changes from agreement. Selection functions in `decision_ideas_codex.py` never read ground-truth or score columns. Ground truth is used only by the runner after selection. New architecture proposals above remain untrained; the file makes a clear distinction between implemented prototypes and future experiments.
