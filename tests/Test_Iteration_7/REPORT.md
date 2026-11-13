# Test_Iteration_7 — Full Report

Branch `palash/dev`. Builds on Test_Iteration_6 (rich dumps with raw LoFTR matches for every candidate of all 27 queries) and on the
`prabhuparth/dev` pipeline blocks. `src/` is not edited.

## 1. Goal

The decision after fine matching decides whether a photo scores +10/+6/+3 or 0, and a wrong answer costs -20. Test_Iteration_6 showed most lost
points are **skips of right places** with 4-10 inliers, and that fresh NUDT city photos (404, 505) bring **look-alike places with 8-11 inliers**
that the inlier bars answer wrongly. Multi-query agreement helped T6 but the team ruled it out as too risky. This iteration works only on the
decision after fine matching, along two routes:

- **A. More lines for right places** (so the bar is reached by right places and not by wrong ones).
- **B. Judge with every signal available after fine matching** (not only inliers and DINO z).

Acceptance (unchanged from T6): a gain counts when it holds on official and NUDT on held-out data with 0 wrong answers and a paired-bootstrap
interval above 0, and on photos no rule has seen. Time per photo on the Jetson is reported in the team's classes: **< 4 s good, 4-6 s viable,
>= 6 s slow** (read as the time a method adds to or removes from a photo, since the classes were stated for a method).

## 2. Pipeline under test (unchanged)

Thermal photo -> north-up canvas (224 px) -> edge ZNCC FFT coarse search -> K = 50 peaks -> DINO re-rank (z) -> LoFTR-MINIMA on M = 10 (Jetson
M = 5) -> RANSAC similarity (3 px; |scale-1| <= 0.15, |angle| <= 12 deg; shift <= 12 px) -> choice by inliers + z. Queries: standard 6
(heading -5/0/+5 x scale 1.0/1.05); wide 27 (-10..+10 step 2.5 x 0.95/1.0/1.05); Jetson second pass: 9 headings x (1.0, 1.05), cap 10 queries.

## 3. Data

| Set | Photos | Role |
|---|---|---|
| official | 500 (pool 500) | seen |
| NUDT seed 101, 202 | 300 each (pool 500) | seen |
| NUDT fresh 404, 505 | 300 each (pool 2400, the 500-pool excluded) | seen |
| **NUDT fresh 606** | 300 (pool 2400, the 500-pool and every 404 / 505 photo excluded) | **blind**: dumped before any T7 rule, signals built and scored only after the freeze |

Photo ids index the loaded pool, so 101 / 202 ids (pool 500) and 404 / 505 / 606 ids (pool 2400) are not comparable; the 606 draw excluded the
whole 500-pool, and shares 0 photos with 404 / 505 (checked in It5).

## 4. Method

One GPU dump per set (T6 `Rich_Dump_Run`, raw matches of every candidate). Everything else is offline:

- `train/checks/decision_signals.py`: per candidate, from the stored matches: counts at two confidence cuts, pipeline-fit inliers, LO-RANSAC
  inliers, translation-consensus inliers, guided (faint) inliers, a-contrario NFA (log10), inlier convex hull and 3x3 coverage, spread, residual,
  inlier confidence, fit scale/angle/shift. ~0.93 ms per candidate on one x86 core.
- `train/checks/candidate_judge.py`: the candidate judge, decision replays (PC and Jetson), cut rules, leave-one-set-out, paired bootstrap, and the
  frozen rule.
- `train/checks/second_look.py`: a second LoFTR look at one candidate (GPU).

Protocol everywhere: **leave one set out** (LOSO). For each of the five seen sets, the judge and its cut are fitted on the other four, and the
held-out set is replayed once. Within the training sets, the judge's P is out-of-fold (5 folds grouped by photo), so the cut is never chosen on
in-sample predictions.

## 5. Ideas explored

| # | Idea | Route | Verdict | Where |
|---|---|---|---|---|
| 1 | Re-fit stored matches: LO-RANSAC, lower confidence cut + guided matching | A | ~0 extra lines (guided15 = inliers on almost every candidate) | It1 |
| 2 | Translation-only consensus (pairwise exact) | A/B | weaker than the similarity fit (AUC 0.88 / 0.95 vs 0.97 / 0.99) | It1 |
| 3 | A-contrario NFA (ORSA-style) instead of an inlier count | B | same ranking as inliers (it is a monotone function of n, k); no gain | It1 |
| 4 | Inlier geometry (hull, 3x3 coverage, spread) | B | **strongest new signal**: hull AUC 0.981 official / 0.99 NUDT | It1, It2 |
| 5 | Candidate-level judge (monotone GBM on all signals, every candidate) | B | **held-out gains, 0 wrong**; blind +0.7 / +0.4, not proven | It2, It4, It5 |
| 6 | Query-local uniqueness (gap to the strongest far place *in the same query*; not agreement) | B | +1.5 on official on top of 5 | It2 |
| 7 | Logistic regression judge | B | much worse (official 49.58) | It2 |
| 8 | Choose the candidate by the judge instead of inliers + z | B | worse (53.16 vs 54.36) | It2 |
| 9 | Second look, tile re-centred on the fitted place | A | +0.6 lines for right places; wrong places collapse (4.5 -> 0) | It3 |
| 10 | Second look, query re-aimed by fitted heading / scale | A | loses lines (-4.3) | It3 |
| 11 | Second-look signals in the judge | B | official -1.08 [-2.12, -0.09]: dropped | It3 |
| 12 | Cut by Learn-then-Test (risk bound) instead of zero-wrong + margin | B | best PC rule (alpha 0.005); 0.01 lets wrong answers in | It4 |
| 13 | Floor 0.9 on the cut + stricter cut (0.95) for the extra queries | B | needed on the Jetson replay: 0 wrong on every held-out set | It4 |
| 14 | **Early exit**: answer after any standard query whose choice has P >= 0.98 | time | **~6.3 -> ~2 queries per photo** (-9.6 s), score ~unchanged | It4, It5 |
| 15 | Hopeless stop (skip the extra queries when every standard candidate has low P) | time | not finished (superseded by 14) | — |
| 16 | Minimum 4 inliers for an answer | B | -0.16 on official, 0 wrong; candidate for the next blind set | It6 |

## 6. The iterations

### It1 — Signal table
Every candidate with >= 3 inliers of every query of every photo (372k rows over five sets). AUC of right (<= 4 sigma) vs wrong (> 8 sigma):
inliers 0.967 official / ~0.99 NUDT; hull 0.981 / 0.99; DINO z ~0.98; translation consensus 0.88 / 0.95. Re-fitting the stored matches with
lower confidence and guided matching gives ~0 extra lines: LoFTR's matches at the 0.22 cut already contain what the fit can use.

### It2 — Candidate-level judge
A verifier on the chosen candidate sees ~5 wrong answers in 1,379 photos, too few to learn from (T6). The judge instead learns from **every
fine-matched candidate**: 9,600 labelled wrong and 23,500 right candidates with >= 3 inliers. Model: `HistGradientBoostingClassifier` (depth 3,
200 iterations, lr 0.05, L2 1.0, no early stopping so it is deterministic), P(right) forced to rise with inliers, z, hull and coverage.
Decision: the pipeline's choice among candidates with >= 3 inliers; answer if P >= cut or inliers >= 12; standard queries first, then wide.

Feature groups (LOSO, zero-wrong + margin 1 cut, PC):

| Features | official | 101 | 202 | 404 | 505 |
|---|---|---|---|---|---|
| inliers + z | 49.90 | 54.84 | 56.46 | 55.74 | 55.56 |
| + geometry | 51.42 | 55.64 | 56.46 | 56.34 | 55.96 |
| + fit | 51.42 | 55.44 | 56.46 | 56.70 | 56.16 |
| + counts | 52.53 | 55.64 | 56.46 | 56.70 | 55.76 |
| all | 52.81 | 55.72 | 56.46 | 56.70 | 56.08 |
| all + query-local gaps | 54.36 | 55.72 | 56.46 | 56.70 | 55.96 |

The plain inliers + z judge gains nothing: the gain comes from the geometry, the counts and query-local uniqueness. Without the monotone
constraints the judge is worse (official 53.1).

### It3 — Second look (GPU, ~0.13 s per look on the PC)
For up to 5 candidates per photo (chosen without the truth) LoFTR looked again with the tile re-centred on the fitted place, and with the query
re-drawn using the fitted heading and scale. Right candidates: median 48 -> 49 lines (re-centred), re-aiming loses lines. Wrong candidates:
median 4.5 -> 0, only ~10 % gain. So LoFTR is not losing lines to the off-centre tile; the second look is a repeatability test. As judge inputs
(lines, ratio, hull, coverage of the second look) it lowered official by 1.08: what it measures is already in hull, coverage and the gaps.

### It4 — Decision rule and freeze
Cut rules on the PC (LOSO, all with the It2 judge):

| Rule | official | 101 | 202 | 404 | 505 |
|---|---|---|---|---|---|
| zero-wrong + margin 0 | 54.52 | 55.72 | 56.46 | 56.70 | 56.36 |
| zero-wrong + margin 1 | 54.36 | 55.72 | 56.46 | 56.70 | 55.96 |
| zero-wrong + margin 2 | 53.00 | 55.52 | 56.46 | 56.52 | 55.96 |
| Learn-then-Test alpha 0.002 | 51.01 | 55.32 | 56.46 | 56.22 | 55.76 |
| **Learn-then-Test alpha 0.005** | **55.16** | **55.72** | **57.06** | **56.70** | **56.96** |
| Learn-then-Test alpha 0.01 | 56.36 | 54.56 / 3w | 55.26 / 3w | 56.70 | 56.36 / 1w |

Learn-then-Test (Angelopoulos et al.) walks the cuts from the strictest down while the 90 % Clopper-Pearson upper bound of the wrong rate among
answered training photos stays <= alpha; with 0 wrong in n answers that bound is ~2.3 / n, so it refuses cuts that are clean only on little data.

On the **Jetson replay** (M = 5, extra queries one at a time) the same rule picks 0.8-0.85 and wrong answers appear (101: 1, 202: 2). A floor of
0.9 (the lowest cut any PC fold chose) removes the 202 pair; the last one (101 photo 101: a 4-inlier look-alike, P 0.93, found on the 9th query)
is an extra-query answer. More queries give a look-alike more chances, which is why today's bars rise from 8 to 11 for the wider queries; a cut
of 0.95 for the extra queries removes it. **Frozen rule** (`results/frozen_rules_t7.json`): judge on all five seen sets, Learn-then-Test alpha
0.005, floor 0.9, extra-query cut 0.95, override 12 inliers, min 3 inliers; cut = 0.9 in all four variants.

| Held out (LOSO) | official | 101 | 202 | 404 | 505 | queries |
|---|---|---|---|---|---|---|
| PC judge | 54.84 (+4.88 [+3.26, +6.62]) | 55.72 (+0.88 [+0.08, +1.80]) | 57.06 (0) | 56.70 (+2.46 [+0.48, +4.98]) | 56.76 (+2.52 [+1.20, +4.04]) | 7.5-8.3 |
| PC + early exit | 54.17 | 55.64 | 56.58 | 57.06 | 56.52 | 3.0-4.4 |
| Jetson judge | 53.16 (+2.88 [+1.52, +4.31]) | 54.92 (+0.68) | 55.26 (0) | 57.00 (+1.56 [+0.48, +2.94]) | 55.56 (+0.88) | 6.2-6.4 |
| Jetson + early exit | 52.50 (+2.22 [+0.66, +3.82]) | 54.84 | 54.78 (-0.48) | 57.36 (+1.92 [+0.78, +3.30]) | 55.32 | 1.7-2.5 |

0 wrong answers in every cell.

### It5 — Blind NUDT fresh 606 (scored once)

| Variant | Score | Wrong | Answered | Queries | Baseline | Gain | Jetson s / photo |
|---|---|---|---|---|---|---|---|
| PC judge | 55.80 | 2 | 96.3 % | 7.54 | 55.08 / 2w | +0.72 [-0.56, +2.12] | — |
| PC + early exit | 55.48 | 2 | 96.3 % | 3.14 | 55.08 / 2w | +0.40 [-0.92, +1.84] | — |
| Jetson judge | 54.82 | 1 | 93.0 % | 6.30 | 54.42 / 2w | +0.40 [-0.60, +1.60] | 13.9 |
| **Jetson + early exit** | **54.50** | **1** | 93.0 % | **1.90** | 54.42 / 2w | +0.08 [-0.96, +1.32] | **4.2** |

Changed photos (PC): +5 right answers gained, 1 wrong removed (254: 8 inliers, P 0.12), 1 wrong added (2071: **3 inliers**, z 2.64, P 0.95, wide
queries). Photo 182 (8 inliers, P 0.96) is wrong under every rule, the baseline included. Jetson: 3 gained, 2 lost, 1 wrong removed, 0 added.

### It6 — After the blind set (seen sets only; not evidence on 606)

| Min inliers for an answer | PC official | Jetson early official | others | wrong |
|---|---|---|---|---|
| 3 (frozen) | 54.84 | 52.50 | — | 0 |
| 4 | 54.68 | 52.34 | unchanged | 0 |
| 5 | 54.05 | 51.55 | unchanged | 0 |

## 7. Time on the Jetson

2.2 s per LoFTR query at M = 5 (T6 measurement) + the judge's signals (~5 candidates x 0.93 ms per query on x86, taken x3 for the Jetson CPU) +
the judge (< 1 ms). The judge adds ~0.1 s per photo (**good**). The early exit removes ~4.3 queries: **~14 s -> ~4.5 s per photo** on the seen
sets and 13.9 -> 4.2 s on the blind set, which moves the whole photo from the *slow* class to *viable/good* and frees ~9.6 s of the 22 s budget
(which could pay for more peaks, a larger M, or XoFTR on hard photos). No port to another language is needed: the signals are numpy / OpenCV
on ~5 candidates per query; if the Jetson CPU is slower than modelled, `candidate_signals` is the hot spot to port (C++ / numba), and the
LO-RANSAC, guided-matching and NFA parts it computes are not used by the judge and can be dropped (`with_lo=False`).

## 8. Recommendation

1. **Adopt the early exit on the Jetson** with the frozen judge (P >= 0.98 on any standard query answers the photo). It is the one result that
   holds on the blind set: same score as today (54.50 vs 54.42), fewer wrong answers (1 vs 2), and ~70 % less matching time.
2. **Do not claim a score gain from the judge yet.** It is clearly better on held-out seen sets (+0.9 to +4.9, 0 wrong, wrong answers on 404 /
   505 removed), but on the blind set the gain interval includes 0 and one new wrong answer appears.
3. **Next blind set:** the frozen rule with a minimum of 4 inliers for an answer, scored once on a new fresh NUDT draw (seed != 101/202/404/505/606,
   pool 2400, the 500-pool and every earlier fresh photo excluded) plus, if available, more official-like photos.
4. Route A (more lines) is closed for the post-matching stage: LoFTR already returns what the fit can use, and a second look adds +0.6 lines.

## 9. Problems hit and how they were handled

- **Thread oversubscription (hours instead of minutes).** Jobs were pinned to 10-12 cores but scikit-learn's OpenMP started 32 threads each;
  three such jobs made judge training ~100x slower. Fix: `OMP_NUM_THREADS` = allowed cores in every notebook, one job at a time, and the five
  leave-one-set-out judges trained once (`held_out_probabilities`) and reused by every replay. A full LOSO replay now takes ~1 minute.
- **Non-deterministic judge.** `HistGradientBoostingClassifier` early stopping draws a random validation split: turned off.
- **NFA of -inf** broke AUCs: floored at -1000.
- **CUDA in forked workers**: the signal table uses a spawn pool with the GPU hidden.
- **USAC_MAGSAC** is not available in `estimateAffinePartial2D`: LO-RANSAC by LMEDS refits instead.
- **Photo-id overlap scare** on 606: ids index the loaded pool, so 101/202 ids are not comparable with 606 ids; checked against 404/505 only.

## 10. Sources and inspiration

- Moisan, Moulon, Monasse — *Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers* (ORSA), IPOL 2012.
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*, arXiv 2110.01052.
- Angelopoulos, Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv 2107.07511.
- Liu et al. — *Anytime Dense Prediction with Confidence Adaptivity* (confidence-gated early exits), ICLR 2022, arXiv 2104.00749.
- Rabanser, Papernot — *What Does It Take to Build a Performant Selective Classifier?*, arXiv 2510.20242 (abstention improves through
  ranking, i.e. new signals, not through monotone recalibration: matches It2, where the gain came from new features).
- Lebeda, Matas, Chum — *Fixing the Locally Optimized RANSAC* (LO-RANSAC), BMVC 2012.
- *Can You Trust Your Pose? Confidence Estimation in Visual Localization* (inlier-based confidence for localization).
- XoFTR / MINIMA (thermal-visible matching), used in T6.

## 11. Reproducing

```bash
# unit tests (made-up data, no GPU)
python tests/pipeline/test_iteration7_blocks.py
# GPU: blind dump and second look (pinned off the faulty cores)
bash tests/Test_Iteration_7/run_blind_dump.sh
bash tests/Test_Iteration_7/run_second_look.sh
# offline: run notebooks It1 (ONLY=nudt606 for the blind table), It2, It3, It4 (writes the frozen rule), It5, It6
```

## 12. Files

- `notebooks/` — It1-It6 and `Second_Look_Run`, each with executed copies.
- `results/frozen_rules_t7.json`, `results/judge_t7.joblib` — the frozen rule and model.
- `results/it2_*`, `it4_*`, `it5_*`, `it6_*` — tables behind every number above; `results/second_look/` — second-look runs.
- `results/signals/*.csv.gz` (46 MB) — signal tables; not committed (rebuilt by It1 from the T6 dumps).
- Blocks: `train/checks/decision_signals.py`, `candidate_judge.py`, `second_look.py`; tests `tests/pipeline/test_iteration7_blocks.py`.
