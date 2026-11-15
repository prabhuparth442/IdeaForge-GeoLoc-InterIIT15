# Test_Iteration_10 — Full Report

Branch `palash/dev`. `src/` unchanged. Brief (from the user): finish T9; fix the evaluation before any new experiment; make the judge robust to
the inlier scale (the T9 finding); then the accuracy experiments; protocol: build on the NUDT design sets only, check on official TRY, a fresh
NUDT 808 draw dumped before choosing any rule, pre-registered finalists, 0 wrong answers everywhere. No multi-query agreement.

## 1. Evaluation fixes (It1)

| Fix | What changed | Effect |
|---|---|---|
| a. Jetson context | `add_context(table, m=5)`: zgap / ligap from the 5 candidates the Jetson actually matches (before: from all 10 stored) | small: official TEST unchanged (+3.46), NUDT 707 +0.80 -> +1.00, design sets +-0.3; the NUDT 202 wrong answer disappears |
| b. Empty shortlist | `decide` returns a skip row for every photo with no usable candidate, also when M = 5 leaves none | no exception, no dropped photo (unit test) |
| c. Failed calibration | `ltt_tau` returns None when no cut passes -> `NO_ANSWER`: answer nothing, override included (before: fell back to the strictest cut) | unit test |
| d. Manifest | `eval_tools.manifest` saved next to results; `compare_with_headless` | differences from the submission: hint = 5 km random (research) vs truth (submission Jetson run); 10-query cap vs 22 s budget; PC 27 queries include scale 0.95 (submission 18 do not); matcher confidence 0.22 + coarse 0.21 vs one documented 0.21 (check in the headless code); judge / override / early exit research-only |
| e. Failure log | `eval_tools.failure_log`: never retrieved / retrieved but unmatched (< 3 inliers) / retrieved but rejected / answered off / wrong / right | official is never "never retrieved"; its loss is "retrieved but rejected" (Jetson today 28 on TEST, judge 16). NUDT loses 3-14 photos per set to "never retrieved" (coarse / DINO) |

Unit tests: `tests/pipeline/test_iteration10_blocks.py` (4 pass); T7 (10) and T8 (3) still pass.

## 2. Inlier-scale robustness (It2, block `train/checks/scale_robust.py`)

Variants, all trained on NUDT design sets only: V0 the T8 judge with its 12-inlier override; V1 without it; V2 relative inliers (inliers
relative to the running mean of the far-candidate (> 4 sigma) inliers of the run: label-free, dev prior); V3 scale augmentation (train on
copies x0.5 ... x2); V4 V1 + safety switch (strict cut when the running far level leaves the dev range); V5 relative + augmentation.
Test: every inlier-count signal of a held-out set scaled x0.5 ... x2 (leave one NUDT set out; official TRY); PC and Jetson.

Wrong answers summed over the five held-out NUDT sets and official TRY:

| | x0.5 | x0.75 | x1 | x1.25 | x1.5 | x2 |
|---|---|---|---|---|---|---|
| PC V0 (override) | 0 | 0 | 1 | 5 | 15 | 31 |
| PC V1 (no override) | 0 | 0 | 1 | 3 | 5 | 6 |
| PC V3 augmentation | 1 | 1 | 2 | 2 | 1 | 2 |
| **PC V5 relative + augmentation** | 1 | 1 | 1 | 1 | 1 | 1 |
| Jetson V0 | 0 | 1 | 1 | 2 | 5 | 16 |
| Jetson V5 | 1 | 1 | 1 | 2 | 2 | 2 |

V5's single PC wrong answer is the same NUDT 606 look-alike at every scale (it is wrong at x1 too): the scale failure is gone, the base-level
look-alike remains. By the brief's rule (0 wrong at every scale) no variant passed on the design hold-outs; V5 and the strict cut went to the
pre-registered final (section 5). On the T9 altered photos (real pipeline runs) V3 / V5 make 2-3 wrong answers, no better than the plain judge.

## 3. Accuracy experiments

- **3a targeted verification** (It3): when two distant candidates are both plausible (judge P >= 0.3 / 0.5), the judge already answers 96 %
  of those photos with P ~ 1 and the choice is right; photos it would actually decide are ~1 per NUDT set and 0 on official TRY. A perfect
  tie-breaker would add <= 0.4 points per 60. Not built.
- **3c stage cache** (It4, block `train/checks/stage_cache.py`): the cached pipeline reproduces the T6 dumps exactly (1,350 of 1,350
  candidates: inliers, places, DINO z); reusing the cache cuts a research run from 8.0 to 4.7 s per photo (-41 %). Sharing the map-window
  FFT between queries is also identical but saves only 3 % on this GPU; adopted for the research runner (`SHARE_FFT=1`). The old dumps stored
  only the 10 kept peaks; the cache now keeps the whole K = 50 shortlist with DINO scores.
- **3b coarse / DINO mixture** (It7): see section 6.

## 4. Pre-registration (It5)

`results/preregistered_t10.json` written 15:25:45, before NUDT 808 was read (it was still being dumped). Finalists, trained on NUDT
101-606, cuts by Learn-then-Test on them: J0 = T8 judge + override 12 (cut 0.9); J1 = J0 without override (0.9); J2 = J1 at a fixed cut 0.98;
V5 = relative + augmentation, no override (0.9). Jetson: context from 5, early exit 0.98. Criteria: a gain needs 0 wrong, no more wrong than
the base and a paired-bootstrap interval above 0 at x1; "scale-robust" needs 0 wrong at every scale x0.5 ... x2.

## 5. Final: NUDT 808 (300 fresh photos), scored once (It6)

| | score | wrong | answered | queries | gain vs base | vs same-answers bar | wrong over x0.5 ... x2 | verdict |
|---|---|---|---|---|---|---|---|---|
| PC base (bar 8 / 11) | 52.08 | 5 | 277 | 8.1 | — | — | 0 / 3 / 5 / 6 / 6 / 12 | — |
| PC J0 (T8, override) | **54.80** | 0 | 276 | 8.2 | **+2.72 [+0.92, +4.92]** | +2.72 | 0 / 0 / 0 / 4 / 5 / 8 | gain, **not scale-robust** |
| PC J1 (no override) | 54.28 | 0 | 273 | 8.3 | +2.20 [+0.28, +4.48] | +1.84 | 0 / 0 / 0 / 1 / 1 / 2 | gain |
| PC J2 (cut 0.98) | 54.20 | 0 | 273 | 8.7 | +2.12 [+0.16, +4.32] | +1.76 | 0 at every scale | gain, scale-robust |
| **PC V5 (relative + augmentation)** | 54.48 | 0 | 274 | 8.3 | **+2.40 [+0.52, +4.64]** | +2.04 | **0 at every scale** (54.1-54.5) | **gain, scale-robust** |
| Jetson base (today) | 52.68 | 2 | 271 | 6.4 | — | — | 0 / 2 / 2 / 3 / 4 / 9 | — |
| Jetson J0 (T8) | 53.32 | 0 | 269 | 2.2 | +0.64 [-0.48, +2.04] | +0.88 | 0 / 0 / 0 / 2 / 2 / 3 | not proven |
| Jetson J1 | 53.00 | 0 | 267 | 2.2 | +0.32 [-0.92, +1.80] | +0.76 | 0 at every scale | not proven; scale-robust |
| Jetson J2 | 52.40 | 0 | 264 | 2.2 | -0.28 [-1.64, +1.24] | -0.04 | 0 at every scale | no gain; scale-robust |
| **Jetson V5** | 53.00 | 0 | 267 | 2.2 | +0.32 [-0.92, +1.80] | +0.76 | **0 at every scale** (53.0 flat) | not proven; scale-robust |

Failure log on 808: never retrieved 8-9 (PC) / 17 (Jetson) photos for every rule (coarse / DINO limit); the judges move the base's 2-5 wrong
answers into skips.

## 6. Coarse / DINO mixture (It7)

Same budget M = 10 per query, the 6 standard queries: the pipeline (first 10 of the DINO order) vs the first 7 of the DINO order + the first
3 of the coarse order not already in (`select_mixture(7, 3)`; official TRY 185 photos, NUDT 101 100 photos; runs `mix_*`, 0 failures).

| | queries | right place in DINO top 5 | top 7 | top 10 | mixture 7 + 3 | DINO dropped it, coarse kept it | mixture lost a DINO 8-10 right place |
|---|---|---|---|---|---|---|---|
| official TRY | 1,110 | 75.7 % | 75.7 % | 75.7 % | 75.7 % | 0 | 0 |
| NUDT 101 | 600 | 84.2 % | 84.2 % | 84.8 % | 85.8 % | 6 | 0 |

Decision on the 6 standard queries (bar 8 / judge 0.9 without override): official TRY identical (50.40 / 52.22, 0 wrong); NUDT 101
54.60 -> 55.20 for both rules (one more photo answered right, 0 wrong).

**Read:** on official, DINO never drops the right place (and places 6-10 add nothing over the first 5). On NUDT, DINO occasionally (1 % of
queries) ranks the right place below places the coarse order keeps; a fixed coarse share recovers it at no cost. The effect is one photo on a
design set: not adopted, not a claim. If pursued, it must be checked on a fresh draw (the 808 cache holds the full K = 50 shortlist, so only
the 3 coarse places per query would need fine matching).

## 7. Recommendation

1. **PC: V5 (relative inliers + scale augmentation, no override)** — +2.40 [+0.52, +4.64] on fresh NUDT 808 with 0 wrong at every inlier scale
   (x0.5-x2), where the T8 rule (J0) makes up to 8 wrong answers at x2 and today's bars up to 12. J0 scores 0.3 more at x1 but is not
   scale-safe. If a simpler rule is preferred: J2 (the same judge, no override, cut 0.98): +2.12, also 0 wrong at every scale.
2. **Jetson: V5 or J1 with early exit** — no proven score gain on 808 (+0.32), but 0 wrong at every scale vs 2-9 for today's rule, and ~2.2
   LoFTR queries per photo instead of 6.4 (~14 s -> ~5 s modelled).
3. **Remove the 12-inlier override everywhere.**
4. Before shipping any NUDT-trained judge: answer the open question (satellite-only training constraint?).
5. The remaining losses are upstream: 9-17 photos per 300 never retrieved on NUDT; official "retrieved but rejected"; T9's metadata / blur
   failures.
