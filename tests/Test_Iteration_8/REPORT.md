# Test_Iteration_8 — Full Report

Branch `palash/dev`. Builds on T6 (rich dumps: raw LoFTR matches of every candidate of all 27 queries) and T7 (signal tables, candidate judge).
`src/` is not edited.

## 1. Why this iteration

The team's review of T7: the judge was trained on the data it was meant to improve, so it fits that data, and the final data should not be
expected to be the same; it performed poorly when trained on NUDT and tested on official. Asked for:
1. check every method proposed so far the same way: is it fitted to the data, and is the gain real or just more answers with some wrong ones;
2. build new methods on NUDT;
3. split official into a TRY and a TEST half so official is not fitted again.

## 2. Protocol

- **Design data:** NUDT 101, 202, 404, 505, 606 (1,300 photos). Every rule and model is built here. (606 was first a check set for the two-tier
  rule; the Jetson rule chosen without it made 2 wrong answers on 606, the same as the Jetson today, so 606 joined the design data and the
  selection was redone; this is recorded in It3.)
- **Official split by place** (`train/checks/holdout.py`, `results/official_split.csv`): the 379 official photos form 12 spatial blocks (k-means on
  the true position, seed 0); the blocks are shared out so both halves have the same number of photos per height band (100-125, 125-175,
  175-275, 275-351 m). TRY 185 photos (blocks 0, 5, 6, 7, 9, 10), TEST 194. No outcome was used. The two halves still see neighbouring ground
  (high photos have wide footprints) and the same map.
- **Locked loader:** `load_t8_table(locked=True)` drops TEST; only It4 unlocks it.
- **Blind NUDT 707:** 300 fresh photos (pool 2400; the 500-photo pool and every 404 / 505 / 606 photo excluded), dumped on the GPU (~50 min)
  before any T8 rule; its signal table was built without scoring (It1).
- **Pre-registration** (`results/preregistered_t8.json`, written 01:47, sha256 3911f559...): rules, numbers, the attitude model and the success
  criterion (a finalist counts as a gain if on BOTH final sets it has no more wrong answers than its base and a gain >= 0, and on at least one a
  paired-bootstrap interval above 0). It4 then scored everything once.
- **Disclosure:** before the lock, one exploration printout showed official TEST scores of three NUDT-only judge feature sets (T7's 19 features,
  inliers + z, a 7-feature subset). The finalist judge is T7's unchanged architecture, fixed before T8; nothing was chosen from that printout.
- **"Real gain or more answers?"** Two tools (`holdout.py`): `versus_frontier` compares a method with the plain inlier bar that answers the same
  number of photos (on that curve = only answering more; above it = judging better); `answer_change` splits a gain into answers added (right /
  wrong), removed, and wrong answers.

## 3. Ideas and verdicts

| # | Idea | Built on | Result | Verdict | Where |
|---|---|---|---|---|---|
| 1 | Bar curve as the reference | — | official TRY: bar 4 = 54.81 / **0 wrong** (+3.4 over bar 8); NUDT: bar 4 = 6-15 wrong per set | the decision problem differs by data set | It2 |
| 2 | T7 judge **with** official in training (frozen T7) | official + NUDT | official TRY 56.95 (in-sample) | **fitted: ~3 points of official are memory** | It2 |
| 3 | T7 judge **NUDT only**, cut on NUDT | NUDT | TRY 54.03 (+0.19 vs same-answers bar); final: official TEST +5.13, 707 +1.40, 0 wrong | **gain, holds on both unseen sets** | It2, It4 |
| 4 | Early exit of the judge (P >= 0.98) on the Jetson | NUDT | TRY -1.23 vs no exit; final TEST -0.12, 707 0; 2-3 queries | keep (time) | It2, It4 |
| 5 | Min 4 inliers for an answer | — | identical decisions to min 3 on every seen set | no effect | It2 |
| 6 | T6 DINO-z veto (z >= 1.3 or >= 15 inliers) | NUDT 404 | removes 1 wrong on 404 / 505 / 606, 0 cost on TRY | safety net only | It2 |
| 7 | Null-calibrated cut: compare the choice with the strongest far candidate of the other photos of the same set (label-free, adapts to how look-alike-rich a map is) | — | inlier score: 0 wrong but below bar 8; judge score: +4.5 TRY but 1-2 wrong per NUDT set | dropped | It2 |
| 8 | Other position estimates (translation fit, inlier-weighted mean over queries, largest hull) | — | the similarity fit is best everywhere (TRY 9.81 / 10 per answer) | no change | It2 |
| 9 | **Two-tier rule**: >= i_hi inliers, or >= a_low with DINO z >= z0 and zgap >= g0 | NUDT (grid of 576) | PC (5, 2.0, 0, 12): TEST +3.77, 707 +0.60 / 1w (base 3w); Jetson (5, 2.0, 0, 10): TEST +1.92 [-0.31, +4.27], 707 -0.20 | PC weaker than 3; Jetson fails | It3, It4 |
| 10 | Two-tier with a_low 4 (ties with 5 on NUDT, +1.8 on TRY) | NUDT + TRY | TEST +4.39, 707 -0.20 / 3w | no: the TRY-chosen variant fails on the blind set | It3, It4 |
| 11 | Two-tier early exit at >= 20 inliers (Jetson) | NUDT + TRY | as 9 at 2.5-3.7 queries | no (as 9) | It3, It4 |
| 12 | Attitude correction, angle only (1.03 deg forward), fitted on TRY | TRY | held out by place inside TRY: 9.78 -> 9.95 per answer; TEST +0.12 to +0.62, intervals through 0 | not proven on TEST | It3, It4 |

## 4. Why the bars, the judge and the two-tier rule behave this way

- On official almost every choice with >= 4 inliers is right; official wrong places stay below 15 inliers and rarely compete. On NUDT, city
  look-alikes reach 8-11 inliers and the right place is missing from the candidates in about half of the skipped photos (T6 It5). So "answer
  more" is nearly free on official and costly on NUDT, and any method's official gain is mostly answering more.
- At 5-11 inliers, wrong choices look alike on both sets (median DINO z 1.2, negative zgap, larger leftover angle and shift); right ones have z
  >= ~2. That is the two-tier rule's logic, and it holds on official TEST. On the blind NUDT draw one look-alike passes it (PC); the judge,
  which also sees inlier hull, coverage and fit plausibility, rejects it.
- The judge is trained on every fine-matched candidate (tens of thousands of right and wrong candidates), not on the few wrong answers, and its
  probability can only rise with inliers, z, hull and coverage. That is why it transfers from NUDT to official when its cut is chosen on NUDT.
  What did not transfer in T6 was a verifier trained on the chosen candidate only.

## 5. Time on the Jetson (model: 2.2 s per LoFTR query at M = 5)

| Rule | queries / photo | s / photo | class |
|---|---|---|---|
| Jetson today | 6.3-6.7 | ~14-15 | slow |
| judge, NUDT only | 6.4-6.5 | ~14 | slow |
| **judge, NUDT only + early exit** | **2.0-2.8** | **~4.5-6.2** | **viable** |

The judge's signals cost ~0.1 s per photo (T7).

## 6. Recommendation

1. Adopt the **T7 judge trained on NUDT only** (cut 0.9 PC / 0.925 Jetson from Learn-then-Test on NUDT, floor 0.9, extra-query cut 0.95,
   override 12, min 3 inliers), with the **early exit** on the Jetson. On sets nothing was tuned on: official TEST +3.5 (Jetson) / +5.1 (PC),
   NUDT 707 +0.8 / +1.4, **0 wrong answers on both**.
2. Train only on NUDT (or other non-official data); use official only to check.
3. Do not adopt the two-tier rule (blind draw: no gain on the Jetson, 1 wrong on the PC) or the attitude correction (not proven).
4. Open: heights above 351 m; a second official-like area to check on; a fresh official-like blind set.

## 7. Problems hit

- The laptop crashed once after It4 (no job was running; every file was intact). Unit tests were re-run after it.
- A scratch script wrote two files into the repo root (it changed into the repo folder); they were moved out before anything was committed.
- `candidate_judge.decide` now also returns the answer's position; made-up test tables without positions get NaN (T7 tests pass).

## 8. Sources

- Angelopoulos et al., *Learn then Test*, arXiv 2110.01052 (the cut rule, unchanged from T7).
- Bates, Candès, Lei, Romano, Sesia, *Testing for outliers with conformal p-values*, Annals of Statistics 2023 (the null-calibrated cut, idea 7).
- Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography 2017 (split
  by place, not by photo).
- Nosek et al., *The preregistration revolution*, PNAS 2018 (write the analysis down before seeing the test data).

## 9. Reproducing

```bash
python tests/pipeline/test_iteration8_blocks.py
bash tests/Test_Iteration_8/run_blind_dump.sh          # GPU, ~50 min
# then run notebooks It1, It2, It3 (It3 refuses to overwrite results/preregistered_t8.json), It4
```
