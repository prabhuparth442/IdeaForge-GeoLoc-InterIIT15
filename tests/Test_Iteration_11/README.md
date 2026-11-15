# Test_Iteration_11: a candidate judge trained on satellite imagery only

**Why:** the team's rule (2026-11-14): learned components are trained on satellite imagery only; thermal photos (NUDT, official) may set cut
thresholds and test. Every judge of T7-T10 was trained on NUDT thermal photos, so none can ship.

**Result: the satellite-trained judge ranks right vs wrong almost as well as the NUDT-trained one (AUC 0.998 / 0.996-0.997 vs 0.9996 / 0.999)
but is badly calibrated on thermal photos. With a cut strict enough for 0 wrong answers on the design sets it gains nothing, and on the fresh
pre-registered NUDT 909 draw it still makes 1 wrong answer (base: 2 PC / 4 Jetson). Not a shippable score gain; it is a safety gain on the
Jetson (S1z: 0 wrong at every inlier scale, +0.16) and robust to inlier inflation (base: 14-17 wrong at x2).**

## How (notebooks)

| Notebook | What |
|---|---|
| `Sat_Photo_Dump` (+ `run_strength.sh`) | block `train/checks/sat_photos.py`: drone-view photos rendered from Cartosat scene B (another pass) and Pleiades Neo (another sensor), never from the map (scene A), > 1.5 km from every official photo; the official camera incl. distortion, the pipeline's 180 deg yaw / x1.04 height convention, random heading with a metadata heading error, height 100-350 m; thermal-like changes (polarity flip, tone curve, random non-monotonic grey-level map, blur, noise). Geometry check: a photo rendered from scene A and un-rendered by the pipeline correlates 0.93 / 0.87 with the map (0.15 / 0.22 shifted 15 m); the pipeline finds all 6 smoke photos within 0.6 sigma. Strength calibrated so the right place's inliers match thermal photos (strength 2-3: median 50-70, 10th percentile 6-12; official 31 / 8, NUDT 101 55 / 15). 378 photos, strength 1.5-3.5, 0 failures |
| `It1_Sat_Judge` | S1 (T7 architecture) and S5 (relative inliers + scale augmentation, T10 V5), trained on satellite-made candidates only. Learn-then-Test on NUDT found **no** cut up to 0.99 (wrong rate at P 0.95-0.98: 84 % NUDT, 26 % official TRY): over-confident on thermal. Finer grid: LTT cut 0.998 = PC +2.66 on official TRY but 1-2 wrong on design sets; zero-wrong cut 0.9995 = 0 wrong everywhere but PC +0.19 / -0.78, Jetson -0.88 / -2.66 on TRY |
| `It2_Preregister_and_Final_909` | `results/preregistered_t11.json` (19:18:24, before 909 was read), then NUDT fresh 909 (300 photos, every earlier photo excluded) scored once |

## Final: NUDT 909, scored once

| Rule | PC score / wrong | PC gain | Jetson score / wrong | Jetson gain | wrong over x0.5 ... x2 (PC; Jetson) |
|---|---|---|---|---|---|
| base | 54.24 / 2 | — | 52.72 / 4 | — | 0-17; 0-14 |
| **S5z** (primary, zero-wrong cut) | 53.52 / **1** | -0.72 [-2.44, +1.08] | 53.28 / **1** | +0.56 [-0.92, +2.28] | 0-1; 0-1 |
| S1z | 53.52 / 1 | -0.72 | **52.88 / 0** | +0.16 [-1.56, +2.04] | 1; **0 at every scale** |
| S5l (cut 0.998 / 0.995) | 54.32 / 2 | +0.08 | 53.96 / 2 | +1.24 [-0.12, +2.84] | 2; 2 |
| S1l (cut 0.998) | 54.72 / 2 | +0.48 | 53.28 / 2 | +0.56 | 2; 2 |

Primary criterion (0 wrong, gain >= 0, interval above 0): **not met** on either machine.

## What it means

1. Satellite-made training data teaches the *ranking* (AUC 0.996-0.998 on thermal photos) but not the *confidence*: thermal look-alikes get
   P 0.95-0.99. A cut set on thermal data (allowed) has to sit at 0.999+, where it keeps few extra answers.
2. The judge's value without thermal training is **safety**: fewer wrong answers than the bars and no growth under inlier inflation. The
   Jetson S1z rule had 0 wrong at every scale on 909 at the same score as today's rule, with ~2.4 instead of ~6.4 queries per photo.
3. To gain score from satellite-only training, the training photos must look more like thermal photos (the gap is in calibration, not
   ranking): better hand-made thermal-like rendering, candidates from more look-alike-rich ground (cities), or more satellite scenes. A
   learned visible-to-thermal translation would need thermal targets to train, so it falls under the satellite-only rule.

## It3: valid methods (no thermal training) that actually improve — scored on sets their design never used (`It3_Valid_Methods`)

Test sets: official TEST, NUDT 707, 808, 909 (1,094 photos; the DINO-z veto excludes official TEST, seen when it was set). Pooled paired
bootstrap vs today's rule; wrong answers summed.

| machine | method | pooled gain | wrong (base) | queries | inlier x2: wrong (base) | altered photos: wrong (base) | verdict |
|---|---|---|---|---|---|---|---|
| PC | **two-tier rule** (T8, 3 numbers) | **+1.91 [+1.04, +2.85]** | 3 (10) | 8.4 | 31 (48) | 3 (12) | **improves score and safety** |
| PC | **DINO-z veto** (T6, 2 numbers) | **+0.65 [+0.08, +1.39]** | 5 (10) | 8.2 | 27 (48) | 4 (12) | **improves (smaller)** |
| PC | satellite judge S5z | +0.07 [-0.94, +1.08] | 1 (10) | 9.4 | 0-2 at every scale | — | safety only |
| PC | satellite judge S1z | -0.41 | 2 (10) | 9.6 | — | — | safety only |
| Jetson | two-tier rule | +0.46 [-0.04, +0.99] | 8 (8) | 6.4 | — | — | not proven |
| Jetson | **two-tier + early exit 20** | +0.31 [-0.22, +0.84] | 8 (8) | **2.8** | 25 (38) | 5 (8) | **speed (6.5 -> 2.8 queries) at no score cost** |
| Jetson | early exit by inliers >= 20 | -0.15 [-0.29, -0.04] | 8 (8) | 2.9 | — | — | speed at a tiny cost |
| Jetson | DINO-z veto | +0.16 | 6 (8) | 6.4 | — | — | not proven |
| Jetson | satellite judge S1z | -1.33 [-2.26, -0.38] | **0** (8) | 2.7 | 0 at every scale | — | safety, costs score |
| Jetson | satellite judge S5z | -0.11 | 2 (8) | 2.4 | 0-1 | — | safety + speed |

None is input-invariant: the hand rules' wrong answers still grow with inlier inflation (two-tier 3 -> 31 at x2, base 10 -> 48), only fewer;
the satellite judges are flat (0-2) but gain no score.
