# Test_Iteration_6: where the marks still go, and what recovers them

2026-11-13, one overnight session. Brief: [`PROMPT.md`](PROMPT.md). Rules: `../../DESIGN.md` (notebook is the face, blocks do the work).
Nothing in `src/` was edited (no `changed_src/` needed); new blocks are in `train/checks/`. Nothing committed.

## The short answer

1. **Today's best rule is not safe on fresh photos.** On 500 NUDT photos that no earlier test had used, the agreement + escalation rule made
   **4 wrong answers on one fresh sample (404) and 1 on another (505)**. The earlier "0 wrong on every set" came from samples drawn from the same
   500-photo pool that the thresholds were tuned on. All wrong answers are on the NUDT **city** flights (look-alike buildings); the official set
   has none.
2. **What holds on every set, including the blind one:** the 27-query escalation at bar 11, and a constant **attitude correction** (official
   camera). Agreement is a large gain on official (+3.9) but adds wrong answers on NUDT city photos.
3. **For the Jetson** (the submitted code loses most of its marks to the 22 s budget): add agreement (each agreeing query >= 4 inliers, free:
   no extra query), a DINO-z veto (free) and the attitude correction to the submitted code, keep its query count and its second-pass order.
   Official: 50.28 -> **53.10, +2.82 [+1.68, +4.02]**, 0 wrong; >= 0 on every NUDT set; same number of queries per photo.
4. A learned verifier gains +3.6 on official but is **not adopted**: with only 5 wrong answers in 1,379 photos, no learned cut stayed at 0 wrong
   on every held-out set.

## Setup on this machine

- Repo `~/GeoLoc` (commit f914ca5, branch `prabhuparth/dev`); venv `~/Ideaforge/GeoLoc/.venv-geoloc`; RTX 4070 Laptop 8 GB.
- DINO weights were missing and were fetched from the Hugging Face hub (`timm/vit_small_patch14_dinov2.lvd142m`).
- No Jetson and no `Submission_Files` here: Jetson time is a **model** (2.2 s per query at M = 5, from the submitted run: 13.06 s mean, ~6 queries).
- **Hardware warning:** the laptop (i9-14900HX) logged machine-check errors and segfaults on CPU 10/11 and crashed once. All runs were pinned off
  that core; no new error was logged during the runs; every replay matched earlier results exactly (below). A BIOS/microcode update is advised.

## Method: one GPU run, then everything offline

`notebooks/Rich_Dump_Run.ipynb` (block `train/checks/rich_dump.py`) runs **all 27 wide queries** on every photo and stores every candidate's
**raw LoFTR matches**, the photo's attitude and each query's seconds (~120 kB per photo). Today's pipeline is a subset of that, so it is replayed
exactly (`train/checks/replay.py`); so are other query policies (`query_policy.py`), verifiers (`verifier_features.py`, `verifier.py`), the
position correction (`pose_bias.py`) and a second matcher (`xoftr_batched.py`).

**Replay check:** official 379 photos give 53.54 with the same points as the earlier agreement-pipeline run on **379 of 379** photos; NUDT 202 gives
57.36, same on 200 of 200; NUDT 101 55.56 as in Decision_Report.

| Set | Photos | Role | Note |
|---|---|---|---|
| official | 379 | choosing | all photos inside the map |
| NUDT 101 | 300 | choosing | 500-photo pool (as Test_Iteration_3) |
| NUDT 202 | 200 | report, then seen | same photos as Test_Iteration_3; informed Iteration 2's "late" mode |
| NUDT 404 | 200 | report, then seen | **fresh:** outside the 500-photo pool; informed Iteration 8 |
| NUDT 505 | 300 | **blind** | fresh, outside the pool and 404; scored once, after `results/frozen_rules.json` was written (05:30) |
| (NUDT 303) | 83 | invalid | overlapped seed 202 (the pool was used up); moved to `results/invalid/` |

Scores are /60. Gains are paired bootstrap means with 95 % intervals (4000 resamples). "Wrong" = answers beyond 8 sigma (-20 each).
PC time per photo: today's rule 2.6-2.9 s (official 2.68 s, ~0.33 s per query).

## Every idea tried

| # | Idea (loss targeted) | Test | Result | Wrong | Time | Verdict |
|---|---|---|---|---|---|---|
| 1 | Query budget for the Jetson: cap, greedy order of extra queries, early exit, M 5 vs 10 (Jetson time cuts) | `It1_Replay_and_Query_Budget` | Greedy order: official +3.42 vs Jetson replay, but **blind NUDT 505 -0.64 [-1.44, -0.04]** (it skips 5 photos the default order finds). Early exit at 15-30 inliers halves the queries (official 6.3 -> 3.1) for -0.2. M = 5 + agreement: 2 wrong on NUDT 101. | greedy 0 / M5+agree 2 | early exit: about -50 % queries | Greedy order **killed** (overfit). Early exit **keep as an option** if time is short. M = 5 needs stricter agreement (see 8). |
| 2 | Learned verifier on fit features: hull, residual, fit angle/scale, ratio, DINO z, rank, agreement (decision) | `It2_Learned_Verifier` | "Late" mode (only photos today skips): official +3.61 [+2.28, +5.07] (out-of-fold), NUDT 101 +0.20, 202 0, 404 +0.36. Early "add" mode: 1 wrong on NUDT 202. Leave-one-set-out as a veto: NUDT 101 held out -> 1 wrong. | 0-1 depending on split | none (one RANSAC re-fit) | **Needs more data.** Not adopted. DINO z, hull and agreement separate right from wrong the same way on every set (AUC table). |
| 3 | Constant attitude bias in the final position; learned refinement (precision at height) | `It3_Attitude_Bias` | Fitted angle 0.98 deg forward (as Patch v0.5.6). Held out by flight: official **+0.82 [+0.38, +1.33]**, within 1 sigma 95.1 -> 98.8 %, at 250-400 m 92.4 -> 97.9 %. NUDT has no attitude: own-fit +0.08 / +0.12 / +0.24. GBM refinement +0.63 official, -0.12 NUDT 404. | 0 | none | **Keep** (official camera). Learned refinement: no better than the 2-term model, **killed**. |
| 4 | Agreement inside the escalation between queries >= 5-10 deg apart (decision) | `It4_Spaced_Agreement` | Every setting that adds answers makes 1-4 wrong answers on NUDT; the only safe one adds nothing. | 1-4 | none | **Killed.** |
| 5 | Headroom on photos today skips (cheap test for a matcher / fine-tuning) | `It5_Headroom` | Official: **all 34 skipped photos have the right place** among the candidates (median 4 inliers, wrong places median 0). NUDT: 21 of 36 skipped have **no right candidate at all** (recall). NUDT false places reach 25-44 inliers (official max 14). | - | - | Diagnostic: official loss = fine matching + decision; NUDT loss = recall. A better matcher could help official only. |
| 6 | XoFTR as a second opinion on the photos the 6 queries do not answer (fine matching) | `It6_XoFTR_Second_Opinion` | Answer when LoFTR and XoFTR 27-query choices agree within 1 sigma: official +2.72 [+1.55, +3.99], NUDT 101 +0.40, 202 +0.30, 404 +0.30; 0 extra wrong. XoFTR on its own bar 8: 4 wrong on NUDT 101. | 0 (agree mode) | +27 XoFTR queries (9 s PC) on ~10 % of photos | **Promising, not for the Jetson** (second network, 27 more queries; memory already at 7.1 of 7.4 GiB). Not confirmed on the blind set (not run there). |
| 7 | All Jetson pieces combined at M = 5 / 10 (exploration) | `It7_Jetson_Policy` | Superseded by 8 (it still had the greedy order and the verifier). | - | - | - |
| 8 | Fresh-photo check; DINO-z veto; blind confirmation (decision) | `It8_Fresh_Photos_and_Z_Veto` | Today's rule: **4 wrong on fresh NUDT 404**, 1 on blind 505. Veto "answer only if DINO z >= 1.3 or >= 15 inliers": 404 4 -> 0 wrong (+2.58), official -0.16, blind 505 **no change** (its wrong answer has the right place absent and passes the veto). Stricter bar 10 + 3 agreeing queries: 0 wrong on the seen sets but official -3.8, blind still 1 wrong. | blind: every rule 1 (the same photo) | none | Veto: **keep as a free safety net** (no cost anywhere, removes the 404 failures; gain on unseen photos not shown). |

**The final comparison** (`results/final_summary_five_sets.csv`, score / wrong / mean queries; all at the PC except the Jetson rows, which are
replayed with M = 5 and a 10-query cap):

| Policy | official | NUDT 101 | NUDT 202 | NUDT 404 | NUDT 505 (blind) |
|---|---|---|---|---|---|
| A: bar 8 only, 6 queries | 49.01 / 0 | 53.00 / 0 | 55.38 / 0 | 53.34 / **2** | 53.04 / 0 |
| B: + agreement | 52.91 / 0 | 54.04 / 0 | 55.68 / 0 | 52.83 / **4** | 53.64 / **1** |
| C: + escalation 27 @ 11 (= today's best) | 53.48 / 0 | 55.56 / 0 | 57.36 / 0 | 53.43 / **4** | 55.04 / **1** |
| D: C + z-veto | 53.32 / 0 | 55.56 / 0 | 57.36 / 0 | 56.01 / 0 | 55.04 / **1** |
| E: **Jetson today** (M 5, 2nd pass @ 8, cap 10) | 50.28 / 0 | 54.24 / 0 | 55.26 / 0 | 55.44 / 0 | 54.68 / **1** |
| G: E + agreement (>= 4 inliers each) + z-veto | 52.27 / 0 | 55.16 / 0 | 55.56 / 0 | 56.13 / 0 | 54.76 / **1** |
| G + attitude correction (official camera) | **53.10 / 0** | - | - | - | - |

Gains, paired: escalation over B is positive on all five sets (blind +1.40 [+0.40, +2.60]). **G over E:** official +1.99 [+0.95, +3.13],
NUDT 101 +0.92 [+0.20, +1.76], 202 +0.30 [0, +0.90], 404 +0.69 [0, +1.59], blind 505 +0.08 [0, +0.24]; G's only wrong answer (blind photo 889)
is also E's. Caveat: G is the frozen J_M5 policy with the greedy order swapped back to the submitted second-pass order after the blind check
(removing a tuned piece, not adding one).
**G + attitude over E (official): +2.82 [+1.68, +4.02].** Queries per photo: E 6.3-6.6, G 6.2-6.5 (the same Jetson time, ~14 s mean modelled).

## Recommendation

**Put in the submission (Main_Pipeline_Headless), in this order of value per hour:**

1. **Attitude correction** of the final position: subtract `h * R(yaw) (a_f, a_r) + b` with a_f = tan(0.99 deg), a_r = tan(0.11 deg), b = (-0.12, -1.46) m
   (fit on all official answers; refit once on the final code's answers, `train/checks/pose_bias.py`). +0.8 on official, no time cost, and it
   grows with height: at 3000 m sigma is ~39 m and a 0.98 deg angle is 51 m (1.3 sigma), so a constant bias would cost far more on the
   high-altitude evaluation photos. Check the sign convention against the submission's own nadir conversion before shipping.
2. **Agreement with >= 4 inliers per agreeing query** after the 6 standard queries (M = 5 as on the Jetson; 3 inliers made 2 wrong answers at M = 5).
   No extra query.
3. **DINO-z veto** (z >= 1.3 or >= 15 inliers): free insurance against the NUDT city failure mode; cost -0.16 on official.
4. Keep the existing second-pass order and bar 8 there; the per-photo budget still decides.

Expected: official 50.28 -> 53.10 (+2.82 [+1.68, +4.02]), 0 wrong; NUDT 0 to +0.9; Jetson time unchanged (no extra query; DINO z and the
agreement count are already computed). The remaining Jetson budget loss is the slow FFT on the Orin Nano (cuFFT fails there), not the decision.

**Do not put in:** the greedy query order (blind -0.64), the learned verifier (not safe across sets), spaced agreement, XoFTR (second network).

## Still open

- **Wrong answers on NUDT city scenes** (photo 889 of the blind set: right place absent from all 27 queries, a false place reaches 6-11 inliers).
  No decision rule tested removes it without losing official marks. This is recall (coarse / heading range), not the decision.
- **Official skipped photos (34):** the right place is always present with 0-10 inliers. A second matcher (XoFTR agreement, +2.7 official) or a
  fine-tuned matcher could recover them; both cost Jetson time or memory.
- **No Jetson here:** time is modelled. M = 10 (better than M = 5 here) may or may not fit the budget; measure on the Jetson.
- **High altitude (> 351 m) is untested** (simulator not run this iteration).
- The z-veto and the Jetson policy G were confirmed on the blind set only as "no harm" (+0.08); their gains come from seen sets.

## Files

- Notion: summary page "Test_Iteration_6 summary: fresh photos, Jetson decision, attitude correction (2026-11-13)"
  (https://app.notion.com/p/41742f5fcd9481f6a9e1e499674dae24) and one "T6 Iteration ..." page per iteration, under InterIIT_UAV_GeoLoc.

- Notebooks (each with `_executed_all.ipynb`): `It1` to `It8` in `notebooks/`; dump runs `Rich_Dump_Run_executed_<set>.ipynb`.
- Blocks: `train/checks/rich_dump.py`, `replay.py`, `query_policy.py`, `verifier_features.py`, `verifier.py`, `pose_bias.py`, `xoftr_batched.py`.
- Unit tests: `python tests/pipeline/test_iteration6_blocks.py` (6 pass, made-up data, no GPU).
- Results: `results/rich_dump/<set>/photo_<id>.npz`; tables `results/it*.csv`, `results/final_summary_five_sets.csv`, `results/frozen_rules.json`.
- Drivers for the GPU runs: `run_dumps.sh`, `run_dumps_phase2.sh` to `phase5.sh` (phase 2 failed on every photo: `KeyError('batch_indexes')`, the
  XoFTR wrapper in `src/` predates the batched LoFTR plumbing; fixed by `train/checks/xoftr_batched.py`, rerun as phase 4).
