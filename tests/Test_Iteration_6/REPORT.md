# Test_Iteration_6 — Full Report

**Date:** 2026-11-13 (one overnight session)
**Branch:** `palash/dev`
**Brief:** [`PROMPT.md`](PROMPT.md)
**Short summary and file map:** [`README.md`](README.md)
**Notion:** summary page "Test_Iteration_6 summary: fresh photos, Jetson decision, attitude correction (2026-11-13)" under InterIIT_UAV_GeoLoc,
plus one "T6 Iteration …" page per iteration.

---

## 1. Goal

The brief asked three things:
- Find where the geo-localisation pipeline still loses marks.
- Try classical and learned fixes, over as many iterations as needed.
- Keep only what holds up.

A change counts as an improvement only if it:
1. has a logical reason for working;
2. holds on official **and** NUDT, with at least 2 NUDT samples, one of them unseen;
3. has a paired-bootstrap 95 % interval above 0;
4. keeps **0 wrong answers** (beyond 8 σ, −20 each) on every set it is checked on;
5. stays inside the Jetson Orin Nano budget of 22 s per photo.

**Scoring per photo:**

| Error | Points |
|---|---|
| ≤ 1 σ | +10 |
| ≤ 2 σ | +6 |
| ≤ 4 σ | +3 |
| ≤ 8 σ | 0 |
| worse than 8 σ | −20 |
| skip | 0 |

Score out of 60 = 6 × mean points per photo. σ ≈ 8.1–9.2 m on the dev photos. One wrong answer cancels two perfect ones.

## 2. Pipeline under test

```
thermal photo
  → north-up canvas
  → edge-direction ZNCC by FFT over a 5.5 km disc
  → top K = 50
  → DINO re-rank
  → LoFTR-MINIMA + RANSAC similarity on the top M per query (PC M = 10, Jetson M = 5)
  → answer or skip
```

**Queries:**
- Standard set: 6 queries (headings −5°, 0°, +5° × scales 1.0, 1.05).
- Wide set: 27 queries (−10° … +10° in steps of 2.5° × scales 0.95, 1.0, 1.05).

**Decision rule at the start of this iteration ("today's rule"):** answer when either:
- the best candidate has ≥ 8 inliers, **or**
- ≥ 2 of the 6 queries' best candidates agree within 1 σ, each with ≥ 3 inliers.

Otherwise escalate to the 27 queries and answer at ≥ 11 inliers.

**The Jetson as submitted** is simpler:
- M = 5.
- Second pass at bar 8.
- At most 10 queries in the budget.

## 3. Setup

| Item | Value |
|---|---|
| Machine | i9-14900HX, RTX 4070 Laptop 8 GB |
| Venv | `~/Ideaforge/GeoLoc/.venv-geoloc` |
| Code base | commit f914ca5 (`prabhuparth/dev`), which these blocks import (see §11) |
| DINO weights | fetched from the Hugging Face hub (`timm/vit_small_patch14_dinov2.lvd142m`) |
| Jetson | not available here. Time is **modelled**: 2.2 s per query at M = 5, taken from the submitted run (13.06 s mean, ~6 queries). |
| Hardware fault | The CPU logs machine-check errors on CPU 10/11, which caused one crash and segfaults. Every run was pinned with `taskset -c 0-9,12-31`. No new error appeared, and every replay matched earlier results exactly. |

## 4. Method: one GPU run, then everything offline

Earlier iterations re-ran the GPU pipeline for every idea, which was slow and made it easy to tune on noise. This iteration ran the GPU **once** per
photo set and stored everything a decision could need. Every idea was then replayed offline, on CPU, in seconds.

**The rich dump** (`train/checks/rich_dump.py`, driven by `notebooks/Rich_Dump_Run.ipynb`) stores, for each photo:
- all 27 wide queries, each with its 10 fine-matched candidates (coarse peak, DINO z, inliers, shift);
- the **raw LoFTR matches** of every candidate (points ×64 as uint16, confidence ×65535 as uint16, kept above 0.15);
- the valid-pixel mask, the photo's attitude (yaw/pitch/roll, height) and the seconds spent per query.

That is about 120 kB per photo.

**Replay** (`train/checks/replay.py`): today's rule is a subset of the dump, so it can be recomputed exactly. As a check:

| Set | Replayed score | Matches earlier run |
|---|---|---|
| official (379) | 53.54 | 379 of 379 photos |
| NUDT 202 (200) | 57.36 | 200 of 200 photos |
| NUDT 101 (300) | 55.56 | Decision_Report value |

Because the raw matches are stored, new RANSAC fits, verifier features and position corrections can be computed without the GPU.

**Photo sets.** 1,379 photos were dumped. The 83-photo NUDT 303 dump was discarded because it overlapped seed 202.

| Set | Photos | Role |
|---|---|---|
| official | 379 | used to choose (all official photos inside the map) |
| NUDT 101 | 300 | used to choose (from the old 500-photo pool) |
| NUDT 202 | 200 | report set; it informed Iteration 2's "late" mode, so it is seen for that |
| NUDT 404 | 200 | **fresh**: drawn from outside the 500-photo pool; it informed Iteration 8 |
| NUDT 505 | 300 | **blind**: fresh and outside 404 too; scored **once**, after the rules were frozen |

The rules were frozen in `results/frozen_rules.json` at 05:30, before NUDT 505 was loaded by anything.

**Statistics.**
- Gains are paired-bootstrap means with 95 % intervals (4,000 resamples), in /60 units.
- "Wrong" = beyond 8 σ.
- PC time per photo for today's rule: 2.6–2.9 s.

## 5. The iterations

### Iteration 1 — Query budget for the Jetson
**Notebook:** `It1_Replay_and_Query_Budget`. **Loss targeted:** the Jetson's 22 s budget. 66 of its 77 skips were photos cut off by the budget.

**What was tested:**
- a cap on the number of queries;
- a *greedy* order for the extra queries (the order that found the most right answers first on the choosing sets);
- early exit at 15–30 inliers;
- M = 5 vs M = 10.

**Results:**

| Policy (official) | Score | Mean queries |
|---|---|---|
| bar 8 only, 6 queries | 49.01 | 6.0 |
| Jetson today, capped | 50.75 | 6.9 |
| bar or agreement, 6 queries | 52.91 | 6.0 |
| today's rule, capped at 12 | 53.32 | 6.6 |
| today's rule, no cap | 53.48 | 7.9 |

- **Greedy order:** +3.42 over the Jetson replay on official. But on the blind NUDT 505 set it scored **−0.64 [−1.44, −0.04]**, because it skips 5 photos the default order finds. Overfit → **killed**.
- **Early exit:** halves the queries (official 6.3 → 3.1) for about −0.2. → **kept as an option** if Jetson time gets tight.
- **M = 5 with agreement at ≥ 3 inliers per query:** 2 wrong answers on NUDT 101. Agreement at M = 5 needs ≥ 4 inliers per query (used in Iteration 8).

### Iteration 2 — Learned verifier
**Notebook:** `It2_Learned_Verifier`. **Loss targeted:** the answer-or-skip decision.

**Features** (block `verifier_features.py`), refit from the raw matches:
- n_pairs, n_inliers, inlier ratio;
- RANSAC fit scale and angle deviation, residual;
- convex-hull area and spread of the inliers, mean inlier confidence;
- DINO z, rank, agreement count, runner-up distance;
- query heading and scale.

Flight height and map level were left out on purpose, because the evaluation heights differ from the dev photos.

**Models** (`verifier.py`): logistic regression and shallow gradient boosting. Folds were grouped by photo, so a photo's two rows are never split across folds.

**Ways to apply it:**
- *replace*: the verifier replaces today's rule;
- *add*: the verifier can also answer at the 6-query step;
- *late*: today's full rule runs first, and the verifier only answers photos that rule would skip.

**Results (logistic):**

| Mode | official (OOF) | NUDT 101 (OOF) | NUDT 202 | NUDT 404 |
|---|---|---|---|---|
| replace | +3.07 | −0.52 | −2.70, **1 wrong** | +2.67 |
| add | +3.48 | +0.20 | −0.90, **1 wrong** | +0.36 |
| late | **+3.61 [+2.28, +5.07]** | +0.20 | 0 | +0.36 |

- In *add* mode, photo 377 of NUDT 202 went wrong: the verifier answered a weak 6-query choice and so skipped the escalation, which would have found the right place. That led to *late* mode, so NUDT 202 counts as seen for it.
- Leave-one-set-out as a veto: holding out NUDT 101 gave 1 wrong answer.
- There are only 5 wrong answers in 1,379 photos, so no learned cut can be shown to stay at 0 wrong.
- DINO z, hull area and agreement separate right from wrong the same way on every set (AUC table in the notebook). That finding fed Iteration 8.

**Verdict:** needs more data (more fresh NUDT city photos with wrong answers). **Not adopted.**

### Iteration 3 — Constant attitude bias
**Notebook:** `It3_Attitude_Bias`. **Loss targeted:** precision. 20–25 % of answers at 300 m+ fell beyond 1 σ, and Patch v0.5.6 had noted a ~0.98° roll bias.

**Model** (`pose_bias.py`): position error e = h · R(yaw) · (a_f, a_r) + b. That is a constant camera angle (projected by height and rotated by yaw) plus a constant offset. Optional tilt terms were also tried. The fit tries both mirror conventions and uses only answers within 4 σ. It was cross-validated **by flight**.

**Results:**

| Set | Terms | Before | After | Gain | Within 1 σ |
|---|---|---|---|---|---|
| official | const | 53.54 | 53.22 | −0.32 [−0.82, +0.19] | 95.1 → 93.6 % |
| official | angle | 53.54 | 54.11 | +0.57 [+0.06, +1.08] | 95.1 → 97.7 % |
| official | **angle + const** | 53.54 | **54.36** | **+0.82 [+0.38, +1.33]** | **95.1 → 98.8 %** |
| official | angle + tilt + const | 53.54 | 54.36 | +0.82 | same |
| NUDT 101 / 202 / 404 | angle + const | — | — | +0.08 / +0.12 / +0.21 | (NUDT has no camera bias) |

- Fitted values: a_f = tan(0.99°) forward, a_r = tan(0.11°) lateral, b = (−0.12, −1.46) m.
- At 250–400 m, answers within 1 σ go from 92.4 % to 97.9 %. The median error drops from 0.56 σ to 0.33 σ.
- The angle term matters most at height. At 3,000 m σ ≈ 39 m, while a 0.98° angle shifts the answer 51 m (1.3 σ).
- A learned GBM refinement on top gave +0.63 on official but −0.12 on NUDT 404. → **killed**.

**Verdict:** **keep** for the official camera. Check the sign against the submission's own nadir conversion before shipping.

### Iteration 4 — Agreement between queries far apart in heading
**Notebook:** `It4_Spaced_Agreement`. **Idea:** inside the 27-query escalation, answer when ≥ 2 queries at least 5–10° apart in heading agree. Queries that far apart should be independent evidence.

**Results:**
- Gap 5°, ≥ 3 inliers: official +1.52, but **4 wrong** on NUDT 101, 2 on 202 and 4 on 404.
- Every setting that adds answers makes 1–4 wrong answers on NUDT. The only safe setting adds nothing.
- Reason: on NUDT city scenes, look-alike buildings make the same wrong place win at several headings, so the queries aren't independent.

**Verdict:** **killed.**

### Iteration 5 — Headroom on skipped photos (diagnostic)
**Notebook:** `It5_Headroom`. **Question:** for photos today's rule skips, is the right place among the candidates at all?

**Official — all 34 skipped photos have the right place among the candidates:**
- the right place has a median of 4 inliers (range 0–10);
- wrong places have a median of 0 and a maximum of 14.

The loss here is fine matching plus the decision, so a better matcher could recover marks.

**NUDT — mostly recall:**
- 21 of 36 skipped photos have **no right candidate at all**.
- False places reach 25–44 inliers, versus at most 14 on official.

That is why inlier thresholds don't transfer from official to NUDT.

**Verdict:** gains on official come from matching and deciding; gains on NUDT would need better recall (coarse search or heading range).

### Iteration 6 — XoFTR as a second opinion
**Notebook:** `It6_XoFTR_Second_Opinion`. **Idea:** on photos the 6 queries don't answer, run XoFTR (MINIMA weights) over the 27 queries. Answer when the LoFTR and XoFTR choices agree within 1 σ.

**Bug found:** `src/fine_matching/xoftr_matcher.py` fails on every call with `KeyError('batch_indexes')`. It predates the batched LoFTR plumbing. Worked around in `train/checks/xoftr_batched.py`, which adds the batch index from XoFTR's `m_bids`; `src/` was not edited. The first run (phase 2) failed and was redone as phase 4. The failed notebooks are in `results/invalid/`.

**Results (agreement mode, 0 extra wrong answers on every set):**

| Set | Gain | Extra answers |
|---|---|---|
| official | **+2.72 [+1.55, +3.99]** | 18 |
| NUDT 101 | +0.40 | 2 |
| NUDT 202 | +0.30 | 1 |
| NUDT 404 | +0.30 | 1 |

- XoFTR alone at bar 8 gave 4 wrong answers on NUDT 101.
- Cost: 27 extra XoFTR queries (about 9 s on the PC) on ~10 % of photos, plus a second network. Jetson memory is already 7.1 of 7.4 GiB.
- Not run on the blind set.

**Verdict:** **promising, but not for the Jetson.**

### Iteration 7 — Combined Jetson policies
**Notebook:** `It7_Jetson_Policy`. Combined the Iteration 1 budget pieces with agreement, the verifier and the veto at M = 5 and M = 10.

**Verdict:** **superseded by Iteration 8.** These policies still had the greedy order and the verifier. The frozen J_M10 and J_M5 policies came from here.

### Iteration 8 — Fresh photos, DINO-z veto, blind confirmation
**Notebook:** `It8_Fresh_Photos_and_Z_Veto`.

**Finding:** **today's rule is not safe on fresh photos.** It gave 4 wrong answers on NUDT 404, all on city flights with look-alike buildings at 8–9 inliers. The earlier "0 wrong" came from samples of the same 500-photo pool that its thresholds were tuned on.

**Fix tested:** a **DINO-z veto**: answer only if DINO z ≥ 1.3 or there are ≥ 15 inliers. It is free, since z is already computed.
- NUDT 404: 4 → **0 wrong** (+2.58).
- Official: −0.16.
- NUDT 101 / 202: no change.

**Stricter alternative:** bar 10 + 3 agreeing queries. 0 wrong on the seen sets, but official −3.8.

**Blind NUDT 505**, scored once after freezing:

| Policy | Score | Wrong | Mean q | Jetson s (model) | vs Jetson today |
|---|---|---|---|---|---|
| Jetson today | 54.68 | 1 | 6.39 | 14.1 | — |
| today (PC, no cap) | 55.04 | 1 | 7.57 | — | +0.36 [−0.56, +1.40] |
| today + z-veto | 55.04 | 1 | 7.57 | — | +0.36 |
| strict bar 10, 3 agree | 54.64 | 1 | 7.79 | — | −0.04 |
| J_M10 (cap 8, greedy, z-veto) | 54.04 | 1 | 6.18 | 16.1 | **−0.64 [−1.44, −0.04]** |
| J_M5 (cap 9, greedy, agree ≥ 4, z-veto) | 54.64 | 1 | 6.27 | 13.8 | −0.04 |

- Every policy has the same single wrong answer: photo 889. The right place is absent from all 27 queries, and a false place reaches 6–11 inliers. That is a recall problem; no decision threshold fixes it.
- The greedy order loses on the blind set, which confirms Iteration 1's verdict.
- Policy **G** is J_M5 with the greedy order swapped back to the submitted second-pass order. This removes a tuned piece; it doesn't add one.

**Verdict:** the veto is a **free safety net; keep it.** Its gain on unseen photos is not shown: blind 505 is unchanged.

## 6. Final comparison (all five sets)

From `results/final_summary_five_sets.csv`. Each cell is score / wrong / mean queries. Rows A–D are PC policies at M = 10. Rows E–G are Jetson replays at M = 5 with a 10-query cap.

| Policy | official | NUDT 101 | NUDT 202 | NUDT 404 | NUDT 505 (blind) |
|---|---|---|---|---|---|
| A: bar 8 only, 6 queries | 49.01 / 0 / 6.0 | 53.00 / 0 / 6.0 | 55.38 / 0 / 6.0 | 53.34 / **2** / 6.0 | 53.04 / 0 / 6.0 |
| B: + agreement | 52.91 / 0 / 6.0 | 54.04 / 0 / 6.0 | 55.68 / 0 / 6.0 | 52.83 / **4** / 6.0 | 53.64 / **1** / 6.0 |
| C: + escalation 27 @ 11 (today) | 53.48 / 0 / 7.9 | 55.56 / 0 / 7.6 | 57.36 / 0 / 7.0 | 53.43 / **4** / 6.9 | 55.04 / **1** / 7.6 |
| D: C + z-veto | 53.32 / 0 / 8.0 | 55.56 / 0 / 7.6 | 57.36 / 0 / 7.0 | 56.01 / 0 / 7.2 | 55.04 / **1** / 7.6 |
| E: Jetson today | 50.28 / 0 / 6.6 | 54.24 / 0 / 6.4 | 55.26 / 0 / 6.3 | 55.44 / 0 / 6.3 | 54.68 / **1** / 6.4 |
| F: E + agreement (≥ 4 inl each) | 52.43 / 0 / 6.5 | 55.16 / 0 / 6.4 | 55.56 / 0 / 6.3 | 55.53 / **1** / 6.2 | 54.76 / **1** / 6.3 |
| **G: F + z-veto** | 52.27 / 0 / 6.5 | 55.16 / 0 / 6.4 | 55.56 / 0 / 6.3 | 56.13 / 0 / 6.2 | 54.76 / **1** / 6.3 |
| **G + attitude correction** | **53.10 / 0** | — | — | — | — |

**Paired gains:**

Escalation over B (C − B): positive on all five sets, including blind +1.40 [+0.40, +2.60].

G over E (Jetson):

| Set | Gain |
|---|---|
| official | **+1.99 [+0.95, +3.13]** |
| NUDT 101 | +0.92 [+0.20, +1.76] |
| NUDT 202 | +0.30 [0, +0.90] |
| NUDT 404 | +0.69 [0, +1.59] |
| NUDT 505 (blind) | +0.08 [0, +0.24] |

G + attitude over E on official: **+2.82 [+1.68, +4.02]**, 0 wrong.

Queries per photo: E 6.3–6.6, G 6.2–6.5. Jetson time is unchanged (~14 s mean, modelled).

## 7. Recommendation

**Put in the submission (Main_Pipeline_Headless), in order of value per hour:**
1. **Attitude correction.** Subtract h · R(yaw)(a_f, a_r) + b from the final position, with a_f = tan 0.99°, a_r = tan 0.11°, b = (−0.12, −1.46) m. Refit once on the final code's answers with `pose_bias.py`. Gain: +0.8 on official, no time cost, and it grows with height. **Check the sign convention first.**
2. **Agreement with ≥ 4 inliers per agreeing query** after the 6 standard queries. No extra query.
3. **DINO-z veto** (z ≥ 1.3 or ≥ 15 inliers). Free insurance against the NUDT city failure; costs −0.16 on official.
4. Keep the existing second-pass order and bar 8.

**Expected:** official 50.28 → 53.10 (+2.82 [+1.68, +4.02]), 0 wrong; NUDT 0 to +0.9; Jetson time unchanged.

**Do not put in:**

| Idea | Reason |
|---|---|
| Greedy query order | blind −0.64 |
| Learned verifier | not safe across sets |
| Spaced agreement | wrong answers on NUDT |
| XoFTR second opinion | needs a second network on the Jetson |

## 8. Still open

- **NUDT city wrong answers** (blind photo 889). This is a recall problem in the coarse search or heading range; no decision rule tested removes it without losing official marks.
- **Official skipped photos (34).** The right place is always present with 0–10 inliers. A second matcher (+2.7) or a fine-tuned matcher could recover them, at a Jetson cost.
- **No Jetson here.** Time is modelled. Measure M = 10 against M = 5 on the device. The remaining budget loss is the slow FFT on the Orin Nano (cuFFT fails there).
- **Heights above 351 m are untested.** The simulator wasn't run this iteration.
- G and the veto showed "no harm" on the blind set (+0.08); their gains come from seen sets.

## 9. Problems hit and how they were handled

| Problem | Handling |
|---|---|
| Scene-A data copy cut off at 1.8 GB | re-copied by the user |
| DINO weights missing | fetched from Hugging Face (approved) |
| 428 MB disk free | Trash emptied by the user (3.8 GB free) |
| Laptop crash, segfaults, "unknown opcode" | traced to MCE on CPU 10/11; all runs pinned with `taskset -c 0-9,12-31` |
| uint8 confidence would drop matches at the 0.22 cut | switched to uint16 ×65535 |
| `paired_bootstrap` scale off ×10 (its test too) | fixed to d = (b − a) × 6 |
| NUDT seed 303 overlapped seed 202 | moved to `results/invalid/`; replaced by fresh 404 / 505 from outside the pool (2,350 frames) |
| XoFTR `KeyError('batch_indexes')` on every photo | `xoftr_batched.py` wrapper, rerun as phase 4 |
| It2 "add" mode made a wrong answer (202, photo 377) | added "late" mode; 202 marked as seen for it |

## 10. Reproducing

```bash
# unit tests (made-up data, no GPU)
python tests/pipeline/test_iteration6_blocks.py

# GPU dumps (pinned off the faulty core), one per set; env options are listed in the notebook
bash tests/Test_Iteration_6/run_dumps.sh        # and run_dumps_phase2.sh ... phase5.sh

# offline analysis: run notebooks/It1_* ... It8_*; each has an _executed_all.ipynb copy
```

The rich dumps (`results/rich_dump/`, 177 MB of `.npz`) are **not in git**. Large data is shared outside the repo, as `.gitignore` says. Re-create them with the dump scripts, or ask for the copy.

## 11. Files on this branch

| Path | What |
|---|---|
| `train/checks/rich_dump.py` | GPU dump of all 27 queries with raw matches; `pack_photo`, `load_rich` |
| `train/checks/replay.py` | exact offline replay of today's rule; scoring, paired bootstrap, headroom |
| `train/checks/query_policy.py` | Jetson-style query policies: cap, order, early exit, agreement, z-veto |
| `train/checks/verifier_features.py`, `verifier.py` | fit features and the learned verifier |
| `train/checks/pose_bias.py` | attitude-bias fit and correction |
| `train/checks/xoftr_batched.py` | XoFTR wrapper that fixes `batch_indexes` |
| `tests/pipeline/test_iteration6_blocks.py` | 6 unit tests |
| `tests/Test_Iteration_6/notebooks/` | `Rich_Dump_Run` + `It1`–`It8`, each with executed copies |
| `tests/Test_Iteration_6/results/*.csv`, `frozen_rules.json` | every table quoted above |

**Dependency note:** the blocks import `heading_query_pipeline` and `fine_matching.xoftr_matcher`. Those files are on `prabhuparth/dev`, not on `palash/dev`. Bring them in from that branch to run the dumps.
