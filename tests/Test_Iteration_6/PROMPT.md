# Task brief for Test_Iteration_6 (from the user, 2026-11-13)

Kept here so every later session works to the same brief. Paths in the original pointed at another computer
(`/home/voidness/Storage/Synced Folders/InterIIT Tech 15/...`); on this machine they are:

| Original | Here |
|---|---|
| repo `.../IdeaForge-GeoLoc-InterIIT15` | `~/GeoLoc` (commit f914ca5, branch `prabhuparth/dev`) |
| venv `.../venv-geoloc/bin/` | `~/Ideaforge/GeoLoc/.venv-geoloc/bin/` |
| `../Submission_Files/` (submission code, Jetson run, TeamKit) | **not on this machine** |

## Goal

Find where the method still loses marks, propose improvements (classical and learned), test them properly, repeat for many iterations. Work alone,
report back with evidence.

## Scoring (everything depends on it)

- Per photo: <= 1 sigma +10, <= 2 sigma +6, <= 4 sigma +3, <= 8 sigma 0, worse -20, skip 0. Score out of 60 = mean per photo x 6.
- sigma = sqrt(8^2 + (3 max(h, 100) / fx)^2) m, about 8.1-9.2 m on dev photos (`src/header/scoring.py`).
- One wrong answer costs two perfect ones: **every change must keep wrong answers at 0 on every set it is checked on.**
- The real evaluation uses a different map and probably heights 100-3000 m (dev photos only reach 350.7 m).
- Submission runs on a Jetson Orin Nano (15 W), per-photo budget 22 s (`TIME_BUDGET_S`).

## Pipeline today

thermal photo -> north-up canvas (TILE 224, levels 3-7, MAP_GSD 0.3) -> edge-direction cells (CELL 4, N_BINS 8), FFT ZNCC over a 5.5 km disc ->
top K = 50 -> DINO re-rank -> LoFTR-MINIMA + RANSAC similarity on M = 10 per query -> most inliers + gamma z -> answer or skip.
Queries: standard 6 (headings -5, 0, +5 x scales 1.0, 1.05); wide 27 (-10..+10 step 2.5 x 0.95, 1.0, 1.05).
Decision (`agreement_query_pipeline.py`, `src/semantic_rerank/agreement.py`): answer at >= 8 inliers, or >= 2 of the 6 queries' own best
candidates (each >= 3 inliers) within 1 sigma; else escalate to 27 queries and answer at >= 11 inliers.

## Known numbers (verify)

- Stage_Evaluation (100 official): 1 query 39.6 -> 6 at bar 47.16 -> + agreement 50.76 -> + escalation 52.56 (0 wrong, 5.98 s/photo PC).
- Decision_Report: official 379 49.01 -> 53.54; NUDT 101 (300) 53.00 -> 55.56; NUDT 202 (200, unseen) 55.38 -> 57.36; all 0 wrong.
- Jetson: 49.57/60, 423 answered, 0 wrong, median 13.1 s, max 21.1 s; **66 of 77 skips were cut by the 22 s budget**.
- Coarse is not the bottleneck; loss is fine matching + deciding + the Jetson budget. 20-25 % of answers at 300 m+ beyond 1 sigma; roll bias
  ~0.98 deg documented. Inlier thresholds do not transfer between data sets. No DEM for official; scale sweep flat.

## What to explore (rank by expected marks per hour)

A. Classical: (1) Jetson speed: early exit, query order, fewer DINO calls, smaller K/M, batching, fp16/TensorRT; a time-aware escalation.
(2) Matcher on hard photos (XoFTR, ELoFTR, RoMa, LightGlue, match size, confidence). (3) Fine-fit geometry: constant roll/pitch offset (fit on
official, check on NUDT), homography / ray fit. (4) High altitude: levels above 7, border mask, per-band bars, simulator at 500-3000 m.
B. Learned (state input, output, data, what it replaces, leakage risk, cheapest test): (1) candidate verifier (logistic / GBM first, MLP only if
better; train NUDT test official and reverse); (2) position refinement; (3) fine-tuned matcher (first count photos just under the bar);
(4) modality translation encoder (check Overnight learned-feature results first); (5) learned coarse encoder / ranker (low priority);
(6) learned decision over queries (DeepSets).

## Rules (follow exactly)

- Notebook is the face, blocks do the work: small swappable blocks (one job, docstring, no plotting); experiments are **executed notebooks**,
  saved as `..._executed_<set>.ipynb` next to the notebook. No `run_*.py` scripts for experiments.
- One `src/` only; new blocks in `src/` or `train/`; a changed existing `src/` file goes to `tests/Test_Iteration_6/changed_src/<same path>`.
- Work in `tests/Test_Iteration_6/` (`notebooks/`, `results/`, `README.md`); keep the README updated per iteration.
- **No commit or push** unless asked.
- **Generalise before trusting:** a gain counts only if (a) a logical reason, (b) holds on official AND NUDT with >= 2 seeds, one unseen while
  choosing, (c) paired bootstrap 95 % interval of the gain above 0, (d) wrong answers stay at 0. Choose on one set, report on the others.
- Report time per photo with every score; flag anything pushing Jetson photos past 22 s.
- Long runs in the background; no sleep-polling. Keep the user's email and personal details out of external requests.
- Do not re-run what Notion says failed without a concrete reason.

## Iteration loop

Hypothesis + reason + targeted stage -> cheapest test (dumps -> GPU subset -> full sets) -> official, NUDT 101, unseen NUDT seed, paired bootstrap
vs agreement + escalation -> verdict (keep / kill / more data) in README + a dated Notion child page under "InterIIT_UAV_GeoLoc" -> re-rank.

## Deliverables

- `README.md`: every idea (hypothesis, test, score per set, wrong, time, verdict).
- Executed notebooks per experiment; new blocks with unit tests in `tests/pipeline/` or `tests/train/`.
- Short recommendation: what to put in the submission, gain with intervals, Jetson cost, what is still open.
- Notion page per iteration and one summary page.
