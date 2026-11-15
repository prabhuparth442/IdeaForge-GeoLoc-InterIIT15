# Prompt: explore, test and improve the geo-localisation pipeline (Test_Iteration_6)

Copy everything below the line into a new session started in
`/home/voidness/Storage/Synced Folders/InterIIT Tech 15/IdeaForge-GeoLoc-InterIIT15`.

---

You are working on the IdeaForge geo-localisation pipeline for Inter-IIT Tech 15. A thermal drone photo (with camera, height, attitude and
a position hint within about 5 km) must be placed on a satellite map. Your job: **find where the method still loses marks, propose ways to
improve it (classical and learned), test them properly, and repeat for many iterations.** Work on your own; report back with evidence.

## 1. How the task is scored (read this first, everything depends on it)

- Per photo: within 1 sigma +10, within 2 sigma +6, within 4 sigma +3, within 8 sigma 0, worse than 8 sigma **-20**, skip 0. Reported as a score
  out of 60 (mean per photo x 6).
- sigma = sqrt(8^2 + (3 * max(h, 100) / fx)^2) metres, about 8.1 to 9.2 m on the dev photos (`src/header/scoring.py`).
- One wrong answer costs as much as two perfect ones gain. **Every change must keep wrong answers at 0** on every set it is checked on.
- The real evaluation uses a different map and probably different heights. The kit's sample generator
  (`Submission_Files/TeamKit/harness/make_sample_data.py`) draws heights from 100 to 3000 m and `score.py` reports bands up to more than
  2000 m. Dev photos only reach 350.7 m.
- The submission runs on a Jetson Orin Nano (15 W), with a per-photo budget of 22 s in our code (`TIME_BUDGET_S`).

## 2. The pipeline today

thermal photo -> north-up canvas (TILE 224, map levels 3-7, MAP_GSD 0.3) -> edge-direction cells (CELL 4, N_BINS 8), FFT ZNCC over a 5.5 km
disc -> top K=50 peaks -> DINO re-rank -> LoFTR-MINIMA + RANSAC similarity (`cv2.estimateAffinePartial2D`) on M=10 per query -> choose the most
inliers + gamma * z -> answer or skip.

- Queries: standard 6 (headings -5, 0, +5 x scales 1.0, 1.05); wide 27 (headings -10..+10 step 2.5 x scales 0.95, 1.0, 1.05).
- Decision (newest, `agreement_query_pipeline.py` + `src/semantic_rerank/agreement.py`): answer if inliers >= 8, or if >= 2 of the 6 queries'
  own best candidates (each >= 3 inliers) lie within 1 sigma of the chosen place. Otherwise escalate to the 27 queries and answer only at
  >= 11 inliers (no agreement there: queries 2.5 degrees apart are not independent).
- Main code: `src/` (blocks: header, image_processing, coarse_search, semantic_rerank, fine_matching, result_thresholding, evaluation),
  `single_photo_pipeline.py`, `heading_query_pipeline.py`, `agreement_query_pipeline.py`. Training code and research blocks: `train/`
  (features incl. learned and MIND, coarse_methods, models/ranker.py, datasets, simulation). Submission code is a separate code base:
  `../Submission_Files/myteam/code/`.

## 3. Where the numbers stand (start by checking these yourself)

**`Stage_Evaluation.ipynb`** (repo root) is the main loss-by-stage notebook; its tables are saved in `tests/Stage_Evaluation/`. Read the notebook
(markdown and outputs) and these CSVs before anything else:

- `all_phases_official_edge_zncc_headings9_scales3_K50_M10.csv` (100 official photos):
  1 query 39.6 -> standard 6 at the bar 47.16 -> + agreement 50.76 -> + escalation 52.56 (0 wrong, 5.98 s/photo on the PC, 5 photos over 22 s).
- `headings_points_lost_per_stage_*.csv`: which stage loses how many points per variant (coarse, DINO, fine, decision, precision).
- `decision_steps_*.csv`, `headings_failures_*.csv`, `time_per_query_*.csv`, `sweep_*.csv`.
- Caveat in the notebook itself: 100 photos, sweeps searched on the photos they are scored on, so intervals are several points wide.

Other results to read:

- `tests/Test_Iteration_3/Decision_Report.md`: decision experiments on candidate dumps. Official 379 photos 49.01 -> 53.54, NUDT seed 101
  (300) 53.00 -> 55.56, NUDT seed 202 unseen (200) 55.38 -> 57.36, all 0 wrong. Also lists rejected ideas (vote choice, mean position,
  agreement inside escalation).
- Candidate dumps (every candidate of every query, per photo): `tests/Test_Iteration_3/results/fresh_run/dump_*_conf0.22/photo_<i>.npz`.
  Loader and offline decision tools: `train/checks/vote_decision.py`. These let you test any decision or verifier idea **without the GPU**.
- `tests/Test_Iteration_4/`: XoFTR vs LoFTR (official and NUDT, several seeds).
- `tests/Test_Iteration_5/`: drone-view simulator (`train/simulation/`): geometry verified, appearance is the open part; fakes are easier
  than real photos downstream.
- `tests/Parameter_Test/Analysis_2026-11-06_Stage_Evaluation_and_Strategies.md` and `tests/Parameter_Test/runs_summary.csv`.
- Jetson run: `../Submission_Files/Jetson_Transfer/runs/20261110_105013/` and `../Submission_Files/Jetson_Transfer/CONTEXT_FOR_JETSON_CLAUDE.md`.
  49.57/60, 423 answered, 0 wrong, median 13.1 s, max 21.1 s; **66 of the 77 skips were cut by the 22 s budget, not by matching.**

Known facts from earlier sessions (verify, do not trust blindly):

- Coarse search is not the bottleneck: the right place is missing from all 60 candidates on only 3/379 official and 8/200 NUDT photos.
- All 34 still-skipped official photos have the right place among their candidates, with only 0-10 inliers (26 at 100-150 m, 8 at 350 m).
  The loss is in **fine matching and deciding**, plus the Jetson time budget.
- About 20-25 % of answers at 300 m and above land outside 1 sigma. A roll bias of about 0.98 degrees is documented (Patch v0.5.6 page);
  1 degree at 350 m is about 6 m, roughly 0.7 sigma. The fine fit is a 2D similarity, which ignores roll/pitch.
- Inlier thresholds do not transfer between datasets; pick them on one set, check on another.
- No DEM for the official area (only NUDT `dsm.tif`); scale sweep was flat (1.0-1.05).

## 4. Notion (fetch it, it is up to date)

The Notion workspace has a project page **"InterIIT_UAV_GeoLoc"** (under "Projects_n_Technical") with dated child pages. Use the Notion search
and fetch tools to read at least:

- "Loss hunt: where the marks go and the decision gate (2026-11-09)" and any newer pages
- "Parameter tests and heading queries: results and what failed (2026-11-07)"
- "Patch v0.5.6 revised: recall geometry and roll bias (2026-11-06)" and the withdrawn v0.5.6 page (why it was withdrawn)
- "Overnight experiment: fusion, MIND, whitening, learned features (2026-11-04)" (what learned features already tried and how they did)
- "Iteration 2: DINO as a semantic layer between coarse and fine (2026-11-04)"
- "Coarse matching: results and decisions (2026-10-29)", "Organiser updates log (2026-10-19 to 2026-11-05)"

Do not re-run something Notion says already failed unless you have a concrete reason it would now behave differently, and say what it is.
Log every iteration as a new dated child page under the project page (what, why, numbers on every set, verdict, next step).

## 5. What to explore

Rank ideas by **expected marks gained per hour of work**, aimed at where marks are actually lost. Starting list (extend it):

### A. Classical / engineering
1. **Speed on the Jetson** (largest measured loss: 66 budget cuts). Profile per stage on the stored timings; try fewer DINO calls, smaller K/M
   per query, early exit once the bar or agreement is met, query order by expected value, batching, fp16/TensorRT for the matcher. The agreement
   escalation adds ~19 s on the PC per escalated photo, so it needs a time-aware version before it can go to the Jetson.
2. **Matcher on the hard photos**: XoFTR results (Test_Iteration_4), ELoFTR, RoMa or LightGlue variants, match size, confidence threshold,
   run only on the still-skipped photos first (cheap).
3. **Fine fit geometry**: fit a constant roll/pitch offset on dev truths (official), check it on NUDT; or a homography / ray-based fit with the
   known attitude. Target: precision at 300 m+.
4. **High altitude** (if the blind set goes above ~480 m): map levels above 7, padding/no-data mask (`USE_BORDER_MASK`), per-band bars. Test
   with the simulator at 500-3000 m, since there are no real photos there.

### B. Learnable models (ANNs, encoders) worth testing
For each, state the input, output, training data, what it replaces, the leakage risk and the cheapest test before training anything big.

1. **Learned candidate verifier**: per candidate features (inliers, inlier spread / convex hull, RANSAC residual, scale and rotation of the
   fit vs the query's heading and scale, DINO z, coarse z, rank, agreement count, height) -> P(right within 1 sigma). Start with logistic
   regression / gradient boosting, then a small MLP only if it beats them. Train on NUDT dumps, test on official, and the reverse. It replaces
   the inlier bar; success = more answers at 0 wrong across sets.
2. **Learned position refinement**: regress the residual offset of the chosen place from the match geometry and attitude (fixes the
   precision floor). Small model, cross-set check mandatory.
3. **Fine-tuned matcher** (LoFTR-MINIMA / XoFTR head or LightGlue on top of a frozen backbone) on thermal-to-map pairs from NUDT plus simulator
   renders. Expensive: first check how many skipped photos have the right place with inliers just under the bar.
4. **Modality translation encoder**: a small U-Net / contrastive encoder mapping thermal and map into a shared look before matching or before
   coarse cells. Check against the "Overnight experiment" learned-features results first.
5. **Learned coarse encoder or ranker** (`train/features/learned_net.py`, `train/models/ranker.py`): low priority on dev since coarse already
   recalls the target, but may matter at high altitude.
6. **Learned decision over queries** (set encoder / DeepSets over all queries' best candidates) as an alternative to the hand rule.

Data you can train on: NUDT photos (several seeds via `NUDT_SEED`), official photos (only as a held-out test unless you split by flight),
candidate dumps, simulator renders (`train/simulation/`, appearance gap to real is known). Keep models small enough for the Jetson budget.

## 6. How to work (rules from the user, follow exactly)

- **Notebook is the face, blocks do the work.** Code goes in small swappable blocks (one job each, docstring at top, no plotting inside);
  notebooks call them. Experiments and concept checks are **executed notebooks**, not `run_*.py` scripts. Save executed copies next to the
  notebook (`..._executed_<set>.ipynb`). Use the project venv: `"/home/voidness/Storage/Synced Folders/InterIIT Tech 15/venv-geoloc/bin/"`.
- **One `src/` only.** New blocks may be added to `src/` or `train/`. If you must change an existing `src/` file, put the changed copy in
  `tests/Test_Iteration_6/changed_src/` (same relative path) instead of editing `src/`. Never duplicate the whole `src/`.
- Work in `tests/Test_Iteration_6/` (`notebooks/`, `results/`, `README.md`). Keep a short README updated with each iteration's result.
- **Do not commit or push** unless the user asks. Current branch: `prabhuparth/dev`.
- **Generalise before trusting.** A gain counts only if (a) there is a logical reason it should work, (b) it holds on official AND on NUDT
  with at least two seeds, one of them unseen while choosing settings, (c) a paired bootstrap 95 % interval of the gain is above 0, and
  (d) wrong answers stay at 0. Choose thresholds on one set and report them on the others. One-off in-sample results are not findings.
- Report time per photo with every score; flag anything that pushes Jetson photos past 22 s.
- Long runs go in the background; do not sleep-poll.
- Keep the user's email and personal details out of any external request.

## 7. Iteration loop (do this many times)

1. Pick the next idea from your ranked list; write the hypothesis and the logical reason it should help, and which stage's loss it targets.
2. Cheapest possible test first: offline on dumps / stored CSVs -> small photo subset on the GPU -> full sets.
3. Run on official, NUDT seed 101, NUDT unseen seed (e.g. 202 or new); paired bootstrap vs the current best (agreement + escalation).
4. Verdict: keep / kill / needs more data. Write it to the iteration README and a Notion page, including failures (they save time later).
5. Update the ranked list with what you learned; go again.

## 8. Final deliverables

- `tests/Test_Iteration_6/README.md`: table of every idea tried (hypothesis, test, score per set, wrong answers, time, verdict).
- Executed notebooks for each experiment; any new blocks with unit tests in `tests/pipeline/` (or `tests/train/`).
- A short recommendation: what to put in the submission, what it gains with intervals, what it costs on the Jetson, and what is still open.
- Notion pages for each iteration and one summary page.
