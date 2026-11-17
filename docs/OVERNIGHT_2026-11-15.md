# Overnight run of 2026-11-15: the ideas planner's second list

Started 23:58 IST. Seven ideas from the second list (Notion: "Ideas planner, second list: after the evening update (2026-11-15)"), one notebook each in `tests/Test_Iteration_7/notebooks/`, run one at a time by `tests/Test_Iteration_7/scripts/run_second_list_overnight.sh`. Nothing runs in parallel; every job is inside `tools/run_capped.sh` (6 GB RAM, 8 of 12 cores).

## Queue (hard time limit per job in brackets)

| order | notebook | data | what it answers |
|---|---|---|---|
| done | `Idea2b_Mosaic_Feasibility` | official dev photos | footprints of our photos overlap into one 2.85 km² cluster; a virtual photo of 1000 m (1.6 x 1.3 km) is 96 % covered, 1500 m is not. So a mosaic can test up to about 1000 m, not 3000 m. Only a footprint check: no mosaic was built. |
| 1 | `Idea1b_Coarse_Aligned_Count` (90 min) | official150, 12 low-altitude skips + 16 controls | do aligned matches at low confidence separate the true place from false candidates when RANSAC cannot (idea 1) |
| 2, 3 | `Idea3_Pose_Sweep_And_Part_Correlation` (150 min each) | NUDT valid 9 never-retrieved + 10 controls; official 12 skips + 8 controls | is the true place in the top 50 at ANY pose (pose coverage), and do part maps with slack rank it higher (idea 3) |
| 4, 5 | `Idea7b_Large_Window_LoFTR` (90 min each) | same two sets | truth block rank and LoFTR inliers inside a large window (idea 7) |
| 6, 7 | `Idea4_Part_Peak_Agreement` (150 min each) | official150 (all 150 photos), NUDT valid draw | AUC of part-peak features against coarse score and LoFTR inliers; threshold table with rescues and wrong answers (idea 4) |
| 8 | `Idea5b_Altered_Copies` (120 min) | 60 official photos x clean / invert / blur | do inverted and sharpened copies of skipped photos recover points without wrong answers (idea 5) |
| 9 | `Idea6b_Refine_Scale_Heading` (120 min) | 60 official photos, clean and height +20 % dumps | does a correlation search of scale and heading at the candidate recover the missing 11 points (idea 6) |

Executed notebooks: `<name>_executed_<tag>.ipynb` next to the source ones. Result tables: `results/idea1b_*`, `idea3_*`, `idea7b_*`, `idea4_*`, `idea5b_*`, `idea6b_*`. Logs: `results/logs/second_list_*.txt`, chain status in `results/logs/second_list_status.txt`.

## Safety against a stuck run

- `scripts/watchdog.sh` (every 5 min, log `results/logs/watchdog.txt`): kills the running notebook when the kernel's CPU time has not moved for 25 min and the GPU is idle, or when free RAM falls under 700 MB. The chain then goes on with the next job.
- Every job also has a hard `timeout` (see the table). A killed or timed-out job leaves no result table for that run; check the status file for `exit` codes other than 0 (124 = timeout).

## Reading the results

Each notebook's first cell states the planned falsification before running. Results are NOT conclusions until read against that and checked on a second set.
Known weaknesses of the tests, written before the results:
- Idea 7: in the one-photo smoke test the control photo (true place at the window centre) got 0 inliers. The half-resolution photo pasted into a black 640 px picture may simply be too small for LoFTR, so a "dead" there is not proof; treat it as inconclusive unless the controls get inliers.
- Idea 1: RANSAC inlier counts are nondeterministic, and the 16-cell grid of (confidence, tolerance) invites a lucky cell; look for a region, and the controls must win too.
- Idea 5, 6: first pass only (6 queries) for 5; the wide-scale NUDT check is still missing for 6.
- Every number is on official dev photos (in-sample) or one NUDT draw.

Nothing was committed.
