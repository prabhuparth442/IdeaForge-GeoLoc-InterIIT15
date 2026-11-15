# Review protocol — 13 November 2026

All new work stays in this directory. Existing source, settings, weights, notebooks and submission files are not changed. No git checkout, reset or commit.

The available real-image dumps are official150 and nudt707, each 150 photos, 27 queries, 10 fine candidates per query. Both have already been inspected in earlier experiments. They are **development/regression data**, not a new blind test for this review. Official data also participated in judge training. Do not call an improvement here a proven generalisation gain.

Before running comparisons, fix these small, mechanically motivated variants (no threshold sweep):

1. Existing agreement + escalation, reproduced through the original replay.
2. Eligible rescue: only when the original decision skips, consider candidates meeting the existing stage's inlier bar before applying the existing inliers + z ranking. This tests whether a high-z, below-bar candidate hides a valid candidate. Preserve every existing answer.
3. Inlier-first: the existing pipeline with gamma = 0. This tests sensitivity to allowing DINO to override geometric support.
4. Early12: stop on the pooled choice after a standard-query prefix when it has at least 12 inliers (the existing judge's unconditional override); otherwise run the unchanged agreement pipeline. This is a speed/accuracy experiment, not a certified confidence threshold.
5. Valid-mask refit: refit stored LoFTR pairs with the existing confidence and geometry gates, excluding keypoints outside the actually photographed canvas, with no erosion. Compare against a control refit of the same quantised raw pairs to separate algorithm changes from dump quantisation.

Also replay the actual headless query order and decision structure on M=5 candidates with no time limit, and separately at a ten-query cap. Saved pairs were generated at the research matcher thresholds; these are **policy replays**, not measurements of the deployed 0.21 matcher or Jetson runtime.

Current-settings check: the active Single_Image_Run notebook uses match confidence 0.21, whereas the stored dumps used 0.22 (both coarse threshold 0.21). Refit the stored raw pairs at 0.21 as a separate, named baseline; the dump retains pairs down to 0.15. This reconstructs the current confidence cut without another network pass, subject to dump quantisation. Also replay the headless policies using these refitted pairs. Do not silently identify the 0.22 regression score with the active notebook configuration.

Report scores /60, wrong answers, coverage, per-photo changes and query counts. Paired photo-bootstrap intervals describe these sets only; spatially correlated images are not independent flight evidence. Never filter photos out because they fail or contain no candidates.

Audit risk-threshold functions with zero certified thresholds, identity/grouping, train/test provenance, timing semantics and current settings. CPU test suite results and any environment failures are recorded separately from score experiments.

Promotion requires a frozen rule evaluated once on new flight/spatial blocks, a separate calibration set, no physical-image overlap, and target-board end-to-end timing. Additional random seeds from an already explored city are weaker evidence than a new acquisition/site.

## Additional exploratory precision test

After the first fixed comparisons (none improved score), test one position-only change: preserve every answer/skip decision, and replace an accepted position by the componentwise median of at least three query-best estimates within the existing one-sigma agreement radius. Each supporting query must meet the existing three-inlier agreement floor. Use only queries already run. This pools correlated geometric estimates to reduce position noise; it does not treat queries as independent statistical evidence. No tuning of radius, minimum support or weighting after scoring. This is exploratory on already-seen data.
