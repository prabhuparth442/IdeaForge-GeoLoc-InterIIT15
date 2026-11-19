# Configuration PV-X, written down before the locked test split was run (2026-11-18, about 21:30 server time)

Switches (Main_Pipeline_Headless/settings.py names; every one is off by default, so the submitted behaviour is unchanged):
XOFTR_FROM_HEIGHT_M=0 (XoFTR for every photo), PHOTO_VIEW_FROM_HEIGHT_M=0 (photo view for every photo), PHOTO_VIEW_MIN_INLIERS=12, PHOTO_VIEW_FIT_POSE=True, PHOTO_VIEW_PRIOR_TILT_DEG=0.4, PHOTO_VIEW_PRIOR_YAW_DEG=0.8, PHOTO_VIEW_PRIOR_HEIGHT=0.008, PHOTO_VIEW_FIT_ROUNDS=1; everything else as in settings.py (agreement rule on, 6 queries, second pass, DINO, MIN_INLIERS 8 for any photo that does not use the photo view: none).
Run with: tests/Test_Iteration_12/scripts/server_chains/pvx_chain.sh (copy Main_Pipeline_V3 = Main_Pipeline_Headless at this date).

How it was chosen (all on development data, so every number below is optimistic): photo view from the true-place diagnostic (130 working photos), pose fit and the prior width x0.4 by odd/even cross-validation offline on 125 working photos, the bar 12 and "XoFTR for every photo" from a handful of runs on the three development sets (working 332, old official 500, NUDT 300). Development results of PV-X: working 50.69 (1 wrong), old official 60.00 (0 wrong), NUDT 52.08 (6 wrong); bases 39.70, 52.69, 48.52.

The locked test split (168 photos) is run ONCE with exactly this configuration, report only. It was already used twice before (by Fable for the coarse rule, by us for XoFTR from 800 m: 38.00 -> 40.61), so it is not a clean untouched set: it is a second look, and nothing may be changed after seeing it. If the result is bad it is reported as it is.

## Result of the locked test split, run once with PV-X (2026-11-18 evening, server time 21:38)
168 photos: **49.32 / 60**, answered 166, 1 wrong, 2.76 s per photo (server, other jobs running). Same split with the base: 38.00 (140 answered, 0 wrong); with XoFTR from 800 m: 40.75 (155 answered, 0 wrong; the 40.61 written earlier was computed with a slightly different rule). So +11.3 over the base on photos the configuration was not tuned on. Caveats: second look at a split already used twice (see above); same flight region as the working photos, so it says nothing about a different region or a different camera; the server timing is not Jetson timing. details: runs/pvx_test_t0.csv.

## Frozen 2026-11-19
Preset in code: Main_Pipeline_Headless/presets/pv_x.py (test: tests/test_pv_x_preset.py). Not committed. PV-X2 (second opinion) is NOT part of the freeze; it needs a check on another set first.

Correction 2026-11-19 (red team): the first preset lacked LEVELS=[3..11] that the dev runs passed on the command line; added.
