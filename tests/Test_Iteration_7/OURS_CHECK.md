# Second-list ideas: checks in the submission pipeline (2026-11-16)

Nothing committed. Scores are mean points per photo x 6 (= out of 60). Jetson not measured.

## Verdicts
| idea | verdict |
|---|---|
| 1 aligned-count rescue | official: strong (500 photos, 25 rescues all right, +2.9/60, +0.2 s mean). Fresh NUDT valid1313 (150 photos, rule fixed in advance): 3 rescues, 1 wrong (-20), net -0.44/60. Does NOT generalise; keep off by default. |
| 5 altered copies | official 60-photo sets, bar-only copies, copies BEFORE second pass: clean 9.13 (same as base), invert 8.53 (base 7.87), blur 6.40 (base 5.67); 0 wrong; cost mean +0.2 / +0.4 / +1.3 s, p95 up to 12 s. Agreement-rule copies score a bit higher (clean 9.30, invert 8.53, blur 7.17) but harmful on NUDT valid earlier. After-order is not better. Worth it only for degraded input; no gain on clean photos. |
| 2 mosaic | footprint check only: virtual photos up to ~1000 m possible. |
| 3, 4, 6, 7 | dead (7c native window also dead). |

## 500 official photos in the submission pipeline
base 8.77 (1 wrong, 2.52 s mean, p95 5.85) | rescue 9.25 (1 wrong, 2.75 s, p95 6.51, 25 rescued) | rescue+copies 9.24 (1 wrong, 2.96 s, p95 9.80, 13 copies used). Copies add nothing beyond rescue on clean photos.

## Caveats
Official numbers are in-sample (dev photos); the only fresh test (NUDT valid1313) rejects idea 1. Idea 5 on NUDT through the headless pipeline is still unmeasured. Log headless_copies*.txt contain one old Traceback from the earlier failed quoting attempt; the runs themselves finished (details.csv complete).
Open: Idea 4 threshold grid range, Idea 5 notebook points_total accounting (known wrong), Stage_Evaluation scratch run, Jetson freeze.

## Follow-up 2026-11-16 (afternoon)
- Idea 5 on NUDT valid (60 photos x clean/invert/blur, offline re-decision of the stored copy inliers with the bar only, bar 8; exact, because the winning copy has the most inliers): 0 wrong copy answers (the two earlier wrong copies had 6 and 4 inliers). Gain: invert +0.60/60, blur +0.30/60, clean 0; copy time 0.27 / 0.74 / 0.73 s mean. So bar-only copies are safe on both datasets; gain is small on NUDT. Research pipeline, not the headless one.
- Idea 1e (notebooks/Idea1e_Rule_Scan_All_Sets.ipynb, results/idea1e_grid.csv): scan of threshold x margin over 5 sets (official150, valid1212, valid1313, fresh707, fresh1010). Only margin >= 16 gives 0 wrong on every set, and it rescues just 6 photos in 733 (mean +0.42/60; official gain falls from +4.1 to ~0). Margin 8-12 always has the one wrong rescue (valid1313, -0.44 to -0.80). valid1313 has now been seen, so margin 16 is a candidate only. Verdict: idea 1 is a tiny gain when safe; not worth enabling.
