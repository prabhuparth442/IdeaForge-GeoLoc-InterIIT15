# Second-matcher fallback (iteration 42): results, measured 2026-11-18

Base = XoFTR from 800 m (frozen). Fallback: when a query misses the 8-inlier bar with the photo's matcher, the other matcher tries it.
Score = 6 x mean points per photo (+10/+6/+3/0/-20 on sigma, 0 skipped), recomputed from the details files here.
Server runs: out_fb_* in /mnt/e/IdeaForge/work. `details_*.csv` = fallback run, `ref_details_*.csv` = same set without fallback.
`dev_fbnative` = fallback with FALLBACK_NATIVE_FROM_HEIGHT_M set (the launch arguments were not logged; the details files do not show the setting, so this label is from the run name).

| set | no fallback | fallback | answered | wrong | mean s/photo |
|---|---|---|---|---|---|
| new working 332 | 39.70 | 40.14 | 284 -> 288 | 0 -> 0 | 1.78 -> 3.04 |
| new working 332, native fallback | 39.70 | 41.44 | 284 -> 296 | 0 -> 0 | 1.78 -> 4.10 |
| old official 500 | 52.69 | 56.05 | 454 -> 482 | 1 -> 1 | 2.44 -> 3.48 |
| NUDT headless 300 | 48.52 | 48.28 | 281 -> 285 | 12 -> 14 | 3.14 -> 2.81 |

Kill rule (written before the runs): score up on >= 2 of 3 sets, wrong +<= 1 per 300 photos, time < 2x.
Verdict: score rule passes (new set, old official), time rule passes, wrong rule FAILS on NUDT (+2 wrong; 5 new wrong, 3 old wrong fixed). The NUDT wrong answers carry 19-38 inliers (look-alike places), so a higher bar would not remove them.
Not adopted as default. Caveat: the 332 working photos and old official are development sets; no locked-test run of the fallback.
