# Overnight run report

Written 2026-11-04 09:36. Mode: NUDT.

## Photo-to-map geometry

yaw sign +1, yaw offset 0 deg, scale 0.80; 75.0 % of the check photos land within 4 sigma with it.
If this is below 30 %% nothing downstream can be trusted.

## C2, fusion (valid flights, truth against the candidate list)

| maps | model | valid top-1 | vs edge_zncc alone (95 % interval) |
|---|---|---|---|
| edge_zncc | monotone | 0.124 | +0.000 [+0.000, +0.000] |
| edge_zncc | gated | 0.124 | +0.000 [+0.000, +0.000] |
| edge_zncc+edge_cosine+edge_mag | linear | 0.204 | +0.080 [+0.044, +0.120] |
| edge_zncc+edge_cosine+edge_mag | monotone | 0.200 | +0.076 [+0.040, +0.116] |
| edge_zncc+edge_cosine+edge_mag | gated | 0.200 | +0.076 [+0.040, +0.116] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w | linear | 0.208 | +0.084 [+0.048, +0.124] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w | monotone | 0.176 | +0.052 [+0.016, +0.092] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w | gated | 0.204 | +0.080 [+0.040, +0.124] |
| edge_zncc+edge_cosine+edge_mag+mind_zncc | linear | 0.200 | +0.076 [+0.040, +0.116] |
| edge_zncc+edge_cosine+edge_mag+mind_zncc | monotone | 0.176 | +0.052 [+0.016, +0.092] |
| edge_zncc+edge_cosine+edge_mag+mind_zncc | gated | 0.192 | +0.068 [+0.028, +0.112] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc | linear | 0.200 | +0.076 [+0.040, +0.116] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc | monotone | 0.172 | +0.048 [+0.012, +0.084] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc | gated | 0.188 | +0.064 [+0.028, +0.104] |
| edge_zncc+learned_zncc | linear | 0.192 | +0.068 [+0.036, +0.104] |
| edge_zncc+learned_zncc | monotone | 0.192 | +0.068 [+0.032, +0.104] |
| edge_zncc+learned_zncc | gated | 0.200 | +0.076 [+0.032, +0.120] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc | linear | 0.248 | +0.124 [+0.084, +0.172] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc | monotone | 0.252 | +0.128 [+0.084, +0.176] |
| edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc | gated | 0.236 | +0.112 [+0.068, +0.156] |

Best on valid: **edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc / monotone** (0.252).

## R1, ranker on the hard candidates

fused score alone 0.252, ranker 0.536, difference +0.284 (95 % +0.208 .. +0.356). Use it only if the interval is above 0.

## Final check on the full score maps (NUDT test flights, never used for choosing)

| map / model | top-1 within 1 sigma | 2 sigma | 4 sigma | best of 25 within 4 sigma | median error (sigma) | 4-sigma change vs edge_zncc |
|---|---|---|---|---|---|---|
| edge_zncc | 0.053 | 0.100 | 0.173 | 0.493 | 165.1 | +0.000 [+0.000, +0.000] |
| edge_cosine | 0.033 | 0.080 | 0.153 | 0.480 | 198.2 | -0.020 [-0.073, +0.033] |
| edge_mag | 0.000 | 0.000 | 0.007 | 0.100 | 358.4 | -0.167 [-0.227, -0.107] |
| edge_zncc_w | 0.020 | 0.033 | 0.087 | 0.293 | 230.5 | -0.087 [-0.133, -0.040] |
| mind_zncc | 0.040 | 0.060 | 0.120 | 0.373 | 210.3 | -0.053 [-0.113, +0.007] |
| learned_zncc | 0.073 | 0.193 | 0.260 | 0.553 | 159.8 | +0.087 [+0.020, +0.153] |
| fused:edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc__monotone | 0.073 | 0.160 | 0.300 | 0.633 | 113.6 | +0.127 [+0.067, +0.193] |
| fused:edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc__linear | 0.087 | 0.193 | 0.307 | 0.627 | 96.0 | +0.133 [+0.080, +0.200] |

Pass rule C2 (fused >= best single by 0.03 at the 25-candidate ceiling): best fused fused:edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc__monotone vs best single learned_zncc (55.3 %): gain +0.080 -> **PASS**.

## Official development set (d1 photos, small_sat_area map): the overfitting / domain check

Models were fitted on NUDT only. Same score maps and fusion models, different city, camera, sensor and map.

| map / model | NUDT test 4-sigma | official 4-sigma | official 1-sigma | official best of 25 | official change vs edge_zncc (95 % interval) |
|---|---|---|---|---|---|
| edge_zncc | 0.173 | 0.805 | 0.554 | 0.900 | +0.000 [+0.000, +0.000] |
| edge_cosine | 0.153 | 0.778 | 0.525 | 0.889 | -0.026 [-0.055, +0.003] |
| edge_mag | 0.007 | 0.306 | 0.201 | 0.562 | -0.499 [-0.551, -0.446] |
| edge_zncc_w | 0.087 | 0.768 | 0.536 | 0.855 | -0.037 [-0.061, -0.013] |
| mind_zncc | 0.120 | 0.815 | 0.557 | 0.942 | +0.011 [-0.024, +0.045] |
| learned_zncc | 0.260 | 0.863 | 0.528 | 0.953 | +0.058 [+0.029, +0.087] |
| fused:edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc__monotone | 0.300 | 0.900 | 0.628 | 0.963 | +0.095 [+0.066, +0.127] |
| fused:edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc__linear | 0.307 | 0.894 | 0.615 | 0.955 | +0.090 [+0.061, +0.121] |

Reading: if a fused model beats edge_zncc on NUDT test but its interval on the official set includes 0 or is negative, the gain was specific to Changsha (overfitting to the domain). Best fused here: fused:edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc__monotone, change +0.095 (lower end +0.066) -> **holds on the official set**.

## Files

run.log (everything that happened), state.json (all numbers), models/ (f4.pt, fusion/*.pt, ranker.pt), shards_*/ (candidate sets).