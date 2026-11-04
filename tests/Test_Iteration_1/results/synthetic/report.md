# Fusion trained on pseudo-thermal photos

Fake photos were cut from the Changsha map (away from the valid/test photos), turned into pseudo-thermal canvases and pushed through the same maps and candidate sets as real photos. Three fusion models are compared on REAL thermal photos:
- `fused:real_only`: trained on the real NUDT train photos (the earlier run)
- `fused:synth`: trained on fake photos only
- `fused:mixed`: fake photos plus the real train photos (each real photo counted 2 times)

Fake-photo strength chosen by comparing with real training photos: **2.0**. Pictures: `pseudo_samples.png`.

Best synth model, chosen on the real valid photos: `edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc` (monotone), valid top-1 0.196.
Best mixed model, chosen on the real valid photos: `edge_zncc+edge_cosine+edge_mag+edge_zncc_w+mind_zncc+learned_zncc` (gated), valid top-1 0.252.

## Real NUDT test photos (same city as the fake photos' map)

| model | top-1 <= 1 sigma | <= 2 sigma | <= 4 sigma | best of 25 <= 4 sigma | vs edge_zncc (4 sigma) | 95% interval |
|---|---|---|---|---|---|---|
| edge_zncc | 0.760 | 0.793 | 0.813 | 0.900 | +0.000 | +0.000 .. +0.000 |
| edge_cosine | 0.740 | 0.760 | 0.780 | 0.927 | -0.033 | -0.073 .. +0.000 |
| edge_mag | 0.220 | 0.227 | 0.240 | 0.513 | -0.573 | -0.647 .. -0.493 |
| edge_zncc_w | 0.713 | 0.740 | 0.753 | 0.900 | -0.060 | -0.107 .. -0.020 |
| mind_zncc | 0.720 | 0.753 | 0.760 | 0.873 | -0.053 | -0.107 .. +0.000 |
| learned_zncc | 0.760 | 0.807 | 0.820 | 0.940 | +0.007 | -0.040 .. +0.053 |
| fused:real_only | 0.813 | 0.853 | 0.860 | 0.940 | +0.047 | +0.013 .. +0.087 |
| fused:synth | 0.780 | 0.820 | 0.833 | 0.953 | +0.020 | -0.013 .. +0.053 |
| fused:mixed | 0.820 | 0.860 | 0.873 | 0.960 | +0.060 | +0.020 .. +0.107 |

Real NUDT test photos (top-1 within 4 sigma): synthetic only -0.027 against the model trained on real photos only; synthetic + real +0.013 against the model trained on real photos only. Differences smaller than the 95% intervals above can be chance.

## Official development photos on the small_sat_area map (another city: the generalisation check)

| model | top-1 <= 1 sigma | <= 2 sigma | <= 4 sigma | best of 25 <= 4 sigma | vs edge_zncc (4 sigma) | 95% interval |
|---|---|---|---|---|---|---|
| edge_zncc | 0.559 | 0.777 | 0.803 | 0.888 | +0.000 | +0.000 .. +0.000 |
| edge_cosine | 0.532 | 0.745 | 0.771 | 0.872 | -0.032 | -0.074 .. +0.005 |
| edge_mag | 0.218 | 0.340 | 0.346 | 0.617 | -0.457 | -0.537 .. -0.378 |
| edge_zncc_w | 0.548 | 0.761 | 0.787 | 0.840 | -0.016 | -0.043 .. +0.011 |
| mind_zncc | 0.574 | 0.798 | 0.835 | 0.931 | +0.032 | -0.016 .. +0.080 |
| learned_zncc | 0.511 | 0.830 | 0.862 | 0.952 | +0.059 | +0.021 .. +0.101 |
| fused:real_only | 0.617 | 0.867 | 0.888 | 0.952 | +0.085 | +0.048 .. +0.128 |
| fused:synth | 0.622 | 0.851 | 0.883 | 0.936 | +0.080 | +0.043 .. +0.122 |
| fused:mixed | 0.617 | 0.867 | 0.899 | 0.952 | +0.096 | +0.059 .. +0.144 |

Official development photos on the small_sat_area map (top-1 within 4 sigma): synthetic only -0.005 against the model trained on real photos only; synthetic + real +0.011 against the model trained on real photos only. Differences smaller than the 95% intervals above can be chance.


## Caveats
- Fake photos come from the very map they are matched against, so they are easier than real photos; the strength step only matches how often the truth wins, not every kind of difficulty.
- Only the official-set table says anything about another place. A gain on NUDT alone could still be Changsha.
- DINO is not part of this run: the fusion still has the same six maps.
