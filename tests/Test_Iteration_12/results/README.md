# tests/Test_Iteration_12/results

| path | what | made by |
|---|---|---|
| `stageeval/` | the three Stage_Evaluation reports (new high-altitude 332, official 150, NUDT valid 150), `.ipynb` + `.html` + `tables_*` | root `Stage_Evaluation.ipynb` |
| `coarse_confidence/` | per-query and per-photo coarse peak confidence (z, margin, votes) for the new set (332), official (500) and NUDT (300); iteration 25 / 25b | `notebooks/Coarse_Confidence.ipynb` |
| `coarse_evidence/` | LoFTR inliers at each coarse place, three sets, and the executed gate analysis; iteration 26 | `Coarse_Evidence.ipynb`, `Coarse_Evidence_Analysis.ipynb` |
| `coarse_averaging/` | executed averaging test (iteration 27) and its inputs (metadata, store.json) | `Coarse_Averaging.ipynb` |
| `coarse_refine/` | parabola and finer-level refinement of the coarse peak, three sets, executed analysis; iteration 28 | `Coarse_Refine.ipynb`, `Coarse_Refine_Analysis.ipynb` |
| `coarse_channels/` | dense structural channels and decoy control, three sets, executed analysis; iteration 29 | `Coarse_Channels.ipynb`, `Coarse_Channels_Analysis.ipynb` |
| `edge_features/` | figure of edge features at height next to the map at the true place, scale study, sigma grid for photo and map blur, tint and CLAHE figure, cleaning test, LoFTR at the true place; iterations 30-32 | `Edge_Features_By_Height.ipynb`, `Edge_Scale_Matching.ipynb`, `Photo_Tint_And_LoFTR.ipynb`, `Loftr_Map_Cleaning.ipynb`, `Loftr_Photo_Strength.ipynb` |
| `native_fine/` | headless runs: on the 332 working photos base, native-resolution LoFTR (35), XoFTR from 800 m and XoFTR + native (37); on the 168 LOCKED test photos base and XoFTR from 800 m (`test_details_*.csv`, iteration 38, report only), per band and per level | `Headless_Native_Analysis.ipynb` |
| `matchers/` | LoFTR, XoFTR and RoMa at the true place with a bar per matcher from wrong-place counts; iteration 36 | `Matchers_At_Height.ipynb` |
| `synthetic_audit/` | the pipeline on 202 photos rendered from the map with known poses (exact and noisy metadata) and the real photos' outputs for comparison; iteration 40 | `Synthetic_Geometry_Audit.ipynb` |
| `pose_from_matches/` | constrained camera-pose fit from the matches on the synthetic arms and the 332 real photos; iteration 41 (killed) | `Pose_From_Matches.ipynb` |
| `figures/`, `fig*.csv` | replay figures 1-9 and their tables | `Mega_Replay_Figures.ipynb` |
| `similarity/` | dataset similarity figures 1-6 | `Dataset_Similarity.ipynb` |
| `highalt_split.csv`, `highalt_dev_bands.csv` | 332 working / 168 locked test split and per-band scores | |
| `ids_official150.csv`, `ids_nudt_valid100.csv` | photo ids used for the 150 / 100 photo runs | |
| `server/` | replays copied back from the GPU server | |
| `rich_dump/`, `highalt_truth_diag/`, `flat_field_look.png` | local 143-query dumps, Fable's diagnostic outputs, the vignette-removal look | |

Never tune on the locked test split (168 photos). Repo map: `REPO_MAP.md`.
