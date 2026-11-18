# train/datasets

Photo sets and training data: NUDT Thermal-UAV as pipeline input, the photo sets to test on, candidate sets and pseudo-thermal queries for training.

| file | what |
|---|---|
| `candidates.py` | Turn full score maps into one small training example per photo ("candidate set"). |
| `nudt_data.py` | NUDT Thermal-UAV (Changsha) as input for the existing pipeline: photos, camera, map, splits. |
| `photo_sets.py` | the photos to test on, as a TestSet, for the two data sets we have. |
| `place_picker.py` | Where to cut fake photos from: random places on the map, away from the photos we test on, spread over scene types. |
| `pseudo_thermal.py` | Pseudo-thermal queries made from ordinary satellite pictures, so the fusion can learn from as many places as we like. |
| `smoke_world.py` | A synthetic stand-in for NUDT (random texture map, photos cut from it) so the whole launcher can be tried in minutes, no dataset. |
| `synthetic_candidates.py` | Candidate sets from pseudo-thermal queries, in the same shard format as candidates.dump_split. |

Related: tests/Test_Iteration_1. Repo map: `REPO_MAP.md` at the repo root.
