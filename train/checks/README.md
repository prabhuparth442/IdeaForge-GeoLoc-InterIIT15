# train/checks

Blocks for the decision and replay experiments (Test_Iteration_6 to 11): the rich dump of every candidate, offline replay of the answer-or-skip decision, hold-outs, altered photos, judges. Most files need no GPU and make no plots; the one-line description of each is its own docstring.

| file | what |
|---|---|
| `candidate_judge.py` | a candidate-level judge for the answer-or-skip decision, trained on every fine-matched candidate (not only the chosen one), plus the |
| `decision_gate.py` | learn the answer-or-skip gate from per-photo results (no plotting, no GPU). |
| `decision_signals.py` | every signal available after fine matching, for every candidate of a rich dump (train/checks/rich_dump.py). No GPU, no plotting. |
| `eval_tools.py` | evaluation bookkeeping for Test_Iteration_10. No GPU, no plotting. |
| `full_eval.py` | Final, honest check: run the whole search over the FULL score maps (not candidate lists) for photos never used to learn. |
| `gate_features.py` | the per-photo numbers a decision gate can look at, from one heading-query run (no plotting, no GPU). |
| `heading_results.py` | turn the queries of one photo into table rows for the heading-query test (no plotting, no GPU). |
| `holdout.py` | honest hold-outs for Test_Iteration_8. No GPU, no plotting. |
| `overnight_report.py` | Writes report.md from the results saved in state.json. |
| `part_maps.py` | coarse score maps of the whole photo and of parts of it (ideas planner, second list: ideas 3, 4, 7, 6). |
| `perturb.py` | altered copies of a drone photo and of its metadata, to red-team the pipeline and the decision on inputs unlike the dev photos. |
| `pose_bias.py` | a constant attitude bias in the final position, fitted on answered dev photos and removed (no GPU, no plotting). |
| `query_policy.py` | time-aware query policies, replayed offline from a rich dump (all 27 queries stored). No GPU, no plotting. |
| `replay.py` | replay the decision of agreement_query_pipeline.py offline from a rich dump (train/checks/rich_dump.py). No GPU, no plotting. |
| `rich_dump.py` | a "rich" candidate dump. Same queries and answers as HeadingQueryPipeline, but every candidate also keeps its raw LoFTR matches, so any |
| `sat_photos.py` | drone-view "photos" rendered from SATELLITE imagery only (Test_Iteration_11), so a learned decision can be trained without any |
| `scale_robust.py` | inlier-scale robustness of the candidate judge (Test_Iteration_10, part 2). No GPU, no plotting, no multi-query agreement. |
| `second_look.py` | a second, better-aimed LoFTR look at one candidate (needs the GPU and a set-up pipeline). No plotting. |
| `stage_cache.py` | stage caching for the research runner (Test_Iteration_10, 3c), and the candidate selection it makes cheap to vary (3b). |
| `synthetic_report.py` | report.md for the synthetic-training run: what was done, and the real-photo results side by side. |
| `two_tier.py` | the two-tier answer rule of Test_Iteration_8 (no learned model, three numbers set on NUDT only), plus the label-free null-calibrated cut |
| `vote_decision.py` | re-make the answer-or-skip decision offline from a dump of every query's candidates (no GPU, no plotting). |

Related: tests/Test_Iteration_6 to 11 and tests/Test_Iteration_7/OURS_CHECK.md. Repo map: `REPO_MAP.md` at the repo root.
