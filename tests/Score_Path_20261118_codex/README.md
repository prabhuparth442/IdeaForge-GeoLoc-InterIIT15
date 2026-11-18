# Score path analysis, 18 November 2026

Read [REPORT_codex.md](REPORT_codex.md), then [Score_Path_Analysis_executed_codex.ipynb](Score_Path_Analysis_executed_codex.ipynb). The notebook recomputes current development losses by height band and map level and bounds how much recovery/precision/model selection could provide. It runs no models and reads no locked-test result CSVs or images.

`PROTOCOL_codex.md` records the analysis scope. Small reusable blocks are in `changed_src/score_budget_codex.py`; generated tables, two plots and input hashes are in `results/`. To repeat, run `Score_Path_Analysis_codex.ipynb` in `venv-geoloc` through `tools/run_capped.sh` using saved result files. No server connection or GPU is needed.

No pipeline, settings or weight changes. Delete this folder independently to remove the analysis; shared docs contain links only. Notion mirror pending because the connector is unavailable.
