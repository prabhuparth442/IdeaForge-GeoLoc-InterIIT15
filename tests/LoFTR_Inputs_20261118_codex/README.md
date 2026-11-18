# LoFTR input inspection

Open [REPORT_codex.md](REPORT_codex.md) for observations and [LoFTR_Input_Inspection_executed_codex.ipynb](LoFTR_Input_Inspection_executed_codex.ipynb) for executed visual evidence. [PROTOCOL_codex.md](PROTOCOL_codex.md) records the scope and decision rule before reconstruction.

Four height-selected development frames only; no training, new matcher inference or pipeline edits. Reconstructs actual 224 inputs and the existing optional direct 448 + cleaning path. Includes enlarged-224 controls, source-footprint coverage, exact arrays and hashes. Recomputes the existing Iteration 34 diagnostic separately from this visual work.

To reproduce, open `LoFTR_Input_Inspection_codex.ipynb` with the workspace `venv-geoloc`. Run through the repository's resource limiter. The notebook uses saved small satellite windows, so no server connection or full map extraction is needed. Current headless preparation code is imported directly; source hashes identify the version used. Source camera files must remain available in the new dataset.

This `_codex` folder is isolated and removable. Shared docs only add pointers. Notion mirror pending: no connector available.
