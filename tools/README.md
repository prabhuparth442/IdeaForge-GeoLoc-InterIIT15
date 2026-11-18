# tools

Small helpers: `run_capped.sh` runs one heavy job inside a 6 GB / 8 core cap (use it for notebooks and dumps on the laptop), `pack_dataset.py` packs photos and metadata into one file.

| file | what |
|---|---|
| `pack_dataset.py` | Pack the photos + metadata.csv into ONE file, and read it back. |
| `run_capped.sh` | Run one heavy job (notebook execution, GPU dump, sweep) inside a memory and CPU cap so it cannot take the whole PC down. |

Repo map: `REPO_MAP.md` at the repo root.
