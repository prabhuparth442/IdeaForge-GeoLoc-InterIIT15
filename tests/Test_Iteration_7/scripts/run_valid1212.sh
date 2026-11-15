#!/usr/bin/env bash
# Rich dump of 150 photos of the NUDT VALID split (seed 1212): the confirmation draw of results/preregistered_guard.json. Memory-capped.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE/notebooks" || exit 1
echo "$(date) start nudt_valid1212"
"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=nudt NUDT_SPLIT=valid NUDT_SEED=1212 N_NUDT=150 POOL_PHOTOS=1425 TAG=nudt_valid1212 \
  "$REPO/../venv-geoloc/bin/jupyter" nbconvert --to notebook --execute Rich_Dump_Run_Valid.ipynb --output Rich_Dump_Run_executed_nudt_valid1212.ipynb \
  --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_nudt_valid1212.txt" 2>&1
echo "$(date) end nudt_valid1212 exit $?"
