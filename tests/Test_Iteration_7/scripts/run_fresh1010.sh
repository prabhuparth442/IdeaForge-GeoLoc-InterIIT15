#!/usr/bin/env bash
# Rich dump of 150 fresh NUDT photos (seed 1010) that no T6-T11 set and none of our earlier sets used (exclude_for_fresh1010.csv = his 404-808 and 909 draws,
# our 707 draw and the T7-seen photos; the 500-pool is skipped by FRESH_EXCLUDE_POOL). Run under the memory cap: tools/run_capped.sh.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE/notebooks" || exit 1
echo "$(date) start nudt_fresh1010"
"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=nudt NUDT_SEED=1010 N_NUDT=150 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 \
  EXCLUDE_IDS_FILE="$HERE/results/exclude_for_fresh1010.csv" TAG=nudt_fresh1010 \
  "$REPO/../venv-geoloc/bin/jupyter" nbconvert --to notebook --execute Rich_Dump_Run.ipynb --output Rich_Dump_Run_executed_nudt_fresh1010.ipynb \
  --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_nudt_fresh1010.txt" 2>&1
echo "$(date) end nudt_fresh1010 exit $?"
