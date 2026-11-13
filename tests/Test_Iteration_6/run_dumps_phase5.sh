#!/usr/bin/env bash
# Phase 5: a second fresh NUDT sample (300 photos never used: outside the 500-photo pool and outside fresh 404), kept blind until the final
# decision rule is fixed, because NUDT 202 and 404 have now both informed decisions. Then XoFTR on its hard photos is NOT run (XoFTR is not
# in the final rule unless the confirmation calls for it).
cd "$(dirname "$0")/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
R=$(pwd)/../results
echo "$(date) start nudt_fresh505"
GATE_DATA=nudt NUDT_SEED=505 N_NUDT=300 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE=$R/exclude_fresh404.csv \
  taskset -c 0-9,12-31 $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb --output Rich_Dump_Run_executed_nudt_fresh505.ipynb \
  --ExecutePreprocessor.timeout=-1 > ../results/log_nudt_fresh505.txt 2>&1
echo "$(date) end nudt_fresh505 exit $?"
