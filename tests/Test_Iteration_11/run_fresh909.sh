#!/usr/bin/env bash
# T11: fresh NUDT 909 draw (pool 2400; the 500-pool and every 404/505/606/707/808 photo excluded), dumped before any T11 rule is chosen.
cd "$(dirname "$0")/../Test_Iteration_10/notebooks" || exit 1
# (the training dump finished before this ran)
R=$(cd ../../Test_Iteration_11/results && pwd)
echo "$(date) start nudt_fresh909"
GATE_DATA=nudt POOL_PHOTOS=2400 NUDT_SEED=909 N_NUDT=300 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE=$R/exclude_fresh404_to_808.csv \
  SELECT=dino QUERIES=all TAG=nudt_fresh909 taskset -c 0-9,12-31 /home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin/jupyter nbconvert --to notebook \
  --execute Rich_Dump_Cached.ipynb --output ../../Test_Iteration_11/notebooks/Rich_Dump_Cached_executed_nudt_fresh909.ipynb \
  --ExecutePreprocessor.timeout=-1 > $R/log_dump_nudt_fresh909.txt 2>&1
echo "$(date) end nudt_fresh909 exit $?"
