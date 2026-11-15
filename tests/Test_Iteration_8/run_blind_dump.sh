#!/usr/bin/env bash
# Test_Iteration_8 blind NUDT set: 300 fresh photos (pool 2400, the 500-pool and every 404 / 505 / 606 photo excluded).
# Dumped before any T8 rule is chosen; its signal table is built and scored only after results/preregistered_t8.json is written.
cd "$(dirname "$0")/../Test_Iteration_6/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
R=$(pwd)/../../Test_Iteration_8/results
echo "$(date) start nudt_fresh707"
GATE_DATA=nudt NUDT_SEED=707 N_NUDT=300 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE=$R/exclude_fresh404_505_606.csv \
  taskset -c 0-9,12-31 $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb \
  --output ../../Test_Iteration_8/notebooks/Rich_Dump_Run_executed_nudt_fresh707.ipynb \
  --ExecutePreprocessor.timeout=-1 > $R/log_nudt_fresh707.txt 2>&1
echo "$(date) end nudt_fresh707 exit $?"
