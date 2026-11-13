#!/usr/bin/env bash
# Test_Iteration_7 blind set: 300 NUDT photos never used (outside the 500-photo pool, fresh 404 and fresh 505).
# Kept blind until results/frozen_rules_t7.json is written. Uses the Test_Iteration_6 dump notebook unchanged.
cd "$(dirname "$0")/../Test_Iteration_6/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
R=$(pwd)/../../Test_Iteration_7/results
echo "$(date) start nudt_fresh606"
GATE_DATA=nudt NUDT_SEED=606 N_NUDT=300 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE=$R/exclude_fresh404_505.csv \
  taskset -c 0-9,12-31 $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb \
  --output ../../Test_Iteration_7/notebooks/Rich_Dump_Run_executed_nudt_fresh606.ipynb \
  --ExecutePreprocessor.timeout=-1 > $R/log_nudt_fresh606.txt 2>&1
echo "$(date) end nudt_fresh606 exit $?"
