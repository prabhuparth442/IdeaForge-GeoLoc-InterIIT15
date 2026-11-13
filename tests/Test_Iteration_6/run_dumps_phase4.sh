#!/usr/bin/env bash
# Phase 4: XoFTR rich dumps of the hard photos again (phase 2 failed on every photo: KeyError batch_indexes, fixed by train/checks/xoftr_batched.py).
cd "$(dirname "$0")/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
CPUS=0-9,12-31
until grep -q -E "phase 3 all done|STOP" ../results/run_dumps_phase3.log; do sleep 60; done
run() {
  local name=$1; shift
  echo "$(date) start $name"
  env "$@" taskset -c $CPUS $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb \
      --output Rich_Dump_Run_executed_$name.ipynb --ExecutePreprocessor.timeout=-1 > ../results/log_$name.txt 2>&1
  echo "$(date) end $name exit $?"
}
R=$(pwd)/../results
run xoftr_official GATE_DATA=official MATCHER=xoftr PHOTOS_FILE=$R/hard_official.csv
run xoftr_nudt101 GATE_DATA=nudt NUDT_SEED=101 MATCHER=xoftr PHOTOS_FILE=$R/hard_nudt_seed101.csv
run xoftr_nudt202 GATE_DATA=nudt NUDT_SEED=202 EXCLUDE_SEEDS=101 N_NUDT=200 MATCHER=xoftr PHOTOS_FILE=$R/hard_nudt_seed202.csv
echo "$(date) phase 4 all done"
