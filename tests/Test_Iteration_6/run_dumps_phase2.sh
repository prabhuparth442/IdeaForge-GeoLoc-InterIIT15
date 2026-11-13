#!/usr/bin/env bash
# Phase 2 (after run_dumps.sh): XoFTR rich dumps of the hard photos (not answered by the 6 standard queries) of every set.
# Pinned off CPUs 10/11 (machine-check errors). Resumable.
cd "$(dirname "$0")" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
CPUS=0-9,12-31
until grep -q -E "all done|STOP" results/run_dumps.log; do sleep 60; done
echo "$(date) phase 2: writing hard-photo lists"
for s in official nudt_seed101 nudt_seed202 nudt_seed303; do
  taskset -c $CPUS $PY/python -c "
import sys; sys.path.insert(0, '../..'); sys.path.insert(0, '../../src')
import pandas as pd
from train.checks.replay import load_set, hard_photo_ids
ids = hard_photo_ids(load_set('results/rich_dump/$s'))
pd.DataFrame(dict(photo=ids)).to_csv('results/hard_$s.csv', index=False); print('$s', len(ids), 'hard photos')"
done
cd notebooks || exit 1
run() {
  local name=$1; shift
  local free_kb=$(df --output=avail -k . | tail -1)
  if [ "$free_kb" -lt 1000000 ]; then echo "$(date) STOP: under 1 GB free before $name"; exit 2; fi
  echo "$(date) start $name"
  env "$@" taskset -c $CPUS $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb \
      --output Rich_Dump_Run_executed_$name.ipynb --ExecutePreprocessor.timeout=-1 > ../results/log_$name.txt 2>&1
  echo "$(date) end $name exit $?"
}
R=$(pwd)/../results
run xoftr_official GATE_DATA=official MATCHER=xoftr PHOTOS_FILE=$R/hard_official.csv
run xoftr_nudt101 GATE_DATA=nudt NUDT_SEED=101 MATCHER=xoftr PHOTOS_FILE=$R/hard_nudt_seed101.csv
run xoftr_nudt202 GATE_DATA=nudt NUDT_SEED=202 EXCLUDE_SEEDS=101 N_NUDT=200 MATCHER=xoftr PHOTOS_FILE=$R/hard_nudt_seed202.csv
run xoftr_nudt303 GATE_DATA=nudt NUDT_SEED=303 EXCLUDE_SEEDS=101,202 N_NUDT=200 MATCHER=xoftr PHOTOS_FILE=$R/hard_nudt_seed303.csv
echo "$(date) phase 2 all done"
