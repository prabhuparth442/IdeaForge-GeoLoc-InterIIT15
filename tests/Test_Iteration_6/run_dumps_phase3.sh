#!/usr/bin/env bash
# Phase 3 (after phase 2): a truly unseen NUDT sample (200 photos from outside the 500-photo pool every earlier test used), then XoFTR on its
# hard photos. The earlier "seed 303" sample overlapped seed 202 (79 of 83 photos) because the pool was used up; it is not used as a test set.
cd "$(dirname "$0")" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
CPUS=0-9,12-31
until grep -q -E "phase 2 all done|STOP" results/run_dumps_phase2.log; do sleep 60; done
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
run nudt_fresh404 GATE_DATA=nudt NUDT_SEED=404 N_NUDT=200 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500
cd .. && taskset -c $CPUS $PY/python -c "
import sys; sys.path.insert(0, '../..'); sys.path.insert(0, '../../src')
import pandas as pd
from train.checks.replay import load_set, hard_photo_ids
ids = hard_photo_ids(load_set('results/rich_dump/nudt_fresh404'))
pd.DataFrame(dict(photo=ids)).to_csv('results/hard_nudt_fresh404.csv', index=False); print('fresh404', len(ids), 'hard photos')" && cd notebooks
R=$(pwd)/../results
run xoftr_nudt_fresh404 GATE_DATA=nudt NUDT_SEED=404 N_NUDT=200 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 MATCHER=xoftr PHOTOS_FILE=$R/hard_nudt_fresh404.csv
echo "$(date) phase 3 all done"
