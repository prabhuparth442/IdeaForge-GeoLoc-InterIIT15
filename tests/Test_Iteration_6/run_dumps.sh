#!/usr/bin/env bash
# Executes Rich_Dump_Run.ipynb for every set, one after the other, pinned off CPUs 10/11 (core 20 logged machine-check errors on 2026-11-13).
# Resumable: the notebook skips photos already saved. Stops when the disk has under 1 GB free.
cd "$(dirname "$0")/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
CPUS=0-9,12-31
run() {  # $1 = output suffix, then env assignments
  local name=$1; shift
  local free_kb=$(df --output=avail -k . | tail -1)
  if [ "$free_kb" -lt 1000000 ]; then echo "$(date) STOP: under 1 GB free before $name"; exit 2; fi
  echo "$(date) start $name"
  env "$@" taskset -c $CPUS $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb \
      --output Rich_Dump_Run_executed_$name.ipynb --ExecutePreprocessor.timeout=-1 > ../results/log_$name.txt 2>&1
  echo "$(date) end $name exit $?"
}
run smoke   GATE_DATA=nudt NUDT_SEED=101 N_LIMIT=3 TAG=smoke || exit 1
run nudt101 GATE_DATA=nudt NUDT_SEED=101
run official GATE_DATA=official
run nudt202 GATE_DATA=nudt NUDT_SEED=202 EXCLUDE_SEEDS=101 N_NUDT=200
run nudt303 GATE_DATA=nudt NUDT_SEED=303 EXCLUDE_SEEDS=101,202 N_NUDT=200
echo "$(date) all done"
