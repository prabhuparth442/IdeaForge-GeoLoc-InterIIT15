#!/usr/bin/env bash
# Second look (GPU) on the five seen sets: re-centred tile, and re-centred + re-aimed query (heading + fitted angle, scale / fitted scale).
cd "$(dirname "$0")/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
run() {  # data pool dump_tag
  echo "$(date) start $3"
  GATE_DATA=$1 POOL_PHOTOS=$2 DUMP_TAG=$3 VARIANTS=recentre,aim_p taskset -c 0-9,12-19 $PY/jupyter nbconvert --to notebook --execute \
    Second_Look_Run.ipynb --output Second_Look_Run_executed_$3.ipynb --ExecutePreprocessor.timeout=-1 > ../results/log_second_look_$3.txt 2>&1
  echo "$(date) end $3 exit $?"
}
run official 500 official
run nudt 500 nudt_seed101
run nudt 500 nudt_seed202
run nudt 2400 nudt_fresh404
run nudt 2400 nudt_fresh505
