#!/usr/bin/env bash
# T9 red team: altered photos through the unchanged pipeline (GPU). 60 official (both halves) + 60 NUDT fresh 707 photos x 7 alterations.
cd "$(dirname "$0")/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
R=$(cd ../results && pwd)
run () {   # $1 data, $2 pool, $3 photos file, $4 tag, $5 kinds
  echo "$(date) start $4 $5"
  GATE_DATA=$1 POOL_PHOTOS=$2 PHOTOS_FILE=$R/$3 BASE_TAG=$4 KINDS=$5 taskset -c 0-9,12-31 $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Perturbed.ipynb \
    --output Rich_Dump_Perturbed_executed_$4.ipynb --ExecutePreprocessor.timeout=-1 > $R/log_dump_$4.txt 2>&1
  echo "$(date) end $4 exit $?"
}
K=${KINDS:-invert,blur,noise,lowres,gamma,yaw15,height20}
run official 500 photos_official.csv official $K
run nudt 2400 photos_nudt707.csv nudt707 $K
