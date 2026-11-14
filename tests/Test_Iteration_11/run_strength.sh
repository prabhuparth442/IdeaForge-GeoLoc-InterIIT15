#!/usr/bin/env bash
# T11: three small runs (20 satellite-made photos each) at rendering strength 1, 2, 3, to compare their matching difficulty with thermal photos.
cd "$(dirname "$0")/notebooks" || exit 1
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
for s in 1 2 3; do
  echo "$(date) start strength $s"
  N_B=14 N_PNEO=6 STRENGTH=$s SEED=$((20+s)) TAG=strength$s taskset -c 0-9,12-31 $PY/jupyter nbconvert --to notebook --execute Sat_Photo_Dump.ipynb \
    --output Sat_Photo_Dump_executed_strength$s.ipynb --ExecutePreprocessor.timeout=-1 > ../results/log_strength$s.txt 2>&1
  echo "$(date) end strength $s exit $?"
done
