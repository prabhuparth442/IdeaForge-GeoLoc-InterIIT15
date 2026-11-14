#!/usr/bin/env bash
# T10 GPU jobs, after the T9 altered-photo dumps: (1) stage-cache check, (2) fresh NUDT 808 draw (before any T10 rule is chosen),
# (3) coarse / DINO mixture on the standard queries.
cd "$(dirname "$0")/notebooks" || exit 1
while pgrep -f run_perturbed_dumps.sh > /dev/null; do sleep 60; done
PY=/home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin
R=$(cd ../results && pwd)
run () {   # $1 tag, then env assignments
  tag=$1; shift
  echo "$(date) start $tag"
  env "$@" TAG=$tag taskset -c 0-9,12-31 $PY/jupyter nbconvert --to notebook --execute Rich_Dump_Cached.ipynb \
    --output Rich_Dump_Cached_executed_$tag.ipynb --ExecutePreprocessor.timeout=-1 > $R/log_dump_$tag.txt 2>&1
  echo "$(date) end $tag exit $?"
}
run cache_first GATE_DATA=official POOL_PHOTOS=500 PHOTOS_FILE=$R/photos_cache_check.csv CACHE=$R/cache/official SELECT=dino QUERIES=all
run cache_reuse GATE_DATA=official POOL_PHOTOS=500 PHOTOS_FILE=$R/photos_cache_check.csv CACHE=$R/cache/official SELECT=dino QUERIES=all
run cache_fft GATE_DATA=official POOL_PHOTOS=500 PHOTOS_FILE=$R/photos_cache_check.csv CACHE=$R/cache/official_fftcheck SELECT=dino QUERIES=all SHARE_FFT=1
run nudt_fresh808 GATE_DATA=nudt POOL_PHOTOS=2400 NUDT_SEED=808 N_NUDT=300 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE=$R/exclude_fresh404_505_606_707.csv CACHE=$R/cache/nudt808 SELECT=dino QUERIES=all
run mix_official_try GATE_DATA=official POOL_PHOTOS=500 PHOTOS_FILE=$R/photos_official_try.csv CACHE=$R/cache/official SELECT=mix_7_3 QUERIES=std
run mix_nudt101 GATE_DATA=nudt POOL_PHOTOS=500 PHOTOS_FILE=$R/photos_nudt101_sub.csv CACHE=$R/cache/nudt101 SELECT=mix_7_3 QUERIES=std
