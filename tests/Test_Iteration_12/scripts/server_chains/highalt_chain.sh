#!/usr/bin/env bash
# High-altitude data (Rajasthan, 500 photos, heights up to ~1800 m): build a map store with levels 3-11, then run the headless pipeline without and with altered copies.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_highalt.txt
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $S; }
LEV='[3,4,5,6,7,8,9,10,11]'
mkdir -p $H/map $H/dataset/images
if [ ! -f $H/map/.extracted ]; then
  say "extract tiles"
  for t in 2684532111 2684532121 2684532171; do
    $PY - $H/sat_imagery/$t.zip $H/map <<"E"
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
E
  done
  touch $H/map/.extracted; say "tiles extracted"
fi
if [ ! -f $H/dataset/metadata.csv ]; then
  cp $H/extracted/d1/images/*.png $H/dataset/images/; cp $H/extracted/d1/images/metadata.csv $H/dataset/metadata.csv
fi
cd $W/Main_Pipeline_Headless
sed -i 's/0 <= level <= 10 for level in settings.LEVELS), "LEVELS must be whole numbers 0 to 10"/0 <= level <= 12 for level in settings.LEVELS), "LEVELS must be whole numbers 0 to 12"/' geoloc/check_settings.py
if [ ! -f $H/store/.done ]; then
  say "preprocess start"
  $PY ./run preprocess --map $H/map --preprocess $H/store --region 71.38,26.98,71.56,27.24 --set "LEVELS=$LEV" > $W/log_highalt_preprocess.txt 2>&1
  code=$?; say "preprocess end exit $code"; [ $code = 0 ] && touch $H/store/.done || exit 1
fi
run() { name=$1; shift
  say "start $name"
  $PY ./run batch --dataset $H/dataset --preprocess $H/store --models $W/Main_Pipeline_Headless/models --out $W/out_$name --mode full --hint-jitter-m 5000 --set "LEVELS=$LEV" "$@" > $W/log_$name.txt 2>&1
  say "end $name exit $?"; }
run highalt_base500
run highalt_copies500 --set USE_ALTERED_COPIES=True
say "HIGHALT DONE"
