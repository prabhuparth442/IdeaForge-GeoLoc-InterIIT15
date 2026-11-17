#!/usr/bin/env bash
# DINO off above 800 m, and a DINO weight of 3, on the 332 working photos of the new set (nothing else changed, bar 8, no copies). Reference: base on the same photos.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_dino.txt
cd $W/Main_Pipeline_Headless
run() { name=$1; shift
  echo "$(date +%m-%d_%H:%M:%S) start $name" >> $S
  $PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models $W/Main_Pipeline_Headless/models --out $W/out_dev_$name --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" "$@" > $W/log_dev_$name.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $name exit $?" >> $S; }
run dino_off_800 --set DINO_ONLY_BELOW_M=800.0
run dino_w3 --set DINO_WEIGHT=3.0
echo "$(date +%m-%d_%H:%M:%S) DINO CHAIN DONE" >> $S
