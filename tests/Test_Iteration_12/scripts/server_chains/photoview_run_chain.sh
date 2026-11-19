#!/usr/bin/env bash
# Photo view in the pipeline (iteration 43), frozen base = XoFTR from 800 m, photo view from 1000 m. usage: photoview_run_chain.sh dev NAME BAR ["extra --set args"]
# dev = the 332 working photos of the new set. (Old official is below 351 m and NUDT below 320 m: the photo view is never active there, so those runs would equal the base.)
# Own status file and output folder (out_pv_NAME); nothing existing is overwritten.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_pv.txt
cd $W/Main_Pipeline_Headless
echo "$(date +%m-%d_%H:%M:%S) start $2" >> $S
$PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models models --out ../out_pv_$2 --mode full --hint-jitter-m 5000 \
  --set "LEVELS=[3,4,5,6,7,8,9,10,11]" --set XOFTR_FROM_HEIGHT_M=800 --set PHOTO_VIEW_FROM_HEIGHT_M=1000 --set PHOTO_VIEW_MIN_INLIERS=$3 $4 > ../log_pv_$2.txt 2>&1
echo "$(date +%m-%d_%H:%M:%S) end $2 exit $?" >> $S
