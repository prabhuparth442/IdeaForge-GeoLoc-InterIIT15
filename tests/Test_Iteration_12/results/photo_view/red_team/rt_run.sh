#!/usr/bin/env bash
# Red-team run of PV-X (Main_Pipeline_V3) on one dataset. usage: rt_run.sh NAME DATASET_DIR STORE LEVELS JITTER SEED ["extra args"]  (status in status_rt.txt, output out_rt_NAME)
W=/mnt/e/IdeaForge/work; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_rt.txt
cd $W/Main_Pipeline_V3
echo "$(date +%m-%d_%H:%M:%S) start $1" >> $S
COMMON="--set XOFTR_FROM_HEIGHT_M=0 --set PHOTO_VIEW_FROM_HEIGHT_M=0 --set PHOTO_VIEW_MIN_INLIERS=12 --set PHOTO_VIEW_FIT_POSE=True --set PHOTO_VIEW_PRIOR_TILT_DEG=0.4 --set PHOTO_VIEW_PRIOR_YAW_DEG=0.8 --set PHOTO_VIEW_PRIOR_HEIGHT=0.008"
$PY ./run batch --dataset $2 --preprocess $3 --models models --out ../out_rt_$1 --mode full --hint-jitter-m $5 --seed $6 --set "LEVELS=$4" $COMMON $7 > ../log_rt_$1.txt 2>&1
echo "$(date +%m-%d_%H:%M:%S) end $1 exit $?" >> $S
