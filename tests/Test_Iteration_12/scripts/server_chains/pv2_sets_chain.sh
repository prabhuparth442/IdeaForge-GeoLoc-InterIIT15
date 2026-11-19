#!/usr/bin/env bash
# Same as pv_sets_chain.sh but runs the separate pipeline copy Main_Pipeline_V2 (iteration 45 and later experiments) so a queue running the first copy is not disturbed. usage: pv2_sets_chain.sh dev|official|nudt NAME GATE BAR ["extra --set args"]
# Base = XoFTR from 800 m, PHOTO_VIEW_FROM_HEIGHT_M=GATE (0 = every photo), PHOTO_VIEW_MIN_INLIERS=BAR, PHOTO_VIEW_FIT_POSE=True unless extra args say otherwise (extra args come last and win).
# Own status file and output folder (out_pv2_SET_NAME); nothing existing is overwritten.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_pv2.txt
cd $W/Main_Pipeline_V2
echo "$(date +%m-%d_%H:%M:%S) start $1_$2" >> $S
COMMON="--set XOFTR_FROM_HEIGHT_M=800 --set PHOTO_VIEW_FROM_HEIGHT_M=$3 --set PHOTO_VIEW_MIN_INLIERS=$4 --set PHOTO_VIEW_FIT_POSE=True"
case $1 in
  dev) $PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models models --out ../out_pv2_$1_$2 --mode full --hint-jitter-m 5000 \
        --set "LEVELS=[3,4,5,6,7,8,9,10,11]" $COMMON $5 > ../log_pv2_$1_$2.txt 2>&1 ;;
  official) $PY ./run batch --dataset $W/Submission_Files/dev_data/dataset --preprocess $W/Submission_Files/dev_store --models models --out ../out_pv2_$1_$2 --mode full --hint-jitter-m 5000 \
        $COMMON $5 > ../log_pv2_$1_$2.txt 2>&1 ;;
  nudt) $PY ./run batch --dataset $W/nudt_hl/dataset --preprocess $W/nudt_hl/store --models models --out ../out_pv2_$1_$2 --mode full --hint-jitter-m 5000 \
        --set "LEVELS=[3,4,5]" --set 'GEOMETRY={"attitude": False, "pitch_sign": -1, "roll_sign": -1, "yaw_offset": 0.0, "scale": 0.8, "height_offset": 0.0}' $COMMON $5 > ../log_pv2_$1_$2.txt 2>&1 ;;
esac
echo "$(date +%m-%d_%H:%M:%S) end $1_$2 exit $?" >> $S
