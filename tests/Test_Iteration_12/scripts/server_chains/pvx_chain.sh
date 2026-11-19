#!/usr/bin/env bash
# The photo-view configuration "PV-X" (frozen 2026-11-18 night) on any set, with a hint-jitter seed. usage: pvx_chain.sh dev|official|nudt|test NAME SEED ["extra --set args"]
# PV-X = XoFTR for every photo, photo view for every photo, bar 12, pose fit with priors x0.4 (0.4, 0.4, 0.8 degrees, 0.8 %), the agreement rule as in settings (on). Runs the copy Main_Pipeline_V3.
# test = the LOCKED test split (report only, never choose with it). Own status file and output folder (out_pvx_SET_NAME).
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_pvx.txt
cd $W/Main_Pipeline_V3
echo "$(date +%m-%d_%H:%M:%S) start $1_$2" >> $S
COMMON="--set XOFTR_FROM_HEIGHT_M=0 --set PHOTO_VIEW_FROM_HEIGHT_M=0 --set PHOTO_VIEW_MIN_INLIERS=12 --set PHOTO_VIEW_FIT_POSE=True --set PHOTO_VIEW_PRIOR_TILT_DEG=0.4 --set PHOTO_VIEW_PRIOR_YAW_DEG=0.8 --set PHOTO_VIEW_PRIOR_HEIGHT=0.008 --seed $3"
case $1 in
  dev) $PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models models --out ../out_pvx_$1_$2 --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" $COMMON $4 > ../log_pvx_$1_$2.txt 2>&1 ;;
  test) $PY ./run batch --dataset $H/dataset_test_LOCKED --preprocess $H/store --models models --out ../out_pvx_$1_$2 --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" $COMMON $4 > ../log_pvx_$1_$2.txt 2>&1 ;;
  official) $PY ./run batch --dataset $W/Submission_Files/dev_data/dataset --preprocess $W/Submission_Files/dev_store --models models --out ../out_pvx_$1_$2 --mode full --hint-jitter-m 5000 $COMMON $4 > ../log_pvx_$1_$2.txt 2>&1 ;;
  nudt) $PY ./run batch --dataset $W/nudt_hl/dataset --preprocess $W/nudt_hl/store --models models --out ../out_pvx_$1_$2 --mode full --hint-jitter-m 5000 \
        --set "LEVELS=[3,4,5]" --set 'GEOMETRY={"attitude": False, "pitch_sign": -1, "roll_sign": -1, "yaw_offset": 0.0, "scale": 0.8, "height_offset": 0.0}' $COMMON $4 > ../log_pvx_$1_$2.txt 2>&1 ;;
esac
echo "$(date +%m-%d_%H:%M:%S) end $1_$2 exit $?" >> $S
