#!/usr/bin/env bash
W=/mnt/e/IdeaForge/work; PY=~/training/ideaforge/venv-geoloc/bin/python; cd $W/Main_Pipeline_V3
for arm in pvx r2wide; do
 if [ $arm = r2wide ]; then X="--set PHOTO_VIEW_FIT_ROUNDS=2 --set PHOTO_VIEW_PRIOR_TILT_DEG=2.0 --set PHOTO_VIEW_PRIOR_YAW_DEG=4.0 --set PHOTO_VIEW_PRIOR_HEIGHT=0.05 --set PHOTO_VIEW_FIT_MAX_MOVE_SIGMA=6"; T=0.4; Y=0.8; Hh=0.008; else X=""; fi
 $PY ./run batch --dataset $W/nudt_hl/dataset --preprocess $W/nudt_hl/store --models models --out ../out_rt_nudt_$arm --mode full --hint-jitter-m 5000 --seed 0 \
  --set "LEVELS=[3,4,5]" --set 'GEOMETRY={"attitude": False, "pitch_sign": -1, "roll_sign": -1, "yaw_offset": 0.0, "scale": 0.8, "height_offset": 0.0}' \
  --set XOFTR_FROM_HEIGHT_M=0 --set PHOTO_VIEW_FROM_HEIGHT_M=0 --set PHOTO_VIEW_MIN_INLIERS=12 --set PHOTO_VIEW_FIT_POSE=True \
  --set PHOTO_VIEW_PRIOR_TILT_DEG=0.4 --set PHOTO_VIEW_PRIOR_YAW_DEG=0.8 --set PHOTO_VIEW_PRIOR_HEIGHT=0.008 $X > ../log_rt_nudt_$arm.txt 2>&1
done
