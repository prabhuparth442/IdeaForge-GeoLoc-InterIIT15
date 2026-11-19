#!/usr/bin/env bash
# r2wide vs PV-X on unseen (odd) photos with a new noise draw, then clean cost on official and NUDT
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; L="[3,4,5,6,7,8,9,10,11]"; R=$W/rt/rt_run.sh; D=$W/rt/data
P2="--set PHOTO_VIEW_FIT_ROUNDS=2 --set PHOTO_VIEW_PRIOR_TILT_DEG=2.0 --set PHOTO_VIEW_PRIOR_YAW_DEG=4.0 --set PHOTO_VIEW_PRIOR_HEIGHT=0.05 --set PHOTO_VIEW_FIT_MAX_MOVE_SIGMA=6"
if [ "$1" = a ]; then
 for n in oddcontrol oddnoise2x oddnoise3x oddscale_p10 oddtilt2; do bash $R $n $D/$n $H/store "$L" 5000 0; bash $R r2wide_$n $D/$n $H/store "$L" 5000 0 "$P2"; done
else
 bash $R r2wide_official $W/Submission_Files/dev_data/dataset $W/Submission_Files/dev_store "[3,4,5,6,7]" 5000 0 "$P2"
 bash $R r2wide_nudt $W/nudt_hl/dataset $W/nudt_hl/store "[3,4,5]" 5000 0 "$P2 --set 'GEOMETRY={\"attitude\": False, \"pitch_sign\": -1, \"roll_sign\": -1, \"yaw_offset\": 0.0, \"scale\": 0.8, \"height_offset\": 0.0}'"
fi
