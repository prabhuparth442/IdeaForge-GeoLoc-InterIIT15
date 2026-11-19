#!/usr/bin/env bash
# slot 2: perturbation arms on every 2nd dev photo
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; L="[3,4,5,6,7,8,9,10,11]"; R=$W/rt/rt_run.sh; D=$W/rt/data
for n in control noise2x noise3x noise1x scale_p10 scale_m10 yaw_bias3 tilt_bias2 blur2 lowcontrast noise12 invert sidecrop alt1.25 alt1.50 alt2.00; do bash $R $n $D/$n $H/store "$L" 5000 0; done
P="--set PHOTO_VIEW_PRIOR_TILT_DEG=1.0 --set PHOTO_VIEW_PRIOR_YAW_DEG=2.0 --set PHOTO_VIEW_PRIOR_HEIGHT=0.02"
bash $R noise2x_prior1 $D/noise2x $H/store "$L" 5000 0 "$P"
bash $R noise2x_nofit $D/noise2x $H/store "$L" 5000 0 "--set PHOTO_VIEW_FIT_POSE=False"
bash $R control_base $D/control $H/store "$L" 5000 0 "--set XOFTR_FROM_HEIGHT_M=None --set PHOTO_VIEW_FROM_HEIGHT_M=None --set PHOTO_VIEW_FIT_POSE=False"
