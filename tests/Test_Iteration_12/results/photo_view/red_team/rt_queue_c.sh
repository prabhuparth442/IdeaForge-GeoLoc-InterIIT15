#!/usr/bin/env bash
# slot 2 after queue B: the submitted base on the metadata-error arms (is PV-X worse than base when the metadata is worse?)
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; L="[3,4,5,6,7,8,9,10,11]"; R=$W/rt/rt_run.sh; D=$W/rt/data
while pgrep -f rt_queue_b.sh > /dev/null; do sleep 30; done
B="--set XOFTR_FROM_HEIGHT_M=None --set PHOTO_VIEW_FROM_HEIGHT_M=None --set PHOTO_VIEW_FIT_POSE=False"
for n in noise2x noise3x scale_p10 scale_m10 tilt_bias2 yaw_bias3 blur2 invert; do bash $R base_$n $D/$n $H/store "$L" 5000 0 "$B"; done
