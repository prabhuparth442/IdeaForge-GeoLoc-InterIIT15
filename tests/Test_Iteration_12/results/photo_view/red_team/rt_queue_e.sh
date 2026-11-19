#!/usr/bin/env bash
# robustness arms: render-match-fit rounds and prior width on the bad-metadata datasets (and the control, to see the cost)
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; L="[3,4,5,6,7,8,9,10,11]"; R=$W/rt/rt_run.sh; D=$W/rt/data
while pgrep -f "rt_queue_b.sh|rt_queue_d.sh" > /dev/null; do sleep 30; done
P1="--set PHOTO_VIEW_PRIOR_TILT_DEG=1.0 --set PHOTO_VIEW_PRIOR_YAW_DEG=2.0 --set PHOTO_VIEW_PRIOR_HEIGHT=0.02"
P2="--set PHOTO_VIEW_PRIOR_TILT_DEG=2.0 --set PHOTO_VIEW_PRIOR_YAW_DEG=4.0 --set PHOTO_VIEW_PRIOR_HEIGHT=0.05 --set PHOTO_VIEW_FIT_MAX_MOVE_SIGMA=6"
for n in noise2x scale_p10 tilt_bias2 noise3x control; do
  bash $R r2p1_$n $D/$n $H/store "$L" 5000 0 "--set PHOTO_VIEW_FIT_ROUNDS=2 $P1"
  bash $R r2wide_$n $D/$n $H/store "$L" 5000 0 "--set PHOTO_VIEW_FIT_ROUNDS=2 $P2"
done
