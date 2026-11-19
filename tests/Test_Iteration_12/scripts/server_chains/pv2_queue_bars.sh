#!/usr/bin/env bash
# Queue 2 (2026-11-18 night): photo view on every photo (gate 0), prior scale 0.4, the agreement rule OFF, bar 12 / 16 / 20, on the three sets. Runs the Main_Pipeline_V2 copy, one run after the other.
W=/mnt/e/IdeaForge/work; cd $W
P="--set PHOTO_VIEW_PRIOR_TILT_DEG=0.4 --set PHOTO_VIEW_PRIOR_YAW_DEG=0.8 --set PHOTO_VIEW_PRIOR_HEIGHT=0.008 --set USE_AGREEMENT=False"
for bar in 12 16 20; do
  for set in official nudt dev; do
    bash pv2_sets_chain.sh $set b${bar}na 0 $bar "$P"
  done
done
echo "$(date +%m-%d_%H:%M:%S) queue2 finished" >> $W/status_pv2.txt
