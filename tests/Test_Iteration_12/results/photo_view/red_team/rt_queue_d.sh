#!/usr/bin/env bash
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; L="[3,4,5,6,7,8,9,10,11]"; R=$W/rt/rt_run.sh; D=$W/rt/data
while pgrep -f rt_queue_b.sh > /dev/null; do sleep 30; done
for n in invert2 lowcontrast2 noise2 blur2b; do bash $R $n $D/$n $H/store "$L" 5000 0; done
