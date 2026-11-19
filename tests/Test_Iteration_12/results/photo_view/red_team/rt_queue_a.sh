#!/usr/bin/env bash
# slot 1: null test (true place outside the 5.5 km disc), then speed replay
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; L="[3,4,5,6,7,8,9,10,11]"; R=$W/rt/rt_run.sh
bash $R null_dev $H/dataset_dev $H/store "$L" 12000 0
bash $R null_dev2 $H/dataset_dev $H/store "$L" 25000 0
for r in 2.4 5 10; do b=$(python3 -c "print(round(22/$r,2))"); bash $R speed_$r $W/rt/data/control $H/store "$L" 5000 0 "--set TIME_BUDGET_S=$b"; done
bash $R speed_1 $W/rt/data/control $H/store "$L" 5000 0
