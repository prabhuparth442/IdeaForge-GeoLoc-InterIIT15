#!/usr/bin/env bash
# after the copies run: the same 500 high-altitude photos with a bar of 4 at map levels 9-11 (BAR_BY_LEVEL), everything else as the base run
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_highalt.txt
until grep -q "HIGHALT DONE" $S 2>/dev/null; do sleep 60; done
cd $W/Main_Pipeline_Headless
echo "$(date +%m-%d_%H:%M:%S) start highalt_bar4_lvl9up" >> $S
$PY ./run batch --dataset $H/dataset --preprocess $H/store --models $W/Main_Pipeline_Headless/models --out $W/out_highalt_bar4_lvl9up --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" --set "BAR_BY_LEVEL={9:4,10:4,11:4}" > $W/log_highalt_bar4_lvl9up.txt 2>&1
echo "$(date +%m-%d_%H:%M:%S) end highalt_bar4_lvl9up exit $?" >> $S
