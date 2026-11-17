#!/usr/bin/env bash
# Final fit one and two levels finer (REFINE_FINER_LEVELS) on the 332 working photos of the new set, nothing else changed (bar 8, no copies). Reference: base on the same photos.
# Starts when the DINO chain is done (one headless job at a time next to the Stage_Evaluation run).
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_refine.txt
until grep -q "DINO CHAIN DONE" $W/status_dino.txt 2>/dev/null; do sleep 30; done
cd $W/Main_Pipeline_Headless
run() { name=$1; shift
  echo "$(date +%m-%d_%H:%M:%S) start $name" >> $S
  $PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models $W/Main_Pipeline_Headless/models --out $W/out_dev_$name --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" "$@" > $W/log_dev_$name.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $name exit $?" >> $S; }
run refine1 --set REFINE_FINER_LEVELS=1
run refine2 --set REFINE_FINER_LEVELS=2
echo "$(date +%m-%d_%H:%M:%S) REFINE CHAIN DONE" >> $S
