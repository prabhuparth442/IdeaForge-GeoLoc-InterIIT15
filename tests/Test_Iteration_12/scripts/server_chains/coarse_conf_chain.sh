#!/usr/bin/env bash
# coarse-only confidence run (Coarse_Confidence.ipynb): new set working photos (332), then the old official set (500). No LoFTR, no DINO, ~1 s per photo.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_cc.txt
cd $W; mkdir -p cc
export CC_HEADLESS=$W/Main_Pipeline_Headless CC_OUT=$W/cc
run() { tag=$1; echo "$(date +%m-%d_%H:%M:%S) start $tag" >> $S
  $PY -m jupyter nbconvert --to notebook --execute Coarse_Confidence.ipynb --output cc/CC_$tag.ipynb --ExecutePreprocessor.timeout=3600 > $W/log_cc_$tag.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $tag exit $?" >> $S; }
CC_DATASET=$H/dataset_dev CC_STORE=$H/store CC_LEVELS=3,4,5,6,7,8,9,10,11 CC_TAG=highalt_dev332 run highalt_dev332
CC_DATASET=$W/Submission_Files/dev_data/dataset CC_STORE=$W/Submission_Files/dev_store CC_LEVELS=3,4,5,6,7 CC_TAG=official500 run official500
echo "$(date +%m-%d_%H:%M:%S) CC DONE" >> $S
