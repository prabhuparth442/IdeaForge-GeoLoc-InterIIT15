#!/usr/bin/env bash
# Coarse_Refine.ipynb (parabola + finer-level ZNCC around the coarse peak), no LoFTR, no DINO. usage: coarse_refine_chain.sh A|B
# A = new set; B = old official then NUDT. Two chains at once at most.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_cr.txt
cd $W; mkdir -p cr
export CR_HEADLESS=$W/Main_Pipeline_Headless CR_OUT=$W/cr
run() { tag=$1; echo "$(date +%m-%d_%H:%M:%S) start $tag" >> $S
  $PY -m jupyter nbconvert --to notebook --execute Coarse_Refine.ipynb --output cr/CR_$tag.ipynb --ExecutePreprocessor.timeout=36000 > $W/log_cr_$tag.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $tag exit $?" >> $S; }
if [ "$1" = A ]; then
  CR_DATASET=$H/dataset_dev CR_STORE=$H/store CR_LEVELS=3,4,5,6,7,8,9,10,11 CR_TAG=highalt_dev332 run highalt_dev332
else
  CR_DATASET=$W/Submission_Files/dev_data/dataset CR_STORE=$W/Submission_Files/dev_store CR_LEVELS=3,4,5,6,7 CR_TAG=official500 run official500
  CR_DATASET=$W/nudt_hl/dataset CR_STORE=$W/nudt_hl/store CR_LEVELS=3,4,5 CR_GEOMETRY="yaw_offset=0,scale=0.8,attitude=False" CR_TAG=nudt300 run nudt300
fi
echo "$(date +%m-%d_%H:%M:%S) CR chain $1 DONE" >> $S
