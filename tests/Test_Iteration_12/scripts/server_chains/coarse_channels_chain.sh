#!/usr/bin/env bash
# Coarse_Channels.ipynb (dense structural channels, decoy control). usage: coarse_channels_chain.sh A|B
# A = new set (332 photos); B = official (first 150) then NUDT (first 150). Two chains at once at most.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_ch.txt
cd $W; mkdir -p ch
export CH_HEADLESS=$W/Main_Pipeline_Headless CH_OUT=$W/ch
run() { tag=$1; echo "$(date +%m-%d_%H:%M:%S) start $tag" >> $S
  $PY -m jupyter nbconvert --to notebook --execute Coarse_Channels.ipynb --output ch/CH_$tag.ipynb --ExecutePreprocessor.timeout=72000 > $W/log_ch_$tag.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $tag exit $?" >> $S; }
if [ "$1" = A ]; then
  CH_DATASET=$H/dataset_dev CH_STORE=$H/store CH_LEVELS=3,4,5,6,7,8,9,10,11 CH_TAG=highalt_dev332 run highalt_dev332
else
  CH_DATASET=$W/Submission_Files/dev_data/dataset CH_STORE=$W/Submission_Files/dev_store CH_LEVELS=3,4,5,6,7 CH_LIMIT=150 CH_TAG=official150 run official150
  CH_DATASET=$W/nudt_hl/dataset CH_STORE=$W/nudt_hl/store CH_LEVELS=3,4,5 CH_LIMIT=150 CH_GEOMETRY="yaw_offset=0,scale=0.8,attitude=False" CH_TAG=nudt150 run nudt150
fi
echo "$(date +%m-%d_%H:%M:%S) CH chain $1 DONE" >> $S
