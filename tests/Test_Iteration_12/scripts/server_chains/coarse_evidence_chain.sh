#!/usr/bin/env bash
# Coarse_Evidence.ipynb on the three sets, DINO off. Two chains at once at most: A = new set then NUDT, B = old official.
# usage: coarse_evidence_chain.sh A|B
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_ce.txt
cd $W; mkdir -p ce
export CE_HEADLESS=$W/Main_Pipeline_Headless CE_OUT=$W/ce
run() { tag=$1; echo "$(date +%m-%d_%H:%M:%S) start $tag" >> $S
  $PY -m jupyter nbconvert --to notebook --execute Coarse_Evidence.ipynb --output ce/CE_$tag.ipynb --ExecutePreprocessor.timeout=36000 > $W/log_ce_$tag.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $tag exit $?" >> $S; }
if [ "$1" = A ]; then
  CE_DATASET=$H/dataset_dev CE_STORE=$H/store CE_LEVELS=3,4,5,6,7,8,9,10,11 CE_TAG=highalt_dev332 run highalt_dev332
  CE_DATASET=$W/nudt_hl/dataset CE_STORE=$W/nudt_hl/store CE_LEVELS=3,4,5 CE_GEOMETRY="yaw_offset=0,scale=0.8,attitude=False" CE_TAG=nudt300 run nudt300
else
  CE_DATASET=$W/Submission_Files/dev_data/dataset CE_STORE=$W/Submission_Files/dev_store CE_LEVELS=3,4,5,6,7 CE_TAG=official500 run official500
fi
echo "$(date +%m-%d_%H:%M:%S) CE chain $1 DONE" >> $S
