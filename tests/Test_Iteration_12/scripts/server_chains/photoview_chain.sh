#!/usr/bin/env bash
# Map drawn in the photo's own view above 1000 m (iteration 43). usage: photoview_chain.sh TAG [extra env, e.g. EF_LIMIT=6]
# Runs notebooks/Map_In_Photo_View.ipynb on the 332 working photos. Own status file and output folder; nothing existing is overwritten.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_photoview.txt
cd $W/repo/tests/Test_Iteration_12/notebooks
echo "$(date +%m-%d_%H:%M:%S) start $1" >> $S
env EF_HEADLESS=$W/Main_Pipeline_Headless EF_DATASET=$H/dataset_dev EF_STORE=$H/store EF_OUT=$W/out_photoview_$1 EF_TAG=_$1 EF_MATCHERS=loftr,xoftr $2 \
  $PY -m jupyter nbconvert --to notebook --execute Map_In_Photo_View.ipynb --output $W/out_photoview_$1/Map_In_Photo_View_executed.ipynb --ExecutePreprocessor.timeout=-1 > $W/log_photoview_$1.txt 2>&1
echo "$(date +%m-%d_%H:%M:%S) end $1 exit $?" >> $S
