#!/usr/bin/env bash
# The real Stage_Evaluation.ipynb (same sections, tables and figures as always) at scale on three data sets, one after the other. The executed notebook is the report.
# 1 high-altitude working set (332 photos, heights 100-2017 m, map levels 3-11, height groups 100-300, 300-500, 500-800, 800-1000, 1000-1300, 1300-1600, 1600-2100)
# 2 old official dev set (150 photos)   3 NUDT valid split (150 photos)
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_stageeval.txt; OUT=$W/reports_stageeval
mkdir -p $OUT; cd $W/repo
export NUDT_DIR=$W/NUDT OMP_NUM_THREADS=4
run() { name=$1; shift
  echo "$(date +%m-%d_%H:%M:%S) start $name" >> $S
  env "$@" timeout 20000 $PY -m jupyter nbconvert --to notebook --execute Stage_Evaluation.ipynb --output $OUT/StageEval_$name.ipynb --ExecutePreprocessor.timeout=19000 > $W/log_stageeval_$name.txt 2>&1
  code=$?; echo "$(date +%m-%d_%H:%M:%S) end $name exit $code" >> $S
  $PY -m jupyter nbconvert --to html $OUT/StageEval_$name.ipynb > /dev/null 2>&1
  mkdir -p $OUT/tables_$name; cp $W/repo/tests/Stage_Evaluation/*_$name*.csv $OUT/tables_$name/ 2>/dev/null; }
run highalt_dev332 DEV_DATA_DIR=$H/stage_root SE_DATA=official SE_N_PHOTOS=1000 SE_N_EVAL=332 SE_TAG=_highalt_dev332 SE_MAP=custom SE_MAP_FILES=$H/map/2684532111/BAND.tif,$H/map/2684532121/BAND.tif,$H/map/2684532171/BAND.tif SE_LEVELS=3,4,5,6,7,8,9,10,11 SE_BAND_EDGES=100,300,500,800,1000,1300,1600,2100
run official150 DEV_DATA_DIR=$W/repo/dataset SE_DATA=official SE_N_PHOTOS=500 SE_N_EVAL=150 SE_TAG=_official150
run nudt_valid150 SE_DATA=nudt SE_N_PHOTOS=1425 SE_N_EVAL=150 SE_NUDT_SPLIT=valid SE_TAG=_nudt_valid150
echo "$(date +%m-%d_%H:%M:%S) STAGEEVAL DONE" >> $S
