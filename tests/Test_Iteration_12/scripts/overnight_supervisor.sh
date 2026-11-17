#!/usr/bin/env bash
# Overnight chain on the GPU server (2026-11-16). Idempotent: every dump resumes where it stopped, so a crash only costs the photo in progress.
# Waits for the official satellite map, then dumps the official 150 photos in four conditions, then runs the replay notebook per group and writes reports.
W=/mnt/e/IdeaForge/work; PY=~/training/ideaforge/venv-geoloc/bin/python; NB=$W/repo/tests/Test_Iteration_12/notebooks
S=$W/status_overnight.txt; REP=$W/reports; mkdir -p $REP
export DEV_DATA_DIR=$W/repo/dataset NUDT_DIR=$W/NUDT OUT_ROOT=$W/mega
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $S; }
say "supervisor start pid $$"
until [ "$(stat -c %s $W/repo/dataset/Dataset_Official_2/268234931/BAND.tif 2>/dev/null)" = "11751253221" ] && [ "$(stat -c %s $W/repo/dataset/Dataset_Official_2/268234921/BAND.tif 2>/dev/null)" = "11751253221" ]; do sleep 60; done
say "official map complete"
dump() { # dump <data> <tag> <kind> <photos file> <pool> [extra env...]
  local data=$1 tag=$2 kind=$3 ids=$4 pool=$5
  [ -f $W/mega/done_${tag}_${kind} ] && return 0
  for attempt in 1 2 3; do
    say "start dump $tag $kind (try $attempt)"
    MAX_PHOTOS=${MAXP:-100000} GATE_DATA=$data TAG=$tag KIND=$kind PHOTOS_FILE=$ids POOL_PHOTOS=$pool NUDT_SPLIT=valid $PY -m jupyter nbconvert --to notebook --execute $NB/Mega_Dump.ipynb --output $REP/Mega_Dump_${tag}_${kind}.ipynb --ExecutePreprocessor.timeout=-1 > $W/log_mega_${tag}_${kind}.txt 2>&1
    code=$?; say "end dump $tag $kind exit $code"
    [ $code = 0 ] && touch $W/mega/done_${tag}_${kind} && return 0
    sleep 30
  done
}
IDS=$W/repo/tests/Test_Iteration_12/results/ids_official150.csv
export MAXP=80   # official: 80 random photos per condition (143 queries are slow: about 1-2.5 min per photo)
for kind in clean blur invert; do dump official official150 $kind $IDS 500; done
# NUDT chain (mega_nudt.sh) writes nudt_done to status_mega.txt; wait for it before replaying NUDT
until grep -q nudt_done $W/status_mega.txt 2>/dev/null; do sleep 60; done
replay() { # replay <name> <comma list of dump folders>
  say "start replay $1"
  DUMP_ROOT=$W/mega/rich_dump SETS=$2 $PY -m jupyter nbconvert --to notebook --execute $NB/Mega_Replay_Figures.ipynb --output $REP/Mega_Replay_$1.ipynb --ExecutePreprocessor.timeout=-1 > $W/log_replay_$1.txt 2>&1
  say "end replay $1 exit $?"
  mkdir -p $REP/figures_$1; cp $W/repo/tests/Test_Iteration_12/results/figures/*.png $REP/figures_$1/ 2>/dev/null; cp $W/repo/tests/Test_Iteration_12/results/*.csv $REP/figures_$1/ 2>/dev/null
}
replay official official150_clean,official150_blur,official150_invert
replay nudt nudt_valid100_clean
dump official official150 height20 $IDS 500
replay official_with_height20 official150_clean,official150_blur,official150_invert,official150_height20
say "ALL DONE"
touch $W/OVERNIGHT_DONE
