#!/usr/bin/env bash
# Rich dumps on our machine (RTX 3050): 150 random official photos, then 150 NUDT photos that no T6/T7 set used (fresh pool, seed 707).
cd "$(dirname "$0")/notebooks" || exit 1
V=../../../../venv-geoloc/bin
run() { local name=$1; shift; echo "$(date) start $name"
  env OMP_NUM_THREADS=6 "$@" $V/jupyter nbconvert --to notebook --execute Rich_Dump_Run.ipynb --output Rich_Dump_Run_executed_$name.ipynb --ExecutePreprocessor.timeout=-1 > ../results/logs/log_$name.txt 2>&1
  echo "$(date) end $name exit $?"; }
run official150 GATE_DATA=official PHOTOS_FILE=/home/voidness/Storage/Synced\ Folders/InterIIT\ Tech\ 15/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_7/results/photos_official150.csv TAG=official150
run nudt_fresh707 GATE_DATA=nudt NUDT_SEED=707 N_NUDT=150 POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE=/home/voidness/Storage/Synced\ Folders/InterIIT\ Tech\ 15/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_7/results/exclude_t7_seen_fresh.csv TAG=nudt_fresh707
echo "$(date) all done"
