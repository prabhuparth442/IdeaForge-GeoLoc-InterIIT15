#!/usr/bin/env bash
# Overnight chain for the ideas planner's second list (ideas 1, 2, 3, 4, 5, 6, 7), one job at a time, each inside the memory cap (tools/run_capped.sh: 6G, 8 of 12 cores).
# A failed job does not stop the chain. Logs: results/logs/second_list_<name>.txt, status lines in results/logs/second_list_status.txt.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin/jupyter"
LOGS="$HERE/results/logs"; STATUS="$LOGS/second_list_status.txt"
cd "$HERE/notebooks" || exit 1
NEVER_NUDT="$HERE/results/never_retrieved_valid1212.csv"; LOW_OFFICIAL="$HERE/results/low_altitude_skips_official150.csv"
NUDT="GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212"
run() {   # run <notebook> <output tag> <max minutes> VAR=value ...
  name=$1; out=$2; limit=$3; shift 3
  echo "$(date +%H:%M:%S) start $out" >> "$STATUS"
  timeout --signal=TERM --kill-after=60 "${limit}m" "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env "$@" "$V" nbconvert --to notebook --execute "$name.ipynb" --output "${name}_executed_${out}.ipynb" \
     --ExecutePreprocessor.timeout=-1 > "$LOGS/second_list_${out}.txt" 2>&1
  echo "$(date +%H:%M:%S) end $out exit $?" >> "$STATUS"
}
echo "chain started $(date)" >> "$STATUS"
run Idea1b_Coarse_Aligned_Count official 90 GATE_DATA=official TAG=official150 OUT_TAG=official CONTROL_N=16
run Idea3_Pose_Sweep_And_Part_Correlation nudt_valid 150 GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212 SEED_IDS_FILE="$NEVER_NUDT" CONTROL_N=10
run Idea3_Pose_Sweep_And_Part_Correlation official 150 GATE_DATA=official TAG=official150 SEED_IDS_FILE="$LOW_OFFICIAL" CONTROL_N=8
run Idea7b_Large_Window_LoFTR nudt_valid 90 GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212 SEED_IDS_FILE="$NEVER_NUDT" CONTROL_N=10
run Idea7b_Large_Window_LoFTR official 90 GATE_DATA=official TAG=official150 SEED_IDS_FILE="$LOW_OFFICIAL" CONTROL_N=8
run Idea4_Part_Peak_Agreement official 150 GATE_DATA=official TAG=official150
run Idea4_Part_Peak_Agreement nudt_valid 150 GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212
run Idea5b_Altered_Copies official 120 GATE_DATA=official
run Idea6b_Refine_Scale_Heading official 120 GATE_DATA=official
echo "chain finished $(date)" >> "$STATUS"
echo "end exit 0"
