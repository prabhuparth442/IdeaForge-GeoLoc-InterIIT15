#!/usr/bin/env bash
# Follow-up chain 1 (2026-11-16): idea 7 redone at native resolution, then low-threshold standard-query dumps of four photo sets for idea 1.
# One job at a time inside tools/run_capped.sh (6G, 8 cores), each with a hard time limit. Status: results/logs/followup1_status.txt.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin/jupyter"
LOGS="$HERE/results/logs"; STATUS="$LOGS/followup1_status.txt"
cd "$HERE/notebooks" || exit 1
R="$HERE/results"
run() {   # run <notebook> <output tag> <max minutes> VAR=value ...
  name=$1; out=$2; limit=$3; shift 3
  echo "$(date +%H:%M:%S) start $name $out" >> "$STATUS"
  timeout --signal=TERM --kill-after=60 "${limit}m" "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env "$@" "$V" nbconvert --to notebook --execute "$name.ipynb" --output "${name}_executed_${out}.ipynb" \
     --ExecutePreprocessor.timeout=-1 > "$LOGS/followup1_${name}_${out}.txt" 2>&1
  echo "$(date +%H:%M:%S) end $name $out exit $?" >> "$STATUS"
}
echo "chain started $(date)" >> "$STATUS"
run Idea7c_Native_Window_LoFTR official 60 GATE_DATA=official TAG=official150 SEED_IDS_FILE="$R/low_altitude_skips_official150.csv" CONTROL_N=8
run Idea7c_Native_Window_LoFTR nudt_valid 60 GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212 SEED_IDS_FILE="$R/never_retrieved_valid1212.csv" CONTROL_N=10
run Rich_Dump_Run_LowConf lowconf_official150 90 GATE_DATA=official PHOTOS_FILE="$R/ids_official150.csv" TAG=lowconf_official150 REFERENCE=official150
run Rich_Dump_Run_LowConf lowconf_nudt_valid1212 90 GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 PHOTOS_FILE="$R/ids_nudt_valid1212.csv" TAG=lowconf_nudt_valid1212 REFERENCE=nudt_valid1212
run Rich_Dump_Run_LowConf lowconf_nudt_fresh707 90 GATE_DATA=nudt POOL_PHOTOS=2400 PHOTOS_FILE="$R/ids_nudt_fresh707.csv" TAG=lowconf_nudt_fresh707 REFERENCE=nudt_fresh707
run Rich_Dump_Run_LowConf lowconf_nudt_fresh1010 90 GATE_DATA=nudt POOL_PHOTOS=2400 PHOTOS_FILE="$R/ids_nudt_fresh1010.csv" TAG=lowconf_nudt_fresh1010 REFERENCE=nudt_fresh1010
echo "chain finished $(date)" >> "$STATUS"
