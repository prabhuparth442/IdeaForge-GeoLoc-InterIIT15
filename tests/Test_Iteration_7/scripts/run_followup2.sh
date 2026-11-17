#!/usr/bin/env bash
# Follow-up chain 2 (2026-11-16): idea 5 with timings on official and on a NUDT valid draw. Waits for chain 1 to finish (one heavy job at a time).
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin/jupyter"
LOGS="$HERE/results/logs"; STATUS="$LOGS/followup2_status.txt"; R="$HERE/results"

cd "$HERE/notebooks" || exit 1
run() { name=$1; out=$2; limit=$3; shift 3
  echo "$(date +%H:%M:%S) start $name $out" >> "$STATUS"
  timeout --signal=TERM --kill-after=60 "${limit}m" "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env "$@" "$V" nbconvert --to notebook --execute "$name.ipynb" --output "${name}_executed_${out}.ipynb" \
     --ExecutePreprocessor.timeout=-1 > "$LOGS/followup2_${name}_${out}.txt" 2>&1
  code=$?; echo "$(date +%H:%M:%S) end $name $out exit $code (see the log for a Traceback)" >> "$STATUS"; }
echo "chain started $(date)" >> "$STATUS"
run Idea5b_Altered_Copies nudt_valid 150 GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212 PHOTOS_FILE="$R/ids_nudt_valid60_idea5.csv" OUT_TAG=_nudt_valid
run Idea5b_Altered_Copies official_v2 120 GATE_DATA=official OUT_TAG=_official_v2
echo "chain finished $(date)" >> "$STATUS"
