#!/usr/bin/env bash
# Chain 7 (2026-11-16): 150 NUDT valid-split photos that the valid1212 draw did not use (ids_nudt_valid1313.csv), low-threshold first-pass dump, for the idea 1 rule fixed in advance.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin/jupyter"
LOGS="$HERE/results/logs"; STATUS="$LOGS/followup4_status.txt"; R="$HERE/results"
until grep -q "chain finished" "$LOGS/headless3_status.txt" 2>/dev/null; do sleep 60; done
cd "$HERE/notebooks" || exit 1
echo "$(date +%H:%M:%S) start" >> "$STATUS"
timeout --signal=TERM --kill-after=60 120m "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 PHOTOS_FILE="$R/ids_nudt_valid1313.csv" \
  TAG=lowconf_nudt_valid1313 "$V" nbconvert --to notebook --execute Rich_Dump_Run_LowConf.ipynb --output Rich_Dump_Run_LowConf_executed_valid1313.ipynb \
  --ExecutePreprocessor.timeout=-1 > "$LOGS/followup4_dump.txt" 2>&1
code=$?; echo "$(date +%H:%M:%S) end exit $code (check the log for a Traceback)" >> "$STATUS"; echo "chain finished" >> "$STATUS"
