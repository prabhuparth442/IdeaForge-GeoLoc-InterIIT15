#!/usr/bin/env bash
# Chain 4 (2026-11-16): a fresh NUDT draw (seed 1313, photos no earlier set used) as a low-threshold first-pass dump, for the idea 1 rule fixed in advance. Waits for chain 3.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin/jupyter"
LOGS="$HERE/results/logs"; STATUS="$LOGS/followup3_status.txt"; R="$HERE/results"
until grep -q "chain finished" "$LOGS/headless_status.txt" 2>/dev/null; do sleep 60; done
cd "$HERE/notebooks" || exit 1
echo "$(date +%H:%M:%S) start" >> "$STATUS"
timeout --signal=TERM --kill-after=60 120m "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=nudt POOL_PHOTOS=2400 FRESH_EXCLUDE_POOL=500 EXCLUDE_IDS_FILE="$R/exclude_for_fresh1313.csv" \
  NUDT_SEED=1313 N_NUDT=150 TAG=lowconf_nudt_fresh1313 "$V" nbconvert --to notebook --execute Rich_Dump_Run_LowConf.ipynb --output Rich_Dump_Run_LowConf_executed_fresh1313.ipynb \
  --ExecutePreprocessor.timeout=-1 > "$LOGS/followup3_dump.txt" 2>&1
code=$?; echo "$(date +%H:%M:%S) end exit $code (check the log for a Traceback)" >> "$STATUS"; echo "chain finished" >> "$STATUS"
