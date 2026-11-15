#!/usr/bin/env bash
# waits for the valid-1212 dump, then runs the oracle test (GPU) and the pre-registered guard confirmation (CPU), one at a time under the memory cap.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin"
cd "$HERE/notebooks" || exit 1
until grep -q "end nudt_valid1212" "$HERE/results/logs/run_valid1212.out"; do sleep 30; done
echo "$(date) oracle start"
"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=official "$V/jupyter" nbconvert --to notebook --execute Idea0_Oracle_Low_Altitude.ipynb \
  --output Idea0_Oracle_Low_Altitude_executed.ipynb --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_idea0_oracle.txt" 2>&1
echo "$(date) oracle end exit $?"
"$REPO/tools/run_capped.sh" -m 4G -c 0-7 -t 4 -- "$V/jupyter" nbconvert --to notebook --execute Ours_Inflation_Guard_Valid1212.ipynb \
  --output Ours_Inflation_Guard_Valid1212_executed.ipynb --ExecutePreprocessor.timeout=3000 > "$HERE/results/logs/log_guard_valid1212.txt" 2>&1
echo "$(date) guard end exit $?"
