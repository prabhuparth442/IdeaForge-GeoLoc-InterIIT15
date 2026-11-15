#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin"
cd "$HERE/notebooks" || exit 1

"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=official POOL_PHOTOS=500 PHOTOS_FILE="$REPO/tests/Test_Iteration_9/results/photos_official.csv" \
  "$V/jupyter" nbconvert --to notebook --execute Idea8_Estimators.ipynb --output Idea8_Estimators_executed.ipynb --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_idea8a.txt" 2>&1
echo "end exit $?"
