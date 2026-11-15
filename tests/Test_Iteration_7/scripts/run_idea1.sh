#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin"
cd "$HERE/notebooks" || exit 1
until grep -q "^end exit" "$HERE/results/logs/run_idea5.out"; do sleep 20; done
"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=official PHOTOS_FILE="$REPO/tests/Test_Iteration_9/results/photos_official.csv" BASE_TAG=official60_scale15 KINDS=clean,height20 \
  "$V/jupyter" nbconvert --to notebook --execute Idea1_Scale_Coverage_Dump.ipynb --output Idea1_Scale_Coverage_Dump_executed.ipynb --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_idea1_dump.txt" 2>&1
echo "end exit $?"
