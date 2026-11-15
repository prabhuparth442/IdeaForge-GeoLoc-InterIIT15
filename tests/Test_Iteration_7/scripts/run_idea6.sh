#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin"
cd "$HERE/notebooks" || exit 1
run() { "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env "$@" "$V/jupyter" nbconvert --to notebook --execute Idea6_Map_Twins.ipynb --output "Idea6_Map_Twins_executed_${OUTNAME}.ipynb" --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_idea6_${OUTNAME}.txt" 2>&1; echo "end $OUTNAME exit $?"; }
OUTNAME=official run GATE_DATA=official POOL_PHOTOS=500 DUMPS=official150 OUT=official
OUTNAME=nudt run GATE_DATA=nudt POOL_PHOTOS=60 DUMPS=nudt_valid1212,nudt_fresh1010,nudt_fresh707 OUT=nudt
echo "end exit 0"
