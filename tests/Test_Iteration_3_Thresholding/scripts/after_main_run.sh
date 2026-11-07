#!/usr/bin/env bash
# waits for the full NUDT run to end, then runs the NUDT oracle-tile notebook (full, QUICK_TRY=False) and exports an HTML copy
set -u
R=/home/palash/Ideaforge/GeoLoc/IdeaForge-GeoLoc-InterIIT15
T=$R/tests/Test_Iteration_3_Thresholding
LOG=$T/results/nudt_after_run.log
source /home/palash/Ideaforge/GeoLoc/.venv-geoloc/bin/activate
mkdir -p $T/results/nudt
while kill -0 30625 2>/dev/null; do sleep 30; done
echo "main run ended $(date)" >> $LOG
CSV=$T/results/nudt_edge_zncc_all_11890.csv
if [ "$(wc -l < $CSV)" -lt 11891 ]; then
  echo "CSV incomplete ($(wc -l < $CSV) lines), resuming once" >> $LOG
  python -I $T/scripts/run_nudt_full.py all 0 edge_zncc >> $LOG 2>&1
fi
echo "CSV lines: $(wc -l < $CSV)" >> $LOG
cd $T/notebooks
sed 's/QUICK_TRY = True /QUICK_TRY = False/' Step1_Oracle_Tile_NUDT.ipynb > _run_full.ipynb
grep -c 'QUICK_TRY = False' _run_full.ipynb >> $LOG
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True _run_full.ipynb --output $T/results/nudt/Step1_NUDT_executed.ipynb >> $LOG 2>&1
jupyter nbconvert --to html $T/results/nudt/Step1_NUDT_executed.ipynb >> $LOG 2>&1
rm -f _run_full.ipynb
echo "ALL DONE $(date)" >> $LOG
