#!/usr/bin/env bash
# restarts the supervisor if it dies (crash, OOM, killed) until OVERNIGHT_DONE exists, at most 20 times
W=/mnt/e/IdeaForge/work
for i in $(seq 1 20); do
  [ -f $W/OVERNIGHT_DONE ] && exit 0
  bash $W/overnight_supervisor.sh
  echo "$(date +%m-%d_%H:%M:%S) supervisor exited, restart $i" >> $W/status_overnight.txt; sleep 60
done
