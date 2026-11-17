#!/usr/bin/env bash
# Prints one status line every 15 minutes (the signal that wakes the session). Usage: tick.sh [first_delay_seconds]
sleep "${1:-900}"
REMOTE='W=/mnt/e/IdeaForge/work; echo "stageeval: $(tail -1 $W/status_stageeval.txt 2>/dev/null) | gpu $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) | reports $(ls $W/reports_stageeval 2>/dev/null | grep -c ipynb) | $(df -h /mnt/e | tail -1)"'
while true; do
  line=$(ssh -o BatchMode=yes -o ConnectTimeout=15 workstation@100.105.160.107 "$REMOTE" 2>/dev/null | grep -v setlocale)
  echo "TICK $(date +%H:%M) ${line:-SSH FAILED}"
  sleep 900
done
