#!/usr/bin/env bash
# Stuck-run watchdog for the overnight chain. Every CHECK seconds it looks at the notebook kernel (ipykernel_launcher) and its CPU time.
# If the kernel's CPU time has not moved for STUCK_AFTER seconds and the GPU is idle, the run is stuck: the notebook job is killed so the chain goes on
# with the next one (the job's timeout in run_second_list_overnight.sh is the second safety net). Everything is written to results/logs/watchdog.txt.
# Also stops everything if the free RAM falls under MIN_FREE_MB (the PC must not die).
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/watchdog.txt"
CHECK=300; STUCK_AFTER=1500; MIN_FREE_MB=700
last_ticks=""; last_change=$(date +%s)
echo "watchdog started $(date)" >> "$LOG"
while true; do
  sleep $CHECK
  pid=$(pgrep -f ipykernel_launcher | tail -1)
  free_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ -z "$pid" ]; then echo "$(date +%H:%M) no kernel running (free ${free_mb} MB)" >> "$LOG"; last_ticks=""; last_change=$(date +%s); continue; fi
  ticks=$(awk '{print $14+$15}' /proc/$pid/stat 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  now=$(date +%s)
  if [ "$ticks" != "$last_ticks" ]; then last_ticks=$ticks; last_change=$now; fi
  idle=$((now - last_change))
  echo "$(date +%H:%M) kernel $pid cpu_ticks=$ticks idle=${idle}s gpu=${gpu}% free=${free_mb}MB" >> "$LOG"
  if [ "$idle" -ge "$STUCK_AFTER" ] && [ "${gpu:-0}" -lt 5 ]; then
    echo "$(date +%H:%M) STUCK: killing the notebook job (kernel $pid)" >> "$LOG"
    pkill -f "nbconvert --to notebook --execute"; kill "$pid" 2>/dev/null; last_ticks=""; last_change=$(date +%s)
  fi
  if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
    echo "$(date +%H:%M) LOW MEMORY (${free_mb} MB): killing the notebook job" >> "$LOG"
    pkill -f "nbconvert --to notebook --execute"; kill "$pid" 2>/dev/null
  fi
done
