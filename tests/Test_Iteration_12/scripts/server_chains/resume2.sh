#!/usr/bin/env bash
# restart the official dumps when the high-altitude follow-ups are done (the GPU takes two jobs, but two of these dumps slow each other to a crawl when memory is full)
W=/mnt/e/IdeaForge/work
until grep -q "HIGHALT CHAIN2 DONE" $W/status_highalt.txt 2>/dev/null; do sleep 60; done
echo "$(date +%m-%d_%H:%M:%S) resume official dumps (after highalt chain2)" >> $W/status_overnight.txt
exec $W/overnight_wrapper.sh
