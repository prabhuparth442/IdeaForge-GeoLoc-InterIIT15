#!/usr/bin/env bash
# Overnight queue of photo-view runs (2026-11-18), one after the other. Waits for the earlier photoview runs (status_pv.txt) to finish their fit20 run first.
W=/mnt/e/IdeaForge/work; cd $W
until grep -q "end fit20" status_pv.txt; do sleep 30; done
bash pv_sets_chain.sh dev g600 600 20
bash pv_sets_chain.sh dev g800 800 20
bash pv_sets_chain.sh dev g0 0 20
bash pv_sets_chain.sh official g0b20 0 20
bash pv_sets_chain.sh nudt g0b20 0 20
bash pv_sets_chain.sh official g0b12 0 12
bash pv_sets_chain.sh nudt g0b12 0 12
echo "$(date +%m-%d_%H:%M:%S) queue finished" >> $W/status_pvs.txt
