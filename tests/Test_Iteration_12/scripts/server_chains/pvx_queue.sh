#!/usr/bin/env bash
# PV-X: the locked test split once (seed 0, report only), then a second hint-jitter seed on the three development sets as a robustness check.
cd /mnt/e/IdeaForge/work
bash pvx_chain.sh test t0 0 &
bash pvx_chain.sh official s1 1
wait
bash pvx_chain.sh dev s1 1 &
bash pvx_chain.sh nudt s1 1
wait
echo "$(date +%m-%d_%H:%M:%S) pvx queue finished" >> status_pvx.txt
