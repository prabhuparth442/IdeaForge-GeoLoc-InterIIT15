#!/usr/bin/env bash
# Run one heavy job (notebook execution, GPU dump, sweep) inside a memory and CPU cap so it cannot take the whole PC down.
# Why: on 2026-11-14 the 14 GB laptop ran out of RAM twice (12 parallel sweep processes next to a GPU dump) and the kernel killed
# the Claude app and every background job (tests/HANDOFF.md sections 5 and 9). An over-limit job is now killed inside its own cap.
# Usage:  tools/run_capped.sh [-m 6G] [-c 0-7] [-t 6] -- <command> [args...]
#   -m  hard memory cap of the job (default 6G; this PC has 15 GB and the desktop apps use about 8)
#   -c  CPUs the job may use (default 0-7 of 12, four are left free)
#   -t  threads for numpy / torch / OpenMP (default 6; more threads than allowed cores made jobs ~100x slower)
# Run one such job at a time; never run a CPU sweep next to a GPU dump. Sweeps: n_jobs <= 4 (about 1.4 GB per process).
MEM=6G; CPUS=0-7; THREADS=6
while getopts "m:c:t:" opt; do case $opt in m) MEM=$OPTARG;; c) CPUS=$OPTARG;; t) THREADS=$OPTARG;; *) exit 2;; esac; done
shift $((OPTIND - 1)); [ "$1" = "--" ] && shift
[ $# -eq 0 ] && { echo "no command given"; exit 2; }
exec systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 -p MemoryHigh="$MEM" \
  -E OMP_NUM_THREADS="$THREADS" -E MKL_NUM_THREADS="$THREADS" -E OPENBLAS_NUM_THREADS="$THREADS" -E NUMEXPR_NUM_THREADS="$THREADS" \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nice -n 10 ionice -c 2 -n 6 taskset -c "$CPUS" "$@"
