#!/usr/bin/env bash
set -euo pipefail

run_root=${1:-/root/winner-calibration-ten-ticket-20260922}

export NANOBODY_SCORER_DAEMON=1
export NANOBODY_PACKAGE_ROOT=/root
export NANOBODY_SCORER_TIMEOUT=7200

for index in $(seq 17 32); do
  name=$(printf 'fresh_%02d' "$index")
  /root/bgen_python.sh -m nanobody_opt.score_boltzgen_waves \
    --sequences "$run_root/sequences.json" \
    --config /root/config/config.yaml \
    --target P20809 \
    --output-dir "$run_root/$name" \
    --gpus 0,1,2,3,4,5,6,7 \
    --shards-per-gpu 1 \
    --wave-size 1000 \
    --predict-batch-size 1 \
    --scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py
done

touch "$run_root/COMPLETE"
