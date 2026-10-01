#!/usr/bin/env bash
set -euo pipefail

run_root=${1:-/root/mc_selector_20260921}
passes=${2:-12}
gpus=${3:-0,1,2,3,4,5,6,7}

export NANOBODY_SCORER_DAEMON=1
export NANOBODY_PACKAGE_ROOT=/root

for pass_index in $(seq 1 "$passes"); do
  pass_name=$(printf 'pass_%02d' "$pass_index")
  /root/bgen_python.sh -m nanobody_opt.score_boltzgen_waves \
    --sequences "$run_root/sequences.json" \
    --config /root/config/config.yaml \
    --target P20809 \
    --output-dir "$run_root/$pass_name" \
    --gpus "$gpus" \
    --shards-per-gpu 1 \
    --wave-size 96 \
    --predict-batch-size 1 \
    --scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py
done

touch "$run_root/COMPLETE"
