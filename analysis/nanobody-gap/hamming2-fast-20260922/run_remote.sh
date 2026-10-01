#!/usr/bin/env bash
set -euo pipefail

run_root=${1:-/root/hamming2-fast-20260922}
gpus=${2:-0,1,2,3,4,5,6,7}

export NANOBODY_SCORER_DAEMON=1
export NANOBODY_PACKAGE_ROOT=/root
export NANOBODY_SCORER_TIMEOUT=7200

run_passes() {
  local input=$1
  local prefix=$2
  local first=$3
  local last=$4
  local wave_size=$5
  for pass_index in $(seq "$first" "$last"); do
    pass_name=$(printf '%s_%02d' "$prefix" "$pass_index")
    /root/bgen_python.sh -m nanobody_opt.score_boltzgen_waves \
      --sequences "$input" \
      --config /root/config/config.yaml \
      --target P20809 \
      --output-dir "$run_root/$pass_name" \
      --gpus "$gpus" \
      --shards-per-gpu 1 \
      --wave-size "$wave_size" \
      --predict-batch-size 1 \
      --scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py
  done
}

run_passes "$run_root/sequences.json" pass 1 12 700
run_passes "$run_root/easynano_sequences.json" easynano_extra 13 20 3
touch "$run_root/COMPLETE"
