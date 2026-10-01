#!/usr/bin/env bash
set -euo pipefail

run_root=${1:-/root/gpu-capacity-validation-20260922}

export NANOBODY_SCORER_DAEMON=1
export NANOBODY_PACKAGE_ROOT=/root
export NANOBODY_SCORER_TIMEOUT=7200

score_pass() {
  local input=$1
  local output=$2
  local gpus=$3
  local wave_size=$4
  /root/bgen_python.sh -m nanobody_opt.score_boltzgen_waves \
    --sequences "$input" \
    --config /root/config/config.yaml \
    --target P20809 \
    --output-dir "$output" \
    --gpus "$gpus" \
    --shards-per-gpu 1 \
    --wave-size "$wave_size" \
    --predict-batch-size 1 \
    --scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py
}

for index in $(seq 1 4); do
  name=$(printf 'discovery_%02d' "$index")
  score_pass "$run_root/sequences.json" "$run_root/$name" 0,1,2,3,4,5,6,7 1000
done

/root/bgen_python.sh "$run_root/select_shortlist.py"

# Confirmation passes are independent. Two four-GPU passes in parallel avoid
# paying twenty serial model-load stalls while preserving one complete metric
# vector and observation id per pass.
for first in $(seq 1 2 19); do
  second=$((first + 1))
  first_name=$(printf 'confirmation_%02d' "$first")
  second_name=$(printf 'confirmation_%02d' "$second")
  score_pass "$run_root/shortlist_sequences.json" "$run_root/$first_name" 0,1,2,3 100 &
  left=$!
  score_pass "$run_root/shortlist_sequences.json" "$run_root/$second_name" 4,5,6,7 100 &
  right=$!
  wait "$left"
  wait "$right"
done

touch "$run_root/COMPLETE"
