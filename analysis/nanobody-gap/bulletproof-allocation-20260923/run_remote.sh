#!/usr/bin/env bash
set -euo pipefail

run_root=${1:-/root/bulletproof-allocation-20260923}
export NANOBODY_SCORER_DAEMON=1
export NANOBODY_PACKAGE_ROOT=/root
export NANOBODY_SCORER_TIMEOUT=7200

for index in $(seq 1 12); do
  name=$(printf 'development_%02d' "$index")
  /root/bgen_python.sh -m nanobody_opt.score_boltzgen_waves \
    --sequences "$run_root/development_sequences.json" \
    --config /root/config/config.yaml \
    --target P20809 \
    --output-dir "$run_root/$name" \
    --gpus 0,1,2,3,4,5,6,7 \
    --shards-per-gpu 1 \
    --wave-size 1200 \
    --predict-batch-size 1 \
    --scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py
done

/root/bgen_python.sh "$run_root/select_fleets.py" --root "$run_root"
test ! -e "$run_root/fresh_01/exact_all.json"
sha256sum "$run_root/selected_fleets.json" "$run_root/fresh_sequences.json" \
  > "$run_root/FLEET_FREEZE.sha256"

for index in $(seq 1 32); do
  name=$(printf 'fresh_%02d' "$index")
  /root/bgen_python.sh -m nanobody_opt.score_boltzgen_waves \
    --sequences "$run_root/fresh_sequences.json" \
    --config /root/config/config.yaml \
    --target P20809 \
    --output-dir "$run_root/$name" \
    --gpus 0,1,2,3,4,5,6,7 \
    --shards-per-gpu 1 \
    --wave-size 1200 \
    --predict-batch-size 1 \
    --scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py
done

touch "$run_root/COMPLETE"
