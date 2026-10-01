#!/bin/bash
# Folding-depth sweep: does --fold-sampling-steps 100/200 lift the interface-PAE
# fidelity that --fold-sampling-steps 50 under-converges? grafting held fixed.
set -euo pipefail
EXP=/root/exp
SCORER=/root/fastscore/release/fast_score_batch.py
PY=/root/fastscore/release/release_python.sh
CONFIG=/root/config/config.yaml
SYNTH=/root/fastscore/release/refdesign/0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif
INPUT=/root/exp/input.json
GPUS="0,1,2,3,4,5,6,7"
REPS="${REPS:-10}"

mkdir -p "$EXP/fold100" "$EXP/fold200"

for rep in $(seq 1 "$REPS"); do
  echo "=== $(date -u +%H:%M:%S) fold100 rep $rep ==="
  "$PY" "$SCORER" --sequences "$INPUT" --config "$CONFIG" --target P20809 \
    --gpus "$GPUS" --shards-per-gpu 1 --out "$EXP/fold100/rep_$rep.json" \
    --synth-reference "$SYNTH" --fold-sampling-steps 100
  echo "=== $(date -u +%H:%M:%S) fold200 rep $rep ==="
  "$PY" "$SCORER" --sequences "$INPUT" --config "$CONFIG" --target P20809 \
    --gpus "$GPUS" --shards-per-gpu 1 --out "$EXP/fold200/rep_$rep.json" \
    --synth-reference "$SYNTH" --fold-sampling-steps 200
done
echo "=== DONE ==="
ls -la "$EXP/fold100" "$EXP/fold200" | tail -6
