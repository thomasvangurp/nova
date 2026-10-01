#!/bin/bash
# SN68 scorer-calibration experiment: fast (grafting) vs full (diffusion), N reps each.
# BoltzGen is stochastic (test-retest ~0.596), so each sequence is scored REP times
# and the analysis uses means. Runs on the research box, never prod.
set -euo pipefail

EXP=/root/exp
SCORER=/root/fastscore/release/fast_score_batch.py
PY=/root/fastscore/release/release_python.sh
CONFIG=/root/config/config.yaml
SYNTH=/root/fastscore/release/refdesign/0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif
INPUT=/root/exp/input.json
REPS="${REPS:-20}"
START_REP="${START_REP:-1}"

mkdir -p "$EXP/fast" "$EXP/full"

GPUS="0,1,2,3,4,5,6,7"

for rep in $(seq "$START_REP" "$REPS"); do
  echo "=== $(date -u +%H:%M:%S) fast rep $rep ==="
  "$PY" "$SCORER" --sequences "$INPUT" --config "$CONFIG" --target P20809 \
    --gpus "$GPUS" --shards-per-gpu 1 --out "$EXP/fast/rep_$rep.json" \
    --synth-reference "$SYNTH" --fold-sampling-steps 50
  echo "=== $(date -u +%H:%M:%S) full rep $rep ==="
  "$PY" "$SCORER" --sequences "$INPUT" --config "$CONFIG" --target P20809 \
    --gpus "$GPUS" --shards-per-gpu 1 --out "$EXP/full/rep_$rep.json" \
    --design-sampling-steps 100 --fold-sampling-steps 50
done
echo "=== DONE ==="
ls -la "$EXP/fast" "$EXP/full" | tail -8
