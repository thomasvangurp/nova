#!/usr/bin/env bash
# Verify the passing release installed on this host before any activation.
set -euo pipefail
TASK_ROOT="${NANOBODY_ROOT:-/root}"
TASK_PYTHON="${BGEN_PYTHON:-$TASK_ROOT/.pixi/envs/bgen/bin/python}"
: "${SN68_TESTED_SHA:?Set the exact SHA from the passing release artifact}"
"$TASK_PYTHON" "$TASK_ROOT/scripts/release_manifest.py" verify \
  --root "$TASK_ROOT" --manifest "$TASK_ROOT/build_stamp.json" \
  --expected-sha "$SN68_TESTED_SHA" --runtime-only
"$TASK_PYTHON" "$TASK_ROOT/scripts/check_launch_contract.py" --root "$TASK_ROOT"
"$TASK_PYTHON" "$TASK_ROOT/scripts/promotion_gate.py" --restart \
  --runtime-root "$TASK_ROOT" --candidate-sha "$SN68_TESTED_SHA"
