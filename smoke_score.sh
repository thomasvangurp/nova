#!/bin/bash
# Scorer smoke test on the throughput box: 8 sequences -> 8 GPU shards.
# Validates the fast scorer adapter + config + checkpoints and checks no NaN.
set -euo pipefail
cd /root
export NANOBODY_FAST_SCORER_CONFIG=/root/artifacts/nanobody_fast_scorer.json
rm -rf /tmp/smoke_work /tmp/smoke_out.json
mkdir -p /tmp/smoke_work

# Snapshot GPU utilization in the background while the scorer runs.
( for i in $(seq 1 40); do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > /tmp/gpu_snap.$i 2>/dev/null
    sleep 3
  done ) &
SNAP_PID=$!

time bash /root/bgen_python.sh /root/nanobody_opt/score_boltzgen_fast_adapter.py \
  --sequences /tmp/smoke_seqs.json \
  --config /root/config/config.yaml \
  --target P20809 \
  --gpus 0,1,2,3,4,5,6,7 \
  --shards-per-gpu 1 \
  --batch-size 1 \
  --work-root /tmp/smoke_work \
  --out /tmp/smoke_out.json 2>&1 | tail -45
RC=${PIPESTATUS[0]}
kill $SNAP_PID 2>/dev/null || true

echo "=== gpu snapshots (last non-idle) ==="
grep -hE "[1-9][0-9]* (MiB|%)" /tmp/gpu_snap.* 2>/dev/null | tail -8 || true

echo "=== out ==="
if [ -f /tmp/smoke_out.json ]; then
  python3 - <<'EOF'
import json, math
d = json.load(open("/tmp/smoke_out.json"))
s = json.dumps(d)
print("out_type", type(d).__name__)
print("raw_has_nan_token", ("NaN" in s) or ("nan" in s.lower()))
# walk for float NaN
bad = 0; nums = 0
def walk(x):
    global bad, nums
    if isinstance(x, float):
        nums += 1
        if math.isnan(x): bad += 1
    elif isinstance(x, dict):
        for v in x.values(): walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)
walk(d)
print("float_count", nums, "nan_count", bad)
print("SCORER_SMOKE_OK" if (nums > 0 and bad == 0) else "SCORER_SMOKE_CHECK_NEEDED")
EOF
else
  echo "NO_OUT_FILE"
fi
exit $RC
