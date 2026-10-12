#!/usr/bin/env bash
# One replay-queue batch on the slave, run FROM MASTER (master is the only DB
# writer): build batch -> ship -> score on slave GPUs (flock-serialised with the
# rival replay) -> pull NDJSON -> ingest. Extra args go to replay_queue.py
# (e.g. --test-limit 4 --test-passes 2 for a verification run).
# Exit: 0 batch done, 4 queue empty, other = failure.
set -euo pipefail
SLAVE_HOST="root@ssh2.vast.ai"; SLAVE_PORT="28358"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15)
BUDGET="${REPLAY_QUEUE_BUDGET_SEQ_PASSES:-350}"
PY=/root/.pixi/envs/bgen/bin/python
WORK=/root/artifacts/replay_queue
mkdir -p "$WORK/incoming"
LOCAL_BATCH="$WORK/batch_current.json"

set +e
$PY /root/tools/replay_queue.py --out "$LOCAL_BATCH" --budget-seq-passes "$BUDGET" "$@"
rc=$?
set -e
if [ "$rc" -eq 4 ]; then echo "RUN_REPLAY_QUEUE_EMPTY"; exit 4; fi
[ "$rc" -eq 0 ] || { echo "RUN_REPLAY_QUEUE_FAILED: build rc=$rc" >&2; exit "$rc"; }

BATCH_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['batch_id'])" "$LOCAL_BATCH")
REMOTE_BATCH="/root/replay_queue_in/${BATCH_ID}.json"
REMOTE_OUT="/root/replay_queue_out/${BATCH_ID}.ndjson"
LOCAL_OUT="$WORK/incoming/${BATCH_ID}.ndjson"

echo "[1/4] shipping batch $BATCH_ID"
ssh "${SSH_OPTS[@]}" -p "$SLAVE_PORT" "$SLAVE_HOST" "mkdir -p /root/replay_queue_in /root/replay_queue_out"
scp -q -P "$SLAVE_PORT" -o StrictHostKeyChecking=accept-new "$LOCAL_BATCH" "${SLAVE_HOST}:${REMOTE_BATCH}"

echo "[2/4] scoring on slave (GPU lock shared with the rival replay)"
START=$(date +%s)
flock -w 14400 /root/artifacts/slave_gpu.lock timeout 3h \
  ssh "${SSH_OPTS[@]}" -p "$SLAVE_PORT" "$SLAVE_HOST" \
  "/root/release_python_slave.sh /root/replay_queue_writeback.py --batch '$REMOTE_BATCH' --gpus 0,1,2,3 --out '$REMOTE_OUT'"
echo "REPLAY_QUEUE_SCORING_SECONDS=$(( $(date +%s) - START ))"

echo "[3/4] pulling NDJSON"
scp -q -P "$SLAVE_PORT" -o StrictHostKeyChecking=accept-new "${SLAVE_HOST}:${REMOTE_OUT}" "$LOCAL_OUT"

echo "[4/4] ingesting into master's local sqlite"
cd /root && $PY /root/tools/ingest_slave_multipass.py --ndjson "$LOCAL_OUT"
ssh "${SSH_OPTS[@]}" -p "$SLAVE_PORT" "$SLAVE_HOST" "rm -f '$REMOTE_BATCH'" || true
find "$WORK/incoming" -name '*.ndjson' -mtime +3 -delete 2>/dev/null || true
echo "RUN_REPLAY_QUEUE_DONE batch=$BATCH_ID"
