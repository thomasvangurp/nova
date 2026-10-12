#!/usr/bin/env bash
# Orchestrate one epoch's slave multipass scoring + writeback, run FROM
# MASTER only. Slave never initiates a connection to master and never opens
# nanobody_mutations.sqlite; master pushes work to slave and pulls results
# back, then inserts them itself. See:
#   /root/slave_multipass_writeback.py   (runs on slave; scores only)
#   /root/tools/ingest_slave_multipass.py (runs on master; writes DB)
#   nanobody_opt/observation_labels.py   (canonical multipass_sum10_pass label, writer slave_multipass_writeback)
#
# Steps:
#   1. Collect every P20809 submission on the epoch's public leaderboard
#      (all miners) plus our qualified shortlist, minus sequences that already
#      have >= PASSES replay passes (tools/collect_slave_multipass_seqs.py).
#   2. scp that list to slave.
#   3. ssh into slave to run the scorer (idle GPUs 0-3, bgen env).
#   4. scp the NDJSON result back to master.
#   5. Ingest it into the live local sqlite DB (master-side insert only).
#
# Usage:
#   /root/tools/run_slave_multipass_epoch.sh <EPOCH> [PASSES] [--dry-run]
#
# Safe to re-run for the same epoch: step 1 is a deterministic re-read,
# scoring is idempotent per (sequence, pass), and the DB insert upserts on
# observation_id. Does not touch the live producer/race/publisher/scorer
# daemon in any way.
set -euo pipefail

EPOCH="${1:?usage: run_slave_multipass_epoch.sh EPOCH [PASSES] [--dry-run]}"
PASSES="${2:-10}"
DRY_RUN=""
for arg in "$@"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN="--dry-run"
done

SLAVE_HOST="root@ssh2.vast.ai"
SLAVE_PORT="28358"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15)

EPOCH_DIR="/root/live/epoch${EPOCH}"
QUALIFIED_DIR="${EPOCH_DIR}/checkpoints/qualified"
LOCAL_SEQ_JSON="/tmp/slave_multipass_seqs_epoch${EPOCH}.json"
LOCAL_NDJSON="/root/artifacts/slave_writeback_incoming/epoch${EPOCH}.ndjson"
REMOTE_SEQ_JSON="/root/slave_multipass_in/epoch${EPOCH}_todo.json"
REMOTE_NDJSON="/root/slave_multipass_out/epoch${EPOCH}.ndjson"

mkdir -p "$(dirname "$LOCAL_NDJSON")"

echo "[1/5] collecting every leaderboard submission + our qualified shortlist still short of $PASSES passes"
set +e
python3 /root/tools/collect_slave_multipass_seqs.py "$EPOCH" "$LOCAL_SEQ_JSON" "$PASSES"
rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  echo "RUN_SLAVE_MULTIPASS_PENDING epoch=$EPOCH (leaderboard not published)"
  exit 3
elif [ "$rc" -ne 0 ]; then
  echo "RUN_SLAVE_MULTIPASS_FAILED: collection rc=$rc" >&2
  exit "$rc"
fi
if [ "$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$LOCAL_SEQ_JSON")" = "0" ]; then
  echo "RUN_SLAVE_MULTIPASS_DONE epoch=$EPOCH (nothing left to score)"
  exit 0
fi

echo "[2/5] shipping sequence list to slave"
ssh "${SSH_OPTS[@]}" -p "$SLAVE_PORT" "$SLAVE_HOST" "mkdir -p $(dirname "$REMOTE_SEQ_JSON") $(dirname "$REMOTE_NDJSON")"
scp -q -P "$SLAVE_PORT" -o StrictHostKeyChecking=accept-new "$LOCAL_SEQ_JSON" "${SLAVE_HOST}:${REMOTE_SEQ_JSON}"

echo "[3/5] running multipass scorer on slave (passes=$PASSES, GPUs 0-3)"
# shared with tools/run_replay_queue_batch.sh: one job on the slave GPUs at a time
flock -w 14400 /root/artifacts/slave_gpu.lock timeout 4h ssh "${SSH_OPTS[@]}" -p "$SLAVE_PORT" "$SLAVE_HOST" \
  "/root/release_python_slave.sh /root/slave_multipass_writeback.py \
     --sequences-json '$REMOTE_SEQ_JSON' --epoch '$EPOCH' --passes '$PASSES' \
     --gpus 0,1,2,3 --out '$REMOTE_NDJSON'"

echo "[4/5] pulling NDJSON result back to master"
scp -q -P "$SLAVE_PORT" -o StrictHostKeyChecking=accept-new "${SLAVE_HOST}:${REMOTE_NDJSON}" "$LOCAL_NDJSON"

echo "[5/5] ingesting into master's local sqlite DB"
cd /root && pixi run --manifest-path /root/pixi.toml --locked --environment bgen python3 /root/tools/ingest_slave_multipass.py --ndjson "$LOCAL_NDJSON" $DRY_RUN

echo "RUN_SLAVE_MULTIPASS_DONE epoch=$EPOCH"
