#!/usr/bin/env bash
# Continuous replay-queue worker for the slave. The rival replay
# (slave-multipass-loop) keeps absolute priority: no queue batch starts while
# any recent epoch's leaderboard submissions are short of 10 passes. A batch is
# ~10 min of GPU time, and a rival run that arrives mid-batch waits on the
# shared flock for at most that long. Runs ON MASTER ONLY.
set -u
LOG=/root/artifacts/replay_queue_loop.log
PY=/root/.pixi/envs/bgen/bin/python
echo "$(date -Is) replay_queue_loop starting" >> "$LOG"
while true; do
  if ! $PY /root/tools/rival_coverage_pending.py >> "$LOG" 2>&1; then
    sleep 120
    continue
  fi
  /root/tools/run_replay_queue_batch.sh >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    continue
  elif [ "$rc" -eq 4 ]; then
    sleep 300
  else
    echo "$(date -Is) REPLAY_QUEUE_LOOP_FAILED rc=$rc" >> "$LOG"
    printf '{"at": %s, "component": "replay_queue_loop", "marker": "REPLAY_QUEUE_FAILED", "context": {"rc": %s}}\n' \
      "$(date +%s)" "$rc" >> /root/artifacts/pipeline_failures.jsonl
    sleep 180
  fi
done
