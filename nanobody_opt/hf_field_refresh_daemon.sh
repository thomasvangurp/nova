#!/bin/bash
# Continuous refresh daemon for the hf_field_20x_rank_sum parent pool.
# Runs the refresh loop body every 30 min; survives transient failures
# (retries next tick). One instance (the inner loop script holds a lock).
# The inner script skips while the one-shot backfills hold the GPUs, and the
# refresh itself is idempotent, so this is safe to leave running forever.
LOG=/root/hf_field_refresh_logs/daemon.log
mkdir -p /root/hf_field_refresh_logs
echo "$(date -u +%FT%TZ) refresh daemon started (pid $$)" >> "$LOG"
while true; do
  bash /root/hf_field_refresh_loop.sh >> "$LOG" 2>&1
  sleep 1800
done
