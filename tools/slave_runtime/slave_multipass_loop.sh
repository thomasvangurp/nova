#!/usr/bin/env bash
# Continuous slave replay of EVERY leaderboard submission (all miners) plus our
# qualified shortlist, PASSES passes each. Walks the last LOOKBACK closed
# epochs from the first known missing epoch, alternating old/new work.
# The freshest submissions (the best parent
# candidates) are covered first, and keeps retrying an epoch until nothing in
# it is short of PASSES passes. After any epoch that did real work it starts
# again from the newest, so a newly closed epoch jumps the queue. Runs ON
# MASTER ONLY; master is the only process that writes the DB.
set -u
LOG=/root/artifacts/slave_multipass_loop.log
PASSES="${SLAVE_MULTIPASS_PASSES:-20}"
EARLIEST="${SLAVE_MULTIPASS_FIRST_EPOCH:-25737}"
STATE_DIR="/root/artifacts/slave_multipass_${PASSES}x_done"
DIRECTION=newest
mkdir -p "$STATE_DIR"

echo "$(date -Is) slave_multipass_loop (all submissions) starting, passes=$PASSES first_epoch=$EARLIEST" >> "$LOG"

while true; do
  LATEST=$(python3 -c "import json; print(json.load(open('/root/artifacts/chain_state.json'))['epoch'])" 2>/dev/null)
  worked=0
  if [ -n "$LATEST" ]; then
    if [ "$DIRECTION" = newest ]; then
      TARGETS=$(seq "$((LATEST - 1))" -1 "$EARLIEST")
    else
      TARGETS=$(seq "$EARLIEST" "$((LATEST - 1))")
    fi
    for TARGET in $TARGETS; do
      # Only this pass-count namespace can certify completion. Recent boards
      # are rechecked so late roster evidence cannot be hidden by a marker.
      if [ "$TARGET" -lt "$((LATEST - 3))" ] && [ -f "$STATE_DIR/$TARGET.done" ]; then
        continue
      fi
      echo "$(date -Is) epoch $TARGET: checking coverage (passes=$PASSES)" >> "$LOG"
      /root/tools/run_slave_multipass_epoch.sh "$TARGET" "$PASSES" >> "$LOG" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ]; then
        if tail -n 1 "$LOG" | grep -q "nothing left to score"; then
          touch "$STATE_DIR/$TARGET.done"
          echo "$(date -Is) epoch $TARGET fully covered" >> "$LOG"
          continue
        fi
        echo "$(date -Is) epoch $TARGET scored; rechecking from newest" >> "$LOG"
        worked=1
        if [ "$DIRECTION" = newest ]; then DIRECTION=oldest; else DIRECTION=newest; fi
        break
      elif [ "$rc" -eq 3 ]; then
        continue
      elif [ "$rc" -eq 4 ]; then
        sleep 20
        worked=1
        break
      else
        echo "$(date -Is) SLAVE_MULTIPASS_LOOP_FAILED epoch=$TARGET rc=$rc" >> "$LOG"
        printf '{"at": %s, "component": "slave_multipass_loop", "marker": "SLAVE_MULTIPASS_FAILED", "context": {"epoch": %s}}\n' \
          "$(date +%s)" "$TARGET" >> /root/artifacts/pipeline_failures.jsonl
      fi
    done
  fi
  [ "$worked" -eq 1 ] || sleep 300
done
