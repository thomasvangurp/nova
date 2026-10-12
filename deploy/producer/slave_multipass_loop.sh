#!/usr/bin/env bash
# Continuous slave replay of EVERY leaderboard submission (all miners) plus our
# qualified shortlist, PASSES passes each. Walks the last LOOKBACK closed
# epochs newest-first, so the freshest submissions (the best parent
# candidates) are covered first, and keeps retrying an epoch until nothing in
# it is short of PASSES passes. After any epoch that did real work it starts
# again from the newest, so a newly closed epoch jumps the queue. Runs ON
# MASTER ONLY; master is the only process that writes the DB.
set -u
LOG=/root/artifacts/slave_multipass_loop.log
STATE_DIR=/root/artifacts/slave_multipass_all_done
PASSES="${SLAVE_MULTIPASS_PASSES:-10}"
LOOKBACK="${SLAVE_MULTIPASS_LOOKBACK:-20}"
mkdir -p "$STATE_DIR"

echo "$(date -Is) slave_multipass_loop (all submissions) starting, passes=$PASSES lookback=$LOOKBACK" >> "$LOG"

while true; do
  LATEST=$(python3 -c "import json; print(json.load(open('/root/artifacts/chain_state.json'))['epoch'])" 2>/dev/null)
  worked=0
  if [ -n "$LATEST" ]; then
    for ((TARGET = LATEST - 1; TARGET >= LATEST - LOOKBACK; TARGET--)); do
      [ -f "$STATE_DIR/$TARGET.done" ] && continue
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
        break
      elif [ "$rc" -eq 3 ]; then
        continue
      else
        echo "$(date -Is) SLAVE_MULTIPASS_LOOP_FAILED epoch=$TARGET rc=$rc" >> "$LOG"
        printf '{"at": %s, "component": "slave_multipass_loop", "marker": "SLAVE_MULTIPASS_FAILED", "context": {"epoch": %s}}\n' \
          "$(date +%s)" "$TARGET" >> /root/artifacts/pipeline_failures.jsonl
      fi
    done
  fi
  [ "$worked" -eq 1 ] || sleep 300
done
