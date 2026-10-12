#!/usr/bin/env bash
# Shadow tournament worker (master only): record every ordering rule's top-13 for each closed
# epoch, evaluate an epoch once all those sequences (and the rivals') have enough independent
# replay passes, and keep REPORT.md current.  The replay itself is done by the replay queue
# (tier 0 in tools/replay_queue.py), so this loop only reads the database.
set -u
DIR=/root/artifacts/shadow_tournament
LOG=$DIR/loop.log
PY=/root/.pixi/envs/bgen/bin/python
mkdir -p "$DIR"
echo "$(date -Is) shadow_tournament_loop starting" >> "$LOG"
while true; do
  $PY /root/tools/shadow_tournament.py run-once >> "$LOG" 2>&1 \
    || echo "$(date -Is) SHADOW_RUN_FAILED rc=$?" >> "$LOG"
  sleep 300
done
