#!/bin/bash
# Durable Mac-side keeper: pulls both bank boxes on a fixed cadence so the
# molecule bank survives a box dying.  Detached from any Claude session.
#   sync   every 40 min  (merge both banks, push the union back)
#   backup every 30 min  (mirror bank.sqlite + json state + rescored table)
ROOT=/Users/tvgdeena/nova-nanobody
LOG=$ROOT/logs/bank_keeper.log
cd "$ROOT" || exit 1
echo "$$" > "$ROOT/state/bank_keeper.pid"

next_sync=0
next_backup=0
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$next_backup" ]; then
    echo "[$(date -u +%FT%TZ)] backup start" >> "$LOG"
    timeout 900 /usr/bin/python3 tools/bank_backup.py >> "$LOG" 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] backup exit=$rc" >> "$LOG"
    next_backup=$(( $(date +%s) + 1800 ))
  fi
  now=$(date +%s)
  # NOVA_KEEPER_SYNC=0 keeps the boxes' banks SEPARATE. The merge is what makes
  # a bank survive a box dying, so it is on by default -- but it also makes the
  # two boxes converge, which silently destroys any A/B between their searches:
  # a best-20 comparison after one merge is measuring the union, not the arms.
  # Backups keep running either way, so pausing the merge costs redundancy, not
  # the bank.
  if [ "${NOVA_KEEPER_SYNC:-1}" != "0" ] && [ "$now" -ge "$next_sync" ]; then
    echo "[$(date -u +%FT%TZ)] sync start" >> "$LOG"
    timeout 1800 /usr/bin/python3 tools/bank_sync.py >> "$LOG" 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] sync exit=$rc" >> "$LOG"
    next_sync=$(( $(date +%s) + 2400 ))
  elif [ "${NOVA_KEEPER_SYNC:-1}" = "0" ] && [ "$now" -ge "$next_sync" ]; then
    echo "[$(date -u +%FT%TZ)] sync PAUSED (NOVA_KEEPER_SYNC=0)" >> "$LOG"
    next_sync=$(( $(date +%s) + 2400 ))
  fi
  sleep 60
done
