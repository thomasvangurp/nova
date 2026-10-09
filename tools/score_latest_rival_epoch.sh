#!/bin/bash
# Score the most recent completed epoch's rivals, during a quiet moment.
#
# Run from the box cron every ~30 min. Guards:
#   - skip epochs already scored (idempotent, one JSON per epoch)
#   - skip when GPUs are busy (the producer's race owns the GPUs; this
#     scoring is strictly opportunistic and must not contend)
#   - try the latest two epoch dirs, because the newest one may still be
#     in-progress (leaderboard API returns empty -> NO_RIVALS, no JSON).
set -u
# One scoring run at a time (a 10-pass run can outlast the cron interval).
exec 9>/tmp/score_rival_epoch.lock
flock -n 9 || exit 0

BIN=/root/multipass_all_rivals.py
OUT=/root/multipass_all_rivals
LOG="$OUT/score_latest.log"
mkdir -p "$OUT"

LATEST=$(ls -d /root/live/epoch*/producer 2>/dev/null \
  | sed 's#.*/epoch##;s#/producer##' | sort -n | tail -1)
[ -z "$LATEST" ] && exit 0

MAXUTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | sort -n | tail -1)
if [ -n "$MAXUTIL" ] && [ "$MAXUTIL" -gt 20 ]; then
  echo "$(date +%m-%d_%H:%M) skip epoch $LATEST: gpu busy $MAXUTIL%" >> "$LOG"
  exit 0
fi

for EPOCH in $(seq "$LATEST" -1 $((LATEST - 5))); do
  [ -f "$OUT/$EPOCH/multipass_$EPOCH.json" ] && continue
  echo "$(date +%m-%d_%H:%M) scoring epoch $EPOCH" >> "$LOG"
  /root/bgen_python.sh "$BIN" "$EPOCH" 10 >> "$LOG" 2>&1
  [ -f "$OUT/$EPOCH/multipass_$EPOCH.json" ] && break
done
