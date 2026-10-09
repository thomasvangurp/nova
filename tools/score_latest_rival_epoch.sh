#!/bin/bash
# Score the most recent completed epoch's rivals, during a quiet moment.
#
# Run from the box cron every ~30 min. Guards:
#   - skip epochs already scored (idempotent, one JSON per epoch)
#   - skip when GPUs are busy (the producer's race owns the GPUs; this
#     scoring is strictly opportunistic and must not contend)
#   - try the latest two epoch dirs, because the newest one may still be
#     in-progress (leaderboard API 404 -> LEADERBOARD_UNAVAILABLE, exit 3,
#     no JSON).
#   - an epoch counts as done only when its JSON holds at least one rival; a
#     JSON with only our own fleet (written before multipass_all_rivals.py
#     refused to) is re-scored rather than cached forever.
set -u
# One scoring run at a time (a 10-pass run can outlast the cron interval).
exec 9>/tmp/score_rival_epoch.lock
flock -n 9 || exit 0

BIN=/root/multipass_all_rivals.py
OUT=/root/multipass_all_rivals
LOG="$OUT/score_latest.log"
mkdir -p "$OUT"

complete() {
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]]; sys.exit(0 if d.get("rivals") else 1)' \
    "$OUT/$1/multipass_$1.json" "$1" 2>/dev/null
}

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
  complete "$EPOCH" && continue
  echo "$(date +%m-%d_%H:%M) scoring epoch $EPOCH" >> "$LOG"
  /root/bgen_python.sh "$BIN" "$EPOCH" 10 >> "$LOG" 2>&1
  complete "$EPOCH" && break
done
