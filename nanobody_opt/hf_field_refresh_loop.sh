#!/usr/bin/env bash
# Refresh the hf_field_20x_rank_sum parent pool each epoch.
#
# Idempotent (hf_field_refresh.py only scores sequences missing from the pool),
# and gated so it never competes with the one-shot backfill jobs
# (backfill_topup.py = rivals+HF top-200, score_top50_32.py = our top-50's 32)
# that are still holding the GPUs. Safe to run from cron every 30 minutes;
# the lockfile prevents overlapping runs and pgrep skips the busy windows.
set -u

LOCK=/tmp/hf_field_refresh.lock
LOGDIR=/root/hf_field_refresh_logs
mkdir -p "$LOGDIR"

# Skip while the one-shot 20x backfills are still running (GPUs busy).
if pgrep -f "backfill_topup.py" >/dev/null 2>&1 \
   || pgrep -f "score_top50_32.py" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) SKIP backfill still running" >> "$LOGDIR/loop.log"
  exit 0
fi

# Lock: no overlapping refreshes.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) SKIP lock held" >> "$LOGDIR/loop.log"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# Import the one-shot JSONs first (cheap, idempotent) so the pool gets the
# current backfill the moment it lands, then extend with anything new.
echo "$(date -u +%FT%TZ) RUN" >> "$LOGDIR/loop.log"
export SN68_MYSQL_HOST=127.0.0.1 SN68_MYSQL_PORT=3306
export NANOBODY_SCORER_DAEMON=1
export NANOBODY_SCORER_SOCKET=/root/artifacts/nanobody_scorer.sock
/root/.pixi/envs/bgen/bin/python /root/import_hf_field_20x.py \
  >> "$LOGDIR/import.log" 2>&1

/root/.pixi/envs/bgen/bin/python /root/hf_field_refresh.py \
  --gpus 0,1,2,3,4,5,6,7 >> "$LOGDIR/refresh.log" 2>&1

echo "$(date -u +%FT%TZ) DONE" >> "$LOGDIR/loop.log"
