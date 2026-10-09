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

# Never compete with the producer's children scoring: skip while the GPUs are
# busy (the children lane runs the exact scorer at ~100% util). The refresh
# only runs in the idle gaps, so the 20x parent pool never steals GPU from a
# live children submission.
#
# Starvation guard: the gate is a single instantaneous sample per 30-min tick,
# so a box whose GPUs are busy at most ticks (master, once the children lane
# scores continuously) could skip forever and the parent pool would go stale
# silently. If the last completed refresh is older than MAX_SKIP_HOURS, run
# anyway. Baked-in default (the daemon passes no env).
MAX_SKIP_HOURS=${HF_REFRESH_MAX_SKIP_HOURS:-3}
MAXUTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
if [ -n "$MAXUTIL" ] && [ "$MAXUTIL" -gt 40 ] 2>/dev/null; then
  LAST_DONE=$(grep ' DONE$' "$LOGDIR/loop.log" 2>/dev/null | tail -1 | cut -d' ' -f1)
  LAST_DONE_S=$(date -u -d "$LAST_DONE" +%s 2>/dev/null || echo 0)
  AGE_H=$(( ( $(date -u +%s) - LAST_DONE_S ) / 3600 ))
  if [ "$AGE_H" -lt "$MAX_SKIP_HOURS" ]; then
    echo "$(date -u +%FT%TZ) SKIP gpu busy $MAXUTIL% (last DONE ${AGE_H}h ago)" >> "$LOGDIR/loop.log"
    exit 0
  fi
  echo "$(date -u +%FT%TZ) OVERRIDE gpu busy $MAXUTIL% but last DONE ${AGE_H}h ago >= ${MAX_SKIP_HOURS}h" >> "$LOGDIR/loop.log"
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

# Auto-scope the scorer to the host's GPUs (8x master vs 2x dashboard box).
NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
GPUS=$(seq -s, 0 $((NGPU - 1)) 2>/dev/null)
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

/root/.pixi/envs/bgen/bin/python /root/hf_field_refresh.py \
  --gpus "$GPUS" >> "$LOGDIR/refresh.log" 2>&1

echo "$(date -u +%FT%TZ) DONE" >> "$LOGDIR/loop.log"
