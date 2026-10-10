#!/usr/bin/env bash
# Refresh the hf_field_sum10_20x_rank_sum pool each epoch (every 30 min tick).
#
# Idempotent (hf_field_refresh.py only scores sequences missing from the pool).
# The GPU/time safety guard lives in hf_field_refresh.py ITSELF, so it also
# holds when the script is started by hand or from another wrapper:
#   - refuses (exit 3) unless no live-producer / GPU-scoring process runs on
#     this host and every GPU is idle in every sample of a ~60 s window;
#   - hard wall-clock cap + sequence-pass budget per invocation;
#   - a watchdog kills the scorer the moment a producer appears (exit 4);
#   - one instance per host (flock), never through the shared scorer daemon.
#
# 2026-10-10: removed this wrapper's old single-sample GPU gate and its
# "busy but last DONE >= 3h ago -> run anyway" OVERRIDE. That override is the
# path by which a refresh could start on top of a live producer race. There
# is no override now: a box that never gets a safe window logs STARVED (and
# hf_field_refresh.py logs every refusal to pipeline_failures.jsonl) instead.
set -u

LOCK=/tmp/hf_field_refresh.lock
LOGDIR=/root/hf_field_refresh_logs
mkdir -p "$LOGDIR"
MAX_SKIP_HOURS=${HF_REFRESH_MAX_SKIP_HOURS:-3}

# Lock: no overlapping wrapper runs (the python script also holds a flock).
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) SKIP lock held" >> "$LOGDIR/loop.log"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

echo "$(date -u +%FT%TZ) RUN" >> "$LOGDIR/loop.log"
export SN68_MYSQL_HOST=127.0.0.1 SN68_MYSQL_PORT=3306
# NOT NANOBODY_SCORER_DAEMON: the refresh must never feed the producer's
# shared warm scorer (hf_field_refresh.py forces it off for its scorer too).
# Import the one-shot JSONs first (cheap, DB-only, idempotent).
/root/.pixi/envs/bgen/bin/python /root/import_hf_field_20x.py \
  >> "$LOGDIR/import.log" 2>&1

# --gpus defaults to every GPU nvidia-smi lists on this host.
/root/.pixi/envs/bgen/bin/python /root/hf_field_refresh.py >> "$LOGDIR/refresh.log" 2>&1
RC=$?

case "$RC" in
  0) echo "$(date -u +%FT%TZ) DONE" >> "$LOGDIR/loop.log" ;;
  2) echo "$(date -u +%FT%TZ) DISABLED (rc=2; HF_FIELD_REFRESH_ENABLED not set)" >> "$LOGDIR/loop.log" ;;
  3|4)
    WHAT=REFUSED; [ "$RC" = 4 ] && WHAT=ABORTED
    REASON=$(grep -E 'HF_REFRESH_(GUARD|LABEL)_REFUSED|HF_REFRESH_ABORTED' "$LOGDIR/refresh.log" | tail -1 | cut -c1-300)
    echo "$(date -u +%FT%TZ) $WHAT rc=$RC $REASON" >> "$LOGDIR/loop.log"
    LAST_DONE=$(grep ' DONE$' "$LOGDIR/loop.log" 2>/dev/null | tail -1 | cut -d' ' -f1)
    LAST_DONE_S=$(date -u -d "$LAST_DONE" +%s 2>/dev/null || echo 0)
    AGE_H=$(( ( $(date -u +%s) - LAST_DONE_S ) / 3600 ))
    if [ "$AGE_H" -ge "$MAX_SKIP_HOURS" ]; then
      # Loud, never an override: the pool is stale because no safe window exists.
      echo "$(date -u +%FT%TZ) ALERT HF_REFRESH_STARVED last DONE ${AGE_H}h ago (>= ${MAX_SKIP_HOURS}h); NOT overriding" >> "$LOGDIR/loop.log"
      echo "HF_REFRESH_STARVED last DONE ${AGE_H}h ago" >&2
    fi
    ;;
  *) echo "$(date -u +%FT%TZ) FAIL rc=$RC (see refresh.log)" >> "$LOGDIR/loop.log" ;;
esac
exit 0
