#!/usr/bin/env bash
# Slave top-up worker: keep the sequences we are about to commit at >= 20 observations by scoring the
# missing rounds on the SLAVE box's GPUs (see tools/topup_slave.py).  Runs ON MASTER ONLY, in tmux
# session topup-slave (kept alive by session_watchdog.sh), next to the master's own topup-publishable.
#
# Additive and failure-isolated: the python loop never raises out of a pass; if the process itself dies
# it is restarted here after a pause and the failure is recorded.  One instance only (lock file).
set -u
OUT=/root/artifacts/topup_slave.out
PY=/root/.pixi/envs/chain310/bin/python   # stdlib only; the importer it launches uses the bgen interpreter
echo "$(date -Is) topup_slave_loop starting" >> "$OUT"
while true; do
  (cd /root && PYTHONPATH=/root nice -n 10 "$PY" /root/tools/topup_slave.py loop) >> "$OUT" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date -Is) topup_slave exited rc=$rc" >> "$OUT"
    printf '{"at": %s, "component": "topup_slave_loop", "marker": "TOPUP_SLAVE_EXITED", "context": {"rc": %s}}\n' \
      "$(date +%s)" "$rc" >> /root/artifacts/pipeline_failures.jsonl
  fi
  sleep 60
done
