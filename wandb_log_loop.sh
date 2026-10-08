#!/bin/bash
# /root/wandb_log_loop.sh
# Wrapper/scheduler for the SN68 nanobody W&B logger.
#
# Sources /root/.wandb_env (chmod 600) if present so the W&B API key is
# available to the child process WITHOUT ever appearing on a command line or
# in a cron environment.
#
# Usage:
#   wandb_log_loop.sh once   -> run the logger once and exit
#   wandb_log_loop.sh        -> poll every 30 min forever (for tmux)
set -u

PY=/root/wandb_logger_venv/bin/python
SCRIPT=/root/wandb_log_epoch.py
ENVFILE=/root/.wandb_env
INTERVAL="${WANDB_LOG_INTERVAL_SEC:-1800}"

[ -f "$ENVFILE" ] && . "$ENVFILE"

run_once() {
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb-loop] no WANDB_API_KEY yet; waiting (put it in $ENVFILE)"
  else
    "$PY" "$SCRIPT" || true
  fi
}

if [ "${1:-loop}" = "once" ]; then
  run_once
  exit 0
fi

while true; do
  run_once
  sleep "$INTERVAL"
done
