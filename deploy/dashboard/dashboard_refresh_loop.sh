#!/bin/bash
# Keeps the multipass dashboard data (ours + rivals, 10-pass) current:
# refreshes the HF archive, then scores the latest unscored epoch.
set -u
LOG=/root/dashboard_refresh_loop.log
while true; do
  echo "$(date -u +%FT%TZ) cycle start" >> "$LOG"
  /root/bgen_python.sh -m nanobody_opt.fetch_hf_archive /root/hf_archive.csv >> "$LOG" 2>&1
  bash /root/score_latest_rival_epoch.sh
  sleep 300
done
