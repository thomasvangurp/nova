#!/usr/bin/env bash
set -euo pipefail
cd /root
bash /root/deploy_gate.sh
# The installed launch contract is verified before stopping the old process.
tmux kill-session -t nova-nanobody-producer 2>/dev/null || true
tmux new-session -d -s nova-nanobody-producer 'bash /root/producer_relaunch.sh'
