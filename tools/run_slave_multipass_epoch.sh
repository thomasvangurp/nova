#!/usr/bin/env bash
set -euo pipefail
exec /root/.pixi/envs/bgen/bin/python /root/tools/run_batched_rival_epoch.py "${1:?epoch required}" --passes "${2:-20}" --max-sequences 8
