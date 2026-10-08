#!/bin/bash
# Post-mirror setup on nova-nanobody-throughput (the new scorer box).
# Installs pixi 0.73 (pinned to pixi.toml requires-pixi >=0.73,<0.74) and
# syncs ONLY the bgen + metanano environments. No chain/miner env, no wallet.
set -euo pipefail
export PATH=/usr/local/bin:/root/.pixi/bin:$PATH

echo "== disk =="; df -h / | tail -1
echo "== gpus =="; nvidia-smi -L 2>/dev/null | wc -l; nvidia-smi -L 2>/dev/null | head -8

# 1. pixi 0.73.0 (static musl binary, deterministic)
if ! /usr/local/bin/pixi --version 2>/dev/null | grep -q 0.73; then
  echo "== installing pixi 0.73.0 =="
  command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl)
  curl -fsSL -o /usr/local/bin/pixi \
    https://github.com/pixi-rs/pixi/releases/download/v0.73.0/pixi-x86_64-unknown-linux-musl
  chmod +x /usr/local/bin/pixi
fi
pixi --version

# 2. Build bgen + metanano from the copied lock file (skips chain310 / validator-bgen)
echo "== pixi install (bgen + metanano) =="
cd /root
pixi install -e bgen
pixi install -e metanano

echo "SETUP_DONE"
