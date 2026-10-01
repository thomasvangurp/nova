#!/bin/bash
set -euo pipefail
echo "$(date +%s) BOOTSTRAP_BEGIN"
ln -sfn /root/fastscore/pinned-530f5dcbc54d /root/fastscore/release
chmod +x /root/bgen_python.sh /root/fastscore/release/release_python.sh 2>/dev/null || true

if [ ! -x /usr/local/bin/pixi ]; then
  curl -fsSL -o /tmp/pixi.tar.gz https://github.com/prefix-dev/pixi/releases/download/v0.73.0/pixi-x86_64-unknown-linux-musl.tar.gz
  tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin
fi
echo "$(date +%s) PIXI_$(/usr/local/bin/pixi --version 2>&1 | head -1)"

/usr/local/bin/pixi install --manifest-path /root/pixi.toml --locked --environment bgen
echo "$(date +%s) BGEN_INSTALLED"

HF_XET_HIGH_PERFORMANCE=1 /root/bgen_python.sh /root/nanobody_opt/fetch_checkpoints.py --local-dir /root/.boltzgen --specs-json "$(cat /root/ckpt_specs.json)"
echo "$(date +%s) WEIGHTS_DONE"

export PYTHONPATH=/root/bgen_shim:/root/fastscore/release/boltzgen_src
/usr/local/bin/pixi run --manifest-path /root/pixi.toml --environment bgen env python -c 'import torch, boltzgen; print("VERIFY_OK", torch.cuda.device_count(), boltzgen.__file__)'
echo "$(date +%s) BOOTSTRAP_DONE"
