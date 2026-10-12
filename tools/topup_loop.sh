#!/usr/bin/env bash
# Pro-active top-up worker (master only): keep the candidates we are about to publish at >= 20 observations.
# Runs inside the producer's pixi environment (boltzgen is on its path); see tools/topup_publishable.py.
cd /root
exec /usr/local/bin/pixi run --manifest-path /root/pixi.toml --locked --environment bgen python /root/tools/topup_publishable.py loop
