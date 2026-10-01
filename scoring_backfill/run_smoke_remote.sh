#!/bin/bash
# Detached smoke test launcher (runs on the box).
cd /root/smoke && rm -rf work out.json
setsid bash -c '/root/fastscore/pinned-530f5dcbc54d/release_python.sh /root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py --sequences /root/smoke/in.json --config /root/config/config.yaml --target P20809 --gpus 0 --shards-per-gpu 1 --batch-size 1 --shard-root /root/smoke/work --out /root/smoke/out.json --synth-reference /root/fastscore/pinned-530f5dcbc54d/refdesign/0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif > /root/smoke/run.log 2>&1 < /dev/null' &
disown
echo "SMOKE_LAUNCHED"
