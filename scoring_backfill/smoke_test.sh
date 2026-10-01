#!/bin/bash
# Smoke-test the scorer on one box: 2 sequences, 1 pass, verify 10 metrics out.
HOST=$1; PORT=$2
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
ssh $SSHO -p $PORT root@$HOST 'bash -s' <<'INNER'
set -e
mkdir -p /root/smoke && cd /root/smoke
cat > in.json <<'J'
[{"sequence": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGGYYDSSGYPLDYWGQGTLVTVSS"}]
J
/root/fastscore/pinned-530f5dcbc54d/release_python.sh \
  /root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py \
  --sequences /root/smoke/in.json --config /root/config/config.yaml --target P20809 \
  --gpus 0 --shards-per-gpu 1 --batch-size 1 --shard-root /root/smoke/work --out /root/smoke/out.json \
  --synth-reference /root/fastscore/pinned-530f5dcbc54d/refdesign/0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif \
  2>&1 | tail -3
python3 - <<'PY'
import json
p = json.load(open('/root/smoke/out.json'))
comp = p["per_nanobody_components"]
n = 0
for by_seq in comp.values():
    for seq, bt in by_seq.items():
        m = bt.get("P20809", {})
        n += 1
        print("SMOKE_METRICS", len(m), sorted(m.keys())[:3], "...")
        assert len(m) == 10, f"expected 10 metrics got {len(m)}"
print("SMOKE_OK sequences=%d" % n)
PY
INNER
