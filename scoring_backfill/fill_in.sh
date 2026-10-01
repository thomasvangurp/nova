#!/bin/bash
# Stage config + data + pixi manifest/lock + pixi binary on every box (small files,
# non-overlapping with the coordinator's rsync fan-out).
REPO=/Users/tvgdeena/nova-nanobody
declare -a BOXES=(
  "53449398 ssh2.vast.ai 13756"
  "53449401 ssh4.vast.ai 18324"
  "53449408 ssh1.vast.ai 38978"
  "53449412 ssh1.vast.ai 25066"
  "53449418 ssh3.vast.ai 34196"
  "53449420 ssh6.vast.ai 27242"
  "53449423 ssh3.vast.ai 31808"
  "53449426 ssh8.vast.ai 36372"
  "53449432 217.138.104.229 10427"
  "53449435 ssh3.vast.ai 20070"
)
for b in "${BOXES[@]}"; do
  set -- $b; ID=$1; HOST=$2; PORT=$3
  (
    LOG=/tmp/fillin_${ID}.log
    SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
    echo "$(date +%s) FILLIN_START $ID" > $LOG
    ssh $SSHO -p $PORT root@$HOST "mkdir -p /root/config /root/data/msa_files /root/data/structures" >> $LOG 2>&1
    scp $SSHO -P $PORT $REPO/scoring_backfill/stage/config/config.yaml $REPO/scoring_backfill/stage/config/boltzgen_config.yaml root@$HOST:/root/config/ >> $LOG 2>&1
    scp $SSHO -P $PORT $REPO/scoring_backfill/stage/data/msa_files/P20809.a3m root@$HOST:/root/data/msa_files/ >> $LOG 2>&1
    scp $SSHO -P $PORT $REPO/scoring_backfill/stage/data/structures/6O4O.cif root@$HOST:/root/data/structures/ >> $LOG 2>&1
    scp $SSHO -P $PORT $REPO/pixi.toml $REPO/pixi.lock root@$HOST:/root/ >> $LOG 2>&1
    if ! ssh $SSHO -p $PORT root@$HOST "test -x /usr/local/bin/pixi" >> $LOG 2>&1; then
      ssh $SSHO -p $PORT root@$HOST "curl -fsSL -o /tmp/pixi.tar.gz https://github.com/prefix-dev/pixi/releases/download/v0.73.0/pixi-x86_64-unknown-linux-musl.tar.gz && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin" >> $LOG 2>&1
    fi
    ssh $SSHO -p $PORT root@$HOST "chmod +x /root/bgen_python.sh /root/fastscore/release/release_python.sh 2>/dev/null; test -f /root/config/config.yaml && test -f /root/data/msa_files/P20809.a3m && test -f /root/data/structures/6O4O.cif && test -x /usr/local/bin/pixi && echo FILLIN_OK" >> $LOG 2>&1
    echo "$(date +%s) FILLIN_DONE $ID" >> $LOG
  ) &
done
wait
echo "FILLIN_ALL_DONE"
