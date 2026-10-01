#!/bin/bash
# Poll until the daemon socket is ready (or timeout).
HOST=$1; PORT=$2
for i in $(seq 1 30); do
  r=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$HOST 'test -S /root/artifacts/nanobody_scorer.sock && echo READY; tail -3 /root/daemon.log 2>/dev/null' 2>/dev/null)
  if echo "$r" | grep -q READY; then echo "DAEMON_READY after ${i}0s"; exit 0; fi
  if echo "$r" | grep -qiE "error|traceback|exception"; then echo "DAEMON_ERROR: $r"; exit 1; fi
  sleep 10
done
echo "DAEMON_TIMEOUT: $(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$HOST 'tail -15 /root/daemon.log' 2>/dev/null)"
exit 1
