#!/bin/bash
# Retry vastai start on the 5 exited boxes every ~15 min; report when one comes up.
while true; do
  any_running=0
  for id in 53449398 53449408 53449423 53449426 53449432; do
    st=$(vastai show instance $id --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); i=d if isinstance(d,dict) else d; print(i.get('actual_status'))" 2>/dev/null)
    if [ "$st" = "running" ]; then
      echo "$(date +%H:%M:%S) $id RUNNING"
      any_running=1
    else
      echo "$(date +%H:%M:%S) $id $st -> retrying start"
      vastai start instance $id 2>&1 | tail -1
    fi
  done
  sleep 900
done
