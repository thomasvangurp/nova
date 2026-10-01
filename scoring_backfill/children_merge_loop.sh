#!/bin/bash
# Every ~5 min: sync children results to Pi/Mac, then merge to the central DB.
while true; do
  ./sync_children.sh >/dev/null 2>&1
  ./merge_children.sh >> /tmp/children_merge.out 2>&1
  sleep 300
done
