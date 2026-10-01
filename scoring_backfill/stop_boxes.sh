#!/bin/bash
# Stand down: PAUSE (stop) all 10 scoring boxes — preserves disk/data, stops billing.
for id in 53449398 53449401 53449408 53449412 53449418 53449420 53449423 53449426 53449432 53449435; do
  r=$(vastai stop instance $id 2>&1 | tail -1)
  echo "$id: $r"
done
echo "STOPPED_ALL"
