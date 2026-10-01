#!/bin/bash
# Stand down: destroy all 10 scoring boxes (run AFTER merge + DB verification).
for id in 53449398 53449401 53449408 53449412 53449418 53449420 53449423 53449426 53449432 53449435; do
  echo "destroying $id ..."
  vastai destroy instance $id --yes 2>&1 | tail -1
done
echo "DESTROYED_ALL"
