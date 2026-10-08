#!/bin/bash
# Continuous frontier mining (single box).  Each epoch: rebuild the fresh top-50,
# generate 6-edit-frontier children (1-2 framework edits, A/S/V, with the
# parent's free final_score), shortlist by that parent score, 14x-score the
# shortlist, ingest into children_frontier.  The lane submits the top-13 by the
# 14x score.  Runs forever -- not a one-shot.
set -u
PIXI=/usr/local/bin/pixi
MANIFEST=/root/pixi.toml
DB=/root/artifacts/nanobody_mutations.sqlite
IDX=/root/artifacts/top50_cdr3_index.json
ARCHIVE=$(ls -t /root/live/*/archive.csv 2>/dev/null | head -1)
LOG=/root/artifacts/frontier_loop.log
STAMP=/root/artifacts/frontier_last_epoch
DEPTH=14
SHORTLIST=200

run_metanano() { "$PIXI" run --manifest-path "$MANIFEST" --locked -e metanano "$@"; }

while true; do
  ep=$(python3 -c "import json;print(json.load(open('/root/artifacts/chain_state.json'))['epoch'])" 2>/dev/null)
  last=$(cat "$STAMP" 2>/dev/null || echo $((ep - 3)))
  if [ -n "$ep" ] && [ "$ep" != "$last" ]; then
    echo "$(date -u +%FT%TZ) mining frontier for epoch $ep" >> "$LOG"
    # 1) validator top-50 from the FULL HF archive (all history), never the API.
    #    A 15-epoch API window indexes a different basin and over-filters (the
    #    25454/25455 wholesale rejections) -- see build_top50_index.py docstring.
    HF=/root/P20809_nanobodies.csv
    run_metanano python /root/nanobody_opt/fetch_hf_archive.py "$HF" >> "$LOG" 2>&1
    run_metanano python /root/nanobody_opt/build_top50_index.py "$HF" "$IDX" "$DB" >> "$LOG" 2>&1
    # 2) generate frontier children (with parent_score)
    run_metanano python /root/nanobody_opt/longtail_children.py "$IDX" /root/artifacts/frontier_shard.json 60 >> "$LOG" 2>&1
    # 3) shortlist by the parent's free final_score (no scoring needed for this)
    python3 -c "
import json
shard = json.load(open('/root/artifacts/frontier_shard.json'))
shard.sort(key=lambda c: (c.get('parent_score', 1e9), c['sequence']))
json.dump([{'sequence': c['sequence'], 'epoch': c.get('epoch', 25456)} for c in shard[:$SHORTLIST]], open('/root/artifacts/frontier_deep_shard.json', 'w'))
print('shortlisted', min($SHORTLIST, len(shard)), 'of', len(shard))
" >> "$LOG" 2>&1
    # 4) 14x deep-score the shortlist
    if [ -s /root/artifacts/frontier_deep_shard.json ]; then
      rm -rf /root/bgen_frontier_deep /root/artifacts/frontier_deep.json
      python3 /root/score_worker.py --shard /root/artifacts/frontier_deep_shard.json \
        --results /root/artifacts/frontier_deep.json --work-root /root/bgen_frontier_deep \
        --batch-size 64 --gpus 0,1,2,3,4,5,6,7 --passes "$DEPTH" --gpu-batch-size 96 >> "$LOG" 2>&1
      # 5) ingest into children_frontier
      mkdir -p /root/deep_ingest_frontier
      cp /root/artifacts/frontier_deep.json /root/deep_ingest_frontier/A.json
      run_metanano python /root/ingest_backfill.py /root/deep_ingest_frontier "$ARCHIVE" "$DB" children_frontier "$DEPTH" >> "$LOG" 2>&1
    fi
    echo "$ep" > "$STAMP"
    echo "$(date -u +%FT%TZ) epoch $ep done" >> "$LOG"
  fi
  sleep 180
done
