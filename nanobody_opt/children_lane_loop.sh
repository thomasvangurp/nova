#!/bin/bash
# Publisher replacement: fleet artifact from the children_20x bank, with the
# validator's duplicate gate replayed against a FRESH top-50 built from the
# FULL HF archive (build_top50_index.py), not the leaderboard API or the stale
# local archive.
#
# Original publisher argv: /root/artifacts/publisher_argv.PRODUCER-BACKUP.bin
# ROLLBACK: restore it over publisher_argv.bin, then
#   tmux kill-session -t nova-nanobody-publisher
# and the watchdog restores the original publisher within 60s.
#
# MUST run under the metanano pixi env: --cdr3-index only gates when CDRs can be
# extracted for OUR candidates too. Under plain python3 the extractor is absent,
# every candidate comes back unannotated, and the filter FAILS OPEN silently.
set -u
PIXI=/usr/local/bin/pixi
MANIFEST=/root/pixi.toml
ART=/root/artifacts/nanobody_result.json
CHAIN=/root/artifacts/chain_state.json
IDX=/root/artifacts/top50_cdr3_index.json
LOG=/root/artifacts/children_lane.log

run_metanano() { "$PIXI" run --manifest-path "$MANIFEST" --locked -e metanano "$@"; }

while true; do
  ep=$(python3 -c "import json;print(json.load(open('$CHAIN'))['epoch'])" 2>/dev/null)
  cur=$(python3 -c "import json;print(json.load(open('$ART'))['epoch'])" 2>/dev/null)
  if [ -n "$ep" ] && [ "$ep" != "$cur" ]; then
    echo "$(date -u +%FT%TZ) epoch $ep (artifact was $cur)" >> $LOG
    # Rebuild the top-50 gate from the FULL HF archive (all history) each epoch,
    # using the validator's own rank_binders. The API's 15-epoch window indexes
    # a different basin and over-filters (25454/25455 wholesale rejections).
    HF=/root/P20809_nanobodies.csv
    run_metanano python /root/nanobody_opt/fetch_hf_archive.py "$HF" >> $LOG 2>&1
    run_metanano python /root/nanobody_opt/build_top50_index.py "$HF" "$IDX" >> $LOG 2>&1
    GATE=""
    [ -s "$IDX" ] && GATE="--cdr3-index $IDX"
    cd /root && run_metanano python -m nanobody_opt.children_submit \
      --database /root/artifacts/nanobody_mutations.sqlite \
      --chain-state $CHAIN \
      --coldkey 5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ \
      --ledger /root/artifacts/nanobody_published_sequences.txt \
      --source children_8x \
      --top-n 300 \
      $GATE \
      --output $ART 2>&1 | grep -vE 'CHILD_SUBMIT|CHILD_CDR3_DROPPED' >> $LOG
    echo "$(date -u +%FT%TZ) rc=$? epoch $ep" >> $LOG
  fi
  sleep 30
done
