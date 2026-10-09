#!/bin/bash
# Dashboard-box multipass: score the latest finalized epoch's rivals.
set -u
exec 9>/tmp/score_rival_epoch.lock
flock -n 9 || exit 0

BIN=/root/multipass_all_rivals.py
OUT=/root/multipass_all_rivals
LOG="$OUT/score_latest.log"
ARCHIVE=/root/hf_archive.csv
CONFIG=/root/config/config.yaml
mkdir -p "$OUT"

# An epoch is done only when its output exists AND holds at least one rival.
# A file with only our own fleet means the run beat the validator's (variable)
# publish lag, so it is NOT cached; re-score it once the leaderboard is out.
# (multipass_all_rivals.py now refuses to write such a file; this also heals
# any written before that fix, e.g. 25616-25618 on 2026-10-09.)
complete() {
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]]; sys.exit(0 if d.get("rivals") else 1)' \
    "$OUT/$1/multipass_$1.json" "$1" 2>/dev/null
}

# No producer here; read the current epoch from the master's chain_state.
LATEST=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p 10616 root@ssh9.vast.ai 'python3 -c "import json; print(json.load(open(\"/root/artifacts/chain_state.json\"))[\"epoch\"])"' 2>/dev/null | tail -1)
[ -z "$LATEST" ] && exit 0

# The current epoch isn't finalized (API 404); start one back.
START=$((LATEST - 1))
for EPOCH in $(seq "$START" -1 $((START - 8))); do
  complete "$EPOCH" && continue
  if [ -f "$OUT/$EPOCH/multipass_$EPOCH.json" ]; then
    echo "$(date +%m-%d_%H:%M) rescoring epoch $EPOCH (cached output has no rivals)" >> "$LOG"
  else
    echo "$(date +%m-%d_%H:%M) scoring epoch $EPOCH" >> "$LOG"
  fi
  # Exit 3 = leaderboard not published yet, 1 = published but no rivals:
  # neither writes output, so the epoch is retried next cycle.
  # Exit 4 = JSON written but the DB write failed; anything else = crash.
  # Those used to vanish into this log with nothing reading it
  # (ARCHITECTURE_REDESIGN P0b): log a MULTIPASS_FAILED marker and append to
  # the box failure log that tools/check_mining_liveness.py alerts on.
  /root/bgen_python.sh "$BIN" "$EPOCH" 10 "$ARCHIVE" "$CONFIG" >> "$LOG" 2>&1
  RC=$?
  case "$RC" in
    0|1|3) ;;
    *)
      echo "$(date +%m-%d_%H:%M) MULTIPASS_FAILED epoch=$EPOCH rc=$RC" >> "$LOG"
      mkdir -p /root/artifacts
      printf '{"at": %s, "component": "score_latest_rival_epoch", "marker": "MULTIPASS_FAILED", "error": "multipass_all_rivals.py rc=%s", "context": {"epoch": %s, "rc": %s}}\n' \
        "$(date +%s)" "$RC" "$EPOCH" "$RC" >> /root/artifacts/pipeline_failures.jsonl
      ;;
  esac
  complete "$EPOCH" && break
done
