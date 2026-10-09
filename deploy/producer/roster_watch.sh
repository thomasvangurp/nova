#!/bin/bash
# Watch the roster re-sync actually happening, epoch after epoch.
#
# The 2026-09-16 freeze was invisible precisely because every available signal
# looked healthy: chain_state.json kept a fresh block and a fresh timestamp
# over a five-day-old roster. So this does not check that the file is being
# written -- it checks the one thing that was false while everything else was
# true, which is that roster_block MOVES.
#
# A healthy roster is a sawtooth: age climbs to roughly one epoch (361 blocks)
# and drops back to near zero when the miner re-syncs at the boundary. This
# records every sample, calls out each boundary crossing as SYNC-OK or
# SYNC-MISSED, and reports the uid set so a silent change of slots is visible
# too.
set -u
LOG=/root/artifacts/roster_watch.log
STATE=/root/artifacts/chain_state.json
CARRY=/root/artifacts/.roster_watch_carry
INTERVAL=${ROSTER_WATCH_INTERVAL:-60}
# One epoch of slack. Past this the miner has skipped a boundary, which is the
# original bug reappearing.
FAIL_AGE=${ROSTER_WATCH_FAIL_AGE:-420}

echo "$(date -Is) roster watch started (interval ${INTERVAL}s, fail above ${FAIL_AGE} blocks)" >> "$LOG"
PREV_EPOCH=""
PREV_ROSTER=""
while true; do
    python3 - "$STATE" "$FAIL_AGE" "${PREV_EPOCH:-}" "${PREV_ROSTER:-}" "$CARRY" >> "$LOG" 2>&1 <<'PY'
import json, sys, time

state_path, fail_age, prev_epoch, prev_roster, carry_path = sys.argv[1:6]
stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
try:
    d = json.load(open(state_path))
    block, roster, length = int(d["block"]), int(d["roster_block"]), int(d["epoch_length"])
except Exception as error:
    print(f"{stamp} UNREADABLE {type(error).__name__}: {error}")
    # Leave the carry file alone: a transient unreadable sample must not erase
    # the boundary we are comparing against.
    raise SystemExit(0)

epoch = block // length
age = block - roster
coldkey = d["neurons"][d["miner_uid"]]["coldkey"]
uids = sorted(n["uid"] for n in d["neurons"] if n["coldkey"] == coldkey)
written = round(time.time() - d["updated_at_unix"], 1)

verdict = "ok"
if age > int(fail_age):
    # The miner is alive and writing but has stopped re-syncing: the exact
    # 2026-09-16 failure. The session watchdog restarts it at 800.
    verdict = "STALE"
elif prev_epoch and prev_epoch != str(epoch):
    # A boundary was crossed since the last sample. The roster must have moved.
    verdict = "SYNC-OK" if prev_roster and roster > int(prev_roster) else "SYNC-MISSED"

print(f"{stamp} {verdict} epoch={epoch} block={block} roster={roster} "
      f"age={age} uids={len(uids)}:{uids} written={written}s")
with open(carry_path, "w") as handle:
    handle.write(f"{epoch}\n{roster}\n")
PY
    # Carry the last sample forward so the next pass can see a boundary.
    if [ -s "$CARRY" ]; then
        PREV_EPOCH=$(sed -n 1p "$CARRY")
        PREV_ROSTER=$(sed -n 2p "$CARRY")
    fi
    sleep "$INTERVAL"
done
