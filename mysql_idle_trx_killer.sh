#!/usr/bin/env bash
# mysql_idle_trx_killer.sh
#
# Reaps leaked InnoDB transactions on the sxlab-mysql container that hold row
# locks while their connection sits in Sleep -- i.e. the client opened a
# transaction (autocommit=0), acquired locks, and then went idle instead of
# committing/closing.  Such a transaction blocks every other writer on the
# rows it locked (Lock wait timeout exceeded / Lost connection).
#
# Conservative by design:
#   * only touches command='Sleep' connections (never an actively running
#     query, so the live producer's in-flight SELECT/DELETE/INSERT work is
#     never disturbed);
#   * only when trx_rows_locked > 0 (plain consistent reads hold no locks);
#   * only when trx_rows_modified = 0 -- the leak signature is an
#     ``INSERT ... ON DUPLICATE KEY UPDATE`` that no-ops on already-present
#     rows: it takes exclusive locks but modifies nothing.  A legitimate
#     writer (e.g. the producer's rebuild_nearest_edges DELETE/INSERT) has
#     rows_modified > 0 while it holds locks and is therefore never killed;
#   * only when the transaction has been open longer than IDLE_MINUTES.
#
# Usage:
#   mysql_idle_trx_killer.sh [idle_minutes]   # default 2
#   DRY_RUN=1 mysql_idle_trx_killer.sh        # report only, do not KILL
#
# Installed by cron on the Hetzner host (sxbot): runs every minute.

set -u

IDLE_MINUTES="${1:-2}"
IDLE_SECONDS=$(( IDLE_MINUTES * 60 ))
DRY_RUN="${DRY_RUN:-0}"

LOCK_FILE="/tmp/mysql_idle_trx_killer.lock"
LOG="${HOME:-/home/sxbot}/mysql_idle_trx_killer.log"
MYSQL="docker exec sxlab-mysql mysql -uroot"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Mutex so overlapping cron invocations can't double-KILL.
exec 200>"$LOCK_FILE"
flock -n 200 || { log "another instance already running; skip"; exit 0; }

candidates=$($MYSQL -N -e "
SELECT t.trx_mysql_thread_id,
       ROUND(TIMESTAMPDIFF(SECOND, t.trx_started, NOW())/60, 1),
       t.trx_rows_locked,
       t.trx_rows_modified,
       p.host,
       p.user,
       LEFT(COALESCE(t.trx_query, ''), 60)
FROM information_schema.innodb_trx t
JOIN information_schema.processlist p ON p.id = t.trx_mysql_thread_id
WHERE p.command = 'Sleep'
  AND t.trx_rows_locked > 0
  AND t.trx_rows_modified = 0
  AND TIMESTAMPDIFF(SECOND, t.trx_started, NOW()) > ${IDLE_SECONDS}
ORDER BY t.trx_started
" 2>/dev/null)

if [ -z "$candidates" ]; then
    exit 0
fi

while IFS=$'\t' read -r tid age_min rows_locked rows_modified host user trx_query; do
    [ -z "${tid:-}" ] && continue
    log "LEAKED-TRX thread=$tid age_min=$age_min rows_locked=$rows_locked rows_modified=$rows_modified host=$host user=$user trx_query='$trx_query'"
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY_RUN: would KILL $tid"
        continue
    fi
    result=$($MYSQL -e "KILL $tid" 2>&1)
    log "KILL $tid -> $result"
done <<< "$candidates"
