# Continuous bank-update loop — scheduling

Entrypoint: `scoring_backfill/epoch_loop.py` (orchestrator) + `epoch_loop_remote.py` (prod DB/leaderboard queries).

Run once per epoch close (non-dry-run):
    scoring_backfill/run_epoch_loop.sh
or directly:
    python3 scoring_backfill/epoch_loop.py --epochs 3

Idempotent by sequence hash (score_worker skips scored passes; ingest upserts on observation_id).
Dry-run (no scoring, no writes):
    python3 scoring_backfill/epoch_loop.py --dry-run --epochs 3

Cron (runs every 20 min; each invocation is idempotent, so overlap is harmless):
    */20 * * * * /Users/tvgdeena/nova-nanobody/scoring_backfill/run_epoch_loop.sh
