#!/usr/bin/env python3
"""Import the 20x HF-field rank-sums into MySQL as ``hf_field_20x_rank_sum``.

Reads the three backfill JSONs produced on the slave, computes each sequence's
20x mean rank-sum (one number on the shared 21k-58k scale, lower = better),
and upserts ONE observation per sequence via ``update_observation_store``:

    observation_id    = "hf20x:{seq}"
    score_kind        = "hf_field_20x_rank_sum"
    observation_count = 20
    score             = mean rank-sum over the 20 passes
    source            = our_top50_20x | public_multipass_rivals | hf_archive_top200

Sources:
  our_top50_20x           <- missing18_hf_field.json + top50_32_hf_field.json
                             (each ``rank_sums`` dict already holds 20x means)
  hf_archive_top200       <- topup_observations.json entries with
                             source=="hf_archive_top200" (passes 1-20 -> mean)
  public_multipass_rivals <- topup_observations.json entries with
                             source=="public_multipass_rivals" (passes 11-20)
                             PLUS the DB passes 1-10 (source
                             public_multipass_rivals, score_kind
                             local_validator_replay_rank_sum) -> mean over 20.

Idempotent: re-running upserts the same ``hf20x:{seq}`` observation ids. Any
JSON that has not appeared yet is skipped (the top-up / 32-scoring jobs are
still running); re-run after they land. With ``--wait`` the script polls for
the files instead of skipping them.

Run on the slave (over the tailscale MySQL relay)::

    SN68_MYSQL_HOST=127.0.0.1 SN68_MYSQL_PORT=3306 \
      /root/.pixi/envs/bgen/bin/python /root/import_hf_field_20x.py --wait
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

TARGET = "P20809"
SCORE_KIND = "hf_field_20x_rank_sum"
N_PASSES = 20

MISSING18 = Path("/root/missing18_hf_field.json")
TOP50_32 = Path("/root/top50_32_hf_field.json")
TOPUP = Path("/root/topup_observations.json")

# DB passes 1-10 for rivals (the top-up adds passes 11-20 to the JSON only).
RIVAL_DB_SQL = """
    SELECT n.sequence, SUM(o.score), COUNT(*)
    FROM observations o JOIN nodes n ON o.node_id = n.id
    WHERE n.target = %s
      AND o.source = 'public_multipass_rivals'
      AND o.score_kind = 'local_validator_replay_rank_sum'
    GROUP BY n.sequence
"""


def _wait_for_files(paths: list[Path], timeout: float) -> None:
    """Block until every path exists (poll every 60s) or timeout elapses."""
    deadline = time.monotonic() + timeout
    pending = [p for p in paths if not p.is_file()]
    while pending and time.monotonic() < deadline:
        time.sleep(60)
        pending = [p for p in pending if not p.is_file()]
        if pending:
            print(
                f"  waiting for {[p.name for p in pending]} ... "
                f"{max(0, int(deadline - time.monotonic()))}s left", flush=True)


def _emit(obs: list[dict], source: str, means: dict[str, float]) -> None:
    for seq, mean in means.items():
        if not seq or not mean or not (mean == mean):  # skip empty / NaN
            continue
        obs.append({
            "observation_id": f"hf20x:{seq}",
            "sequence": seq,
            "target": TARGET,
            "source": source,
            "score_kind": SCORE_KIND,
            "score": float(mean),
            "observation_count": N_PASSES,
        })


def _read_our_top50_20x() -> dict[str, float]:
    """Merge the two 'our top-50' JSONs (their rank_sums are already 20x means)."""
    means: dict[str, float] = {}
    for path in (MISSING18, TOP50_32):
        if not path.is_file():
            print(f"  skip {path.name} (not present)", flush=True)
            continue
        blob = json.loads(path.read_text())
        for seq, v in (blob.get("rank_sums") or {}).items():
            means[str(seq).strip().upper()] = float(v)
    return means


def _read_topup_json() -> list[dict]:
    """topup_observations.json is a LIST of per-pass observation dicts."""
    if not TOPUP.is_file():
        print(f"  skip {TOPUP.name} (not present)", flush=True)
        return []
    return json.loads(TOPUP.read_text())


def _retry(fn, *args, retries=6, delay=12, **kwargs):
    """Retry a DB operation on the transient relay 'Lost connection' errors.

    The slave reaches Hetzner MySQL over a tailscale relay (127.0.0.1:3306)
    that occasionally drops the initial/long-lived connection. Retry rather
    than spin: every operation here is an idempotent upsert or a read.
    """
    from MySQLdb import OperationalError

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except OperationalError as exc:
            print(f"  DB retry {attempt}/{retries}: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _read_rival_db_passes() -> dict[str, tuple[float, int]]:
    from nanobody_opt import db

    def _run():
        with db.connect() as conn:
            rows = conn.execute(RIVAL_DB_SQL, (TARGET,)).fetchall()
        return {str(r[0]).strip().upper(): (float(r[1]), int(r[2])) for r in rows}

    return _retry(_run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", action="store_true",
                    help="poll for the JSONs before importing")
    ap.add_argument("--timeout", type=float, default=3 * 3600,
                    help="max seconds to wait for the JSONs (default 3h)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the observations without writing")
    args = ap.parse_args()

    if args.wait:
        _wait_for_files([MISSING18, TOP50_32, TOPUP], args.timeout)

    sys.path.insert(0, "/root")
    from nanobody_opt.mutation_network import update_observation_store

    obs: list[dict] = []
    by_source: dict[str, list[float]] = defaultdict(list)

    # 1) our top-50 (20x means straight from the JSONs).
    ours = _read_our_top50_20x()
    _emit(obs, "our_top50_20x", ours)
    for v in ours.values():
        by_source["our_top50_20x"].append(v)

    # 2) topup JSON: HF top-200 (20 passes) + rivals (passes 11-20).
    hf_passes: dict[str, list[float]] = defaultdict(list)
    rival_passes: dict[str, list[float]] = defaultdict(list)
    for item in _read_topup_json():
        src = str(item.get("source") or "")
        seq = str(item.get("sequence") or "").strip().upper()
        score = float(item.get("score"))
        if src == "hf_archive_top200":
            hf_passes[seq].append(score)
        elif src == "public_multipass_rivals":
            rival_passes[seq].append(score)

    hf_means = {s: sum(v) / len(v) for s, v in hf_passes.items() if len(v) == N_PASSES}
    _emit(obs, "hf_archive_top200", hf_means)
    for v in hf_means.values():
        by_source["hf_archive_top200"].append(v)

    # 3) rivals: DB passes 1-10 + JSON passes 11-20 -> mean over 20.
    rival_means: dict[str, float] = {}
    db_passes = _read_rival_db_passes() if rival_passes else {}
    for seq, json_scores in rival_passes.items():
        if seq not in db_passes:
            continue
        db_sum, db_cnt = db_passes[seq]
        total = db_cnt + len(json_scores)
        if total != N_PASSES:
            continue  # incomplete rival (no full 20x) -> skip
        rival_means[seq] = (db_sum + sum(json_scores)) / total
    _emit(obs, "public_multipass_rivals", rival_means)
    for v in rival_means.values():
        by_source["public_multipass_rivals"].append(v)

    print(
        "IMPORT_SUMMARY "
        f"total={len(obs)} "
        f"our_top50_20x={len(ours)} "
        f"hf_archive_top200={len(hf_means)} "
        f"public_multipass_rivals={len(rival_means)}", flush=True)
    for src, vals in sorted(by_source.items()):
        if vals:
            print(f"  {src}: n={len(vals)} min={min(vals):.1f} "
                  f"max={max(vals):.1f} mean={sum(vals)/len(vals):.1f}", flush=True)

    if args.dry_run:
        print(f"DRY_RUN (not writing {len(obs)} observations)")
        return 0
    written = _retry(
        update_observation_store, Path("/root/.sn68_mysql_creds"), obs,
        commit_every=1000, commit_interval=5.0)
    print(f"IMPORT_DONE written={written}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
