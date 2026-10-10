#!/usr/bin/env python3
"""Relabel the mixed-scale dashboard multipass rows (ARCHITECTURE_REDESIGN P4).

Until 2026-10-09 ``tools/multipass_all_rivals.py`` wrote every pass row as
``source='public_multipass_rivals'``, ``score_kind='local_validator_replay_rank_sum'``.
That is the producer's ~200-scale label. Over one day the dashboard box ran
three formulas under it, so the rows still under that label (no raw metrics
except 25536-25538) hold three scales. Each regime was verified on
2026-10-09/10 by recomputing all three formulas from raw metrics
(``metrics_json``, or the passes still on the dashboard disk). The
registry, ``nanobody_opt/observation_labels.py``, has the evidence.

| epochs              | formula   | written (UTC, 2026-10-09)               | new label                         |
|---------------------|-----------|-----------------------------------------|-----------------------------------|
| 25559-25610         | algo2     | before the 14:01:40Z category3 deploy   | legacy_multipass_algo2_pass       |
| 25536-25538, 25611-2| category3 | 14:01:40Z-17:19:41Z (category3 build)   | legacy_multipass_category3_pass   |
| 25613-25618         | sum10     | after the 17:25:15Z sum10 deploy        | multipass_sum10_pass (current)    |

The relabel only changes ``score_kind``; ``score`` is never touched. It adds
provenance (``relabeled_from``, ``formula``, evidence) via JSON_MERGE_PATCH,
so nothing is deleted or reinterpreted. Every epoch is re-checked against its
formula's fingerprint before any row moves: the mean must sit in the
formula's band, category3/sum10 values must be integers, and algo2 must have
fractional values. An epoch that does not match is refused, not relabelled.
Rows of ``public_multipass_rivals`` under the old label in any other epoch
are reported and left alone.

Idempotent: it only touches rows still labelled
``local_validator_replay_rank_sum``. ``tools/backfill_multipass_metrics.py``
can later upgrade any relabelled epoch whose raw passes are still on disk to
sum10-with-metrics. It records the legacy label as ``original_score_kind``.

Usage (any host with the central-DB creds, e.g. the dashboard box)::

    /root/bgen_python.sh /root/relabel_multipass_legacy.py            # dry run
    /root/bgen_python.sh /root/relabel_multipass_legacy.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _root in (HERE, HERE.parent):
    if (_root / "nanobody_opt").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from nanobody_opt import observation_labels  # noqa: E402

SOURCE = "public_multipass_rivals"
OLD_KIND = "local_validator_replay_rank_sum"
WRITER = observation_labels.RELABEL_WRITER

# formula -> (new score_kind, epochs, (mean_lo, mean_hi), integer_only, evidence)
REGIMES = {
    "algo2": (
        "legacy_multipass_algo2_pass", set(range(25559, 25611)), (25_000, 48_000), False,
        "deployed before 2026-10-09T14:01:40Z (box backup "
        "multipass_all_rivals.py.bak.1791554500 = algo2 build); stored == algo2 "
        "recomputed from disk passes, rho=1.000 (25594, 25599, 25604, 25610); "
        "fractional average-rank values",
    ),
    "category3": (
        "legacy_multipass_category3_pass", {25536, 25537, 25538, 25611, 25612},
        (45_000, 60_000), True,
        "written 2026-10-09T14:01:40Z-17:19:41Z while the category3 build "
        "(box backup .bak.1791566381, commit b310b14) was live; stored == "
        "category3 recomputed, rho=1.000 (25536-25538 from metrics_json, "
        "25611-25612 from disk passes)",
    ),
    "sum10": (
        "multipass_sum10_pass", set(range(25613, 25619)), (100_000, 140_000), True,
        "written after the 2026-10-09T17:25:15Z sum10 deploy (commit 62c6e86); "
        "stored == sum10 recomputed from disk passes, rho=1.000 (25613-25617; "
        "25618's disk passes are from a later rescore)",
    ),
}
FIELD_NOTE = {
    "algo2": "hf_archive.csv (dashboard box) or the epoch's producer archive.csv "
             "(master); snapshot unrecorded",
    "category3": "/root/hf_archive.csv snapshot at write time (unrecorded)",
    "sum10": "hf_archive_full: /root/hf_archive.csv snapshot at write time "
             "(unrecorded, no metrics)",
}


def epoch_stats(connection) -> dict[int, dict]:
    rows = connection.execute(
        "SELECT epoch, COUNT(*), AVG(score), MIN(score), MAX(score), "
        "SUM(score <> FLOOR(score)) FROM observations "
        "WHERE source=%s AND score_kind=%s GROUP BY epoch",
        (SOURCE, OLD_KIND),
    ).fetchall()
    return {(int(r[0]) if r[0] is not None else -1): {
        "n": int(r[1]), "mean": float(r[2]), "min": float(r[3]),
        "max": float(r[4]), "nonint": int(r[5] or 0)} for r in rows}


def fingerprint_ok(formula: str, stats: dict) -> tuple[bool, str]:
    _, _, (lo, hi), integer_only, _ = REGIMES[formula]
    if not lo <= stats["mean"] <= hi:
        return False, f"mean {stats['mean']:.0f} outside {formula} band {lo}-{hi}"
    if integer_only and stats["nonint"]:
        return False, f"{stats['nonint']} fractional values; {formula} is integer-only"
    if not integer_only and stats["n"] >= 20 and stats["nonint"] / stats["n"] < 0.2:
        return False, (f"only {stats['nonint']}/{stats['n']} fractional values; "
                       "algo2 average ranks are ~45% fractional")
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args()

    from nanobody_opt import db

    with db.connect() as connection:
        stats = epoch_stats(connection)
    plan: list[tuple[int, str]] = []
    refused = unmapped = 0
    for epoch in sorted(stats):
        formula = next((f for f, spec in REGIMES.items() if epoch in spec[1]), None)
        s = stats[epoch]
        if formula is None:
            unmapped += s["n"]
            print(f"epoch {epoch}: n={s['n']} mean={s['mean']:.0f} UNMAPPED, left alone")
            continue
        ok, why = fingerprint_ok(formula, s)
        new_kind = REGIMES[formula][0]
        print(f"epoch {epoch}: n={s['n']} mean={s['mean']:.0f} nonint={s['nonint']} "
              f"-> {formula} -> {new_kind}: {'OK' if ok else 'REFUSED ' + why}")
        if ok:
            plan.append((epoch, formula))
        else:
            refused += s["n"]

    # The registry must accept these writes (writer + formula + source).
    observation_labels.check_observations(
        {"score_kind": REGIMES[f][0], "source": SOURCE, "writer": WRITER, "formula": f}
        for f in {f for _, f in plan})

    total = sum(stats[e]["n"] for e, _ in plan)
    print(f"PLAN rows={total} epochs={len(plan)} refused_rows={refused} "
          f"unmapped_rows={unmapped} apply={args.apply}", flush=True)
    if not args.apply:
        return 0

    now = int(time.time())
    moved = 0
    for epoch, formula in plan:
        new_kind, _, _, _, evidence = REGIMES[formula]
        patch = json.dumps({
            "relabeled_from": OLD_KIND, "relabeled_at": now, "writer": WRITER,
            "original_writer": "multipass_all_rivals (pre-registry build)",
            "formula": formula, "formula_evidence": evidence,
            "passes": 1, "field": FIELD_NOTE[formula],
        })
        with db.connect() as connection:
            ids = [str(r[0]) for r in connection.execute(
                "SELECT observation_id FROM observations "
                "WHERE source=%s AND score_kind=%s AND epoch=%s",
                (SOURCE, OLD_KIND, epoch)).fetchall()]
        for start in range(0, len(ids), args.batch):
            chunk = ids[start:start + args.batch]
            marks = ",".join(["%s"] * len(chunk))
            for attempt in range(1, 6):
                try:
                    with db.connect() as connection:
                        connection.execute(
                            "UPDATE observations SET score_kind=%s, provenance_json="
                            "JSON_MERGE_PATCH(COALESCE(NULLIF(provenance_json,''),'{}'), %s) "
                            f"WHERE observation_id IN ({marks}) "
                            "AND source=%s AND score_kind=%s",
                            (new_kind, patch, *chunk, SOURCE, OLD_KIND))
                        connection.commit()
                    break
                except Exception as error:  # noqa: BLE001
                    if not db.is_deadlock(error) or attempt == 5:
                        raise
                    print(f"  epoch {epoch} batch {start}: lock error, retry {attempt}",
                          flush=True)
                    time.sleep(2 * attempt)
            moved += len(chunk)
        print(f"epoch {epoch}: relabelled {len(ids)} -> {new_kind}", flush=True)

    with db.connect() as connection:
        left = epoch_stats(connection)
    print("RELABEL_DONE " + json.dumps({
        "moved": moved, "left_under_old_label": sum(s["n"] for s in left.values()),
        "left_epochs": sorted(left)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
