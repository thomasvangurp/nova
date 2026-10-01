#!/usr/bin/env python3
"""Join what we SUBMITTED to what the validator SCORED, one row per uid-epoch.

Why this exists: assessing a pipeline change by watching our best score across
epochs does not work. Between 25242 and 25255 the FIELD median alone moved
148 -> 182, and our own best has sd ~31, so an epoch-to-epoch comparison
cannot resolve any change smaller than the noise. Worse, changes arrive
stacked -- several land between two observations and none is attributable.

This makes every epoch an experiment record instead:

  epoch, uid, config      what produced it
  predicted               selection.score, the model's own estimate
  actual                  the validator's realised score
  rank, field_median      the epoch's difficulty, so it can be divided out

Two things become possible that are impossible from a score series:

* CALIBRATION. predicted vs actual over many rows says whether the selector
  knows what it is doing, independent of whether any epoch went well.
* WITHIN-EPOCH A/B. We hold 8-10 uids. Running config A on half and B on the
  other half puts both in the SAME epoch against the SAME field, so the epoch
  effect -- the dominant noise term -- cancels exactly. A handful of epochs
  then resolves differences a time series never could.

Nanobody scores are LOWER-is-better; molecule scores are higher-is-better.
Both are recorded raw, with the direction named, so no caller has to guess.
"""
import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path

LEADERBOARD = ("https://compound-api.metanova-labs.ai/api/competitions/"
               "leaderboard/{epoch}/molecules?limit=10000&molecule_detail=full")
SCHEMA = """
CREATE TABLE IF NOT EXISTS observations(
  epoch INT, uid INT, hotkey TEXT, config TEXT,
  predicted REAL, nano_actual REAL, mol_actual REAL,
  nano_rank INT, mol_rank INT, field_size INT,
  nano_median REAL, mol_winner REAL, molecules INT,
  PRIMARY KEY(epoch, uid))
"""


def leaderboard(epoch: int):
    with urllib.request.urlopen(LEADERBOARD.format(epoch=epoch), timeout=90) as r:
        return json.loads(r.read().decode()).get("data") or []


def ingest(connection, epoch, rows, ours, predictions, config):
    nano, mol = [], []
    for r in rows:
        v = r.get("nanobody_final_score")
        if v not in (None, "") and float(v) > -900:
            nano.append((int(r["uid"]), float(v)))
        m = r.get("molecules")
        if isinstance(m, str):
            try: m = json.loads(m)
            except Exception: m = []
        f = r.get("final_score")
        if len(m or []) and f not in (None, "") and float(f) > -900:
            mol.append((int(r["uid"]), float(f), len(m)))
    nano.sort(key=lambda x: x[1])            # lower is better
    mol.sort(key=lambda x: -x[1])            # higher is better
    nano_order = [u for u, _ in nano]
    mol_order = [u for u, _, _ in mol]
    nano_median = (sorted(v for _, v in nano)[len(nano) // 2]) if nano else None
    mol_winner = mol[0][1] if mol else None

    written = 0
    for r in rows:
        if str(r.get("hotkey")) not in ours:
            continue
        uid = int(r["uid"])
        nv = dict(nano).get(uid)
        mv = next(((s, n) for u, s, n in mol if u == uid), None)
        connection.execute(
            "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (epoch, uid, str(r.get("hotkey")), config,
             predictions.get(uid), nv, mv[0] if mv else None,
             (1 + nano_order.index(uid)) if uid in nano_order else None,
             (1 + mol_order.index(uid)) if uid in mol_order else None,
             len(nano), nano_median, mol_winner, mv[1] if mv else None))
        written += 1
    connection.commit()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("epochs", nargs="+", type=int)
    parser.add_argument("--ledger", default="data/score_ledger.sqlite")
    parser.add_argument("--hotkeys", required=True,
                        help="JSON with an 'addresses' list. Refresh it: a "
                             "stale set silently reads our own newest uids as "
                             "another operator's.")
    parser.add_argument("--artifact", default=None,
                        help="nanobody_result.json for this epoch, to record "
                             "each uid's PREDICTED selection score")
    parser.add_argument("--config", default="unlabelled",
                        help="what produced these submissions; the whole point "
                             "is being able to compare two of them")
    arguments = parser.parse_args()

    text = Path(arguments.hotkeys).read_text()
    ours = set(json.loads(text[text.index("{"):])["addresses"])
    predictions = {}
    if arguments.artifact and Path(arguments.artifact).is_file():
        art = json.loads(Path(arguments.artifact).read_text())
        for entry in art.get("submissions") or []:
            sel = entry.get("selection") or {}
            if sel.get("score") is not None:
                predictions[int(entry["uid"])] = float(sel["score"])

    Path(arguments.ledger).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(arguments.ledger)
    connection.execute(SCHEMA)
    total = 0
    for epoch in arguments.epochs:
        try:
            rows = leaderboard(epoch)
        except Exception as error:
            print(json.dumps(dict(epoch=epoch, error=str(error)[:120])))
            continue
        if not rows:
            print(json.dumps(dict(epoch=epoch, rows=0, note="not scored yet")))
            continue
        total += ingest(connection, epoch, rows, ours, predictions, arguments.config)
    held = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    epochs = connection.execute(
        "SELECT COUNT(DISTINCT epoch) FROM observations").fetchone()[0]
    connection.close()
    print(json.dumps(dict(written=total, ledger_rows=held, ledger_epochs=epochs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
