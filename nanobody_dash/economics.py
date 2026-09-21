"""Where the money comes from and where it goes, per subnet.

Three sources, because no single one is sufficient:

* ``delegation/v1``  every DELEGATE (TAO -> alpha) and UNDELEGATE (alpha -> TAO),
  carrying netuid, alpha, TAO and the price at the time.
* ``transfer/v1``    plain TAO transfers in and out of the coldkey.
* the archive node   the alpha BALANCE per subnet per day.

Mining yield cannot be read from any of them alone.  Rewards arrive as alpha
without an extrinsic, so what we earned is what the balance gained that was not
bought:

    earned_alpha = d(balance) - delegated_alpha + undelegated_alpha

Reading the balance alone would count alpha we paid for as profit, which is the
mistake that made a rising portfolio look like a burn-down.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from . import DEFAULT_COLDKEY
from .taostats import TaostatsError, _api_key, get

RAO = 1e9

SCHEMA = """
CREATE TABLE IF NOT EXISTS delegations (
    id TEXT PRIMARY KEY, timestamp TEXT, block INTEGER, netuid INTEGER,
    action TEXT, alpha REAL, tao REAL, alpha_price_tao REAL,
    -- A DELEGATE with is_transfer set is alpha ARRIVING as a stake
    -- transfer, i.e. mining rewards being swept in. Without this column a
    -- reward is indistinguishable from a purchase and the subnet reads as
    -- unprofitable.
    is_transfer INTEGER
);
CREATE INDEX IF NOT EXISTS deleg_netuid ON delegations(netuid, timestamp);
CREATE TABLE IF NOT EXISTS transfers (
    id TEXT PRIMARY KEY, timestamp TEXT, block INTEGER,
    tao REAL, fee REAL, direction TEXT, counterparty TEXT
);
CREATE TABLE IF NOT EXISTS balances (
    block INTEGER NOT NULL, netuid INTEGER NOT NULL,
    alpha REAL, alpha_price_tao REAL, free_tao REAL, sampled_at REAL,
    PRIMARY KEY (block, netuid)
);
CREATE TABLE IF NOT EXISTS costs (
    day TEXT NOT NULL, item TEXT NOT NULL, usd REAL, note TEXT,
    PRIMARY KEY (day, item)
);
CREATE TABLE IF NOT EXISTS deploys (
    sha TEXT PRIMARY KEY, timestamp REAL, epoch INTEGER, subject TEXT, area TEXT
);
"""


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    return connection


def _number(value, scale=RAO):
    try:
        return float(value) / scale
    except (TypeError, ValueError):
        return None


def collect_delegations(connection, coldkey: str, pages: int = 12,
                        per_page: int = 200, pause: float = 1.5) -> int:
    key = _api_key()
    kept = 0
    for page in range(1, pages + 1):
        try:
            body = get("delegation/v1", dict(nominator=coldkey, limit=per_page,
                                             page=page), key=key)
        except TaostatsError:
            break                                   # rate limit or end of data
        rows = body.get("data") or []
        if not rows:
            break
        connection.executemany(
            "INSERT OR REPLACE INTO delegations "
            "(id, timestamp, block, netuid, action, alpha, tao, alpha_price_tao,"
            " is_transfer) VALUES (?,?,?,?,?,?,?,?,?)",
            [(str(r.get("id")), r.get("timestamp"), r.get("block_number"),
              r.get("netuid"), r.get("action"), _number(r.get("alpha")),
              _number(r.get("amount")), _number(r.get("alpha_price_in_tao"), 1),
              int(bool(r.get("is_transfer"))))
             for r in rows])
        connection.commit()
        kept += len(rows)
        if len(rows) < per_page:
            break
        time.sleep(pause)
    return kept


def collect_transfers(connection, coldkey: str, pages: int = 4,
                      per_page: int = 200, pause: float = 1.5) -> int:
    key = _api_key()
    kept = 0
    for page in range(1, pages + 1):
        try:
            body = get("transfer/v1", dict(address=coldkey, limit=per_page,
                                           page=page), key=key)
        except TaostatsError:
            break
        rows = body.get("data") or []
        if not rows:
            break
        values = []
        for r in rows:
            sender = r.get("from")
            sender = sender.get("ss58") if isinstance(sender, dict) else sender
            recipient = r.get("to")
            recipient = recipient.get("ss58") if isinstance(recipient, dict) else recipient
            outgoing = sender == coldkey
            values.append((str(r.get("id")), r.get("timestamp"), r.get("block_number"),
                           _number(r.get("amount")), _number(r.get("fee")),
                           "out" if outgoing else "in",
                           recipient if outgoing else sender))
        connection.executemany(
            "INSERT OR REPLACE INTO transfers "
            "(id, timestamp, block, tao, fee, direction, counterparty) "
            "VALUES (?,?,?,?,?,?,?)", values)
        connection.commit()
        kept += len(rows)
        if len(rows) < per_page:
            break
        time.sleep(pause)
    return kept


def record_balances(connection, samples) -> int:
    """Store archive-node balance samples: [{block, free, positions:{netuid:(alpha,price)}}]."""
    now = time.time()
    rows = []
    for sample in samples:
        for netuid, (alpha, price) in sample["positions"].items():
            rows.append((sample["block"], int(netuid), alpha, price,
                         sample.get("free"), now))
    connection.executemany(
        "INSERT OR REPLACE INTO balances "
        "(block, netuid, alpha, alpha_price_tao, free_tao, sampled_at) "
        "VALUES (?,?,?,?,?,?)", rows)
    connection.commit()
    return len(rows)


def record_costs(connection, entries) -> int:
    """entries: [{day: 'YYYY-MM-DD', item: 'vast.ai', usd: 12.3, note: ''}]"""
    connection.executemany(
        "INSERT OR REPLACE INTO costs (day, item, usd, note) VALUES (?,?,?,?)",
        [(e["day"], e["item"], float(e["usd"]), e.get("note", "")) for e in entries])
    connection.commit()
    return len(entries)


def record_deploys(connection, entries) -> int:
    connection.executemany(
        "INSERT OR REPLACE INTO deploys (sha, timestamp, epoch, subject, area) "
        "VALUES (?,?,?,?,?)",
        [(e["sha"], e["timestamp"], e.get("epoch"), e["subject"], e.get("area", ""))
         for e in entries])
    connection.commit()
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--coldkey", default=DEFAULT_COLDKEY)
    parser.add_argument("--pages", type=int, default=12)
    arguments = parser.parse_args()
    connection = connect(arguments.database)
    report = dict(
        delegations=collect_delegations(connection, arguments.coldkey, arguments.pages),
        transfers=collect_transfers(connection, arguments.coldkey))
    for label, query in (("netuids", "SELECT DISTINCT netuid FROM delegations"),
                         ("rows", "SELECT COUNT(*) FROM delegations")):
        report[label] = [r[0] for r in connection.execute(query)] if label == "netuids" \
            else connection.execute(query).fetchone()[0]
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
