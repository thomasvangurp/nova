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
import datetime as dt
import json
import sqlite3
import subprocess
import time
import urllib.request
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
CREATE TABLE IF NOT EXISTS market_prices (
    asset TEXT PRIMARY KEY, usd REAL NOT NULL, fetched_at REAL NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS infrastructure_samples (
    sampled_at INTEGER NOT NULL, provider TEXT NOT NULL, scope TEXT NOT NULL,
    usd_per_hour REAL NOT NULL, running_instances INTEGER NOT NULL,
    gpus INTEGER NOT NULL,
    PRIMARY KEY (sampled_at, provider, scope)
);
"""

PRICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_prices (
    asset TEXT PRIMARY KEY, usd REAL NOT NULL, fetched_at REAL NOT NULL,
    source TEXT NOT NULL
)
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


def _valid_usd_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if 0 < price < 1_000_000 else None


def _taostats_tao_usd() -> float:
    body = get("price/latest/v1", {"asset": "tao"}, key=_api_key())
    rows = _api_rows(body)
    row = rows[0] if rows else {}
    for field in ("price", "price_usd", "usd"):
        price = _valid_usd_price(row.get(field))
        if price is not None:
            return price
    raise TaostatsError("TAO price response had no valid USD price")


def _coingecko_tao_usd(timeout: int = 20) -> float:
    request = urllib.request.Request(
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=bittensor&vs_currencies=usd",
        headers={"User-Agent": "nova-sn68-dashboard/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read(100_000))
    price = _valid_usd_price((body.get("bittensor") or {}).get("usd"))
    if price is None:
        raise ValueError("CoinGecko response had no valid TAO price")
    return price


def resolve_tao_usd(connection: sqlite3.Connection, explicit: float | None = None,
                    max_age: float = 600.0, now: float | None = None,
                    taostats_fetch=None, coingecko_fetch=None) -> float:
    """Return a non-zero TAO/USD price and persist the last known good value.

    A price API outage must never be converted to ``0``: doing so erases all
    TAO-denominated revenue and registration burn from the rendered dashboard
    while leaving USD infrastructure costs behind. Fresh cached values avoid
    hammering either provider; a stale valid value is safer than fictional zero.
    """
    connection.execute(PRICE_SCHEMA)
    timestamp = time.time() if now is None else float(now)

    def cached():
        return connection.execute(
            "SELECT usd, fetched_at FROM market_prices WHERE asset = 'TAO'"
        ).fetchone()

    def store(price: float, source: str) -> float:
        connection.execute(
            "INSERT OR REPLACE INTO market_prices (asset, usd, fetched_at, source) "
            "VALUES ('TAO', ?, ?, ?)",
            (price, timestamp, source),
        )
        connection.commit()
        return price

    if explicit is not None:
        price = _valid_usd_price(explicit)
        if price is None:
            raise ValueError("explicit TAO/USD price must be positive")
        return store(price, "argument")

    previous = cached()
    if previous and timestamp - float(previous[1]) <= max_age:
        return float(previous[0])

    providers = (
        ("taostats", taostats_fetch or _taostats_tao_usd),
        ("coingecko", coingecko_fetch or _coingecko_tao_usd),
    )
    for source, fetch in providers:
        try:
            price = _valid_usd_price(fetch())
        except Exception:
            continue
        if price is not None:
            return store(price, source)

    if previous and _valid_usd_price(previous[0]) is not None:
        return float(previous[0])
    raise RuntimeError("no valid TAO/USD price is available")


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


def _api_rows(body) -> list[dict]:
    if isinstance(body, list):
        return [row for row in body if isinstance(row, dict)]
    if not isinstance(body, dict):
        return []
    rows = body.get("data") or body.get("results") or []
    return [row for row in rows if isinstance(row, dict)]


def collect_current_balances(connection, coldkey: str,
                             per_page: int = 200) -> int:
    """Persist a current per-subnet stake snapshot from Taostats.

    The API returns one row per delegated hotkey.  Aggregate those rows before
    storing so a subnet's history represents the coldkey's complete alpha
    position, not whichever validator happened to be listed first.  Only
    numeric balance fields reach SQLite; no response body or authorization
    material is logged.
    """
    try:
        key = _api_key()
        rows: list[dict] = []
        for page in range(1, 20):
            body = get(
                "dtao/stake_balance/latest/v1",
                dict(coldkey=coldkey, page=page, limit=per_page,
                     order="netuid_asc"),
                key=key,
            )
            batch = _api_rows(body)
            rows.extend(batch)
            if len(batch) < per_page:
                break
        account_rows = _api_rows(get(
            "account/latest/v1",
            dict(address=coldkey, network="finney", page=1, limit=1),
            key=key,
        ))
    except TaostatsError:
        return 0
    if not rows:
        return 0

    free = _number(account_rows[0].get("balance_free")) if account_rows else 0.0
    grouped: dict[int, dict[str, float]] = {}
    blocks: list[int] = []
    stamps: list[float] = []
    for row in rows:
        try:
            netuid = int(row["netuid"])
            block = int(row["block_number"])
        except (KeyError, TypeError, ValueError):
            continue
        alpha = _number(row.get("balance")) or 0.0
        as_tao = _number(row.get("balance_as_tao")) or 0.0
        bucket = grouped.setdefault(netuid, {"alpha": 0.0, "tao": 0.0})
        bucket["alpha"] += alpha
        bucket["tao"] += as_tao
        blocks.append(block)
        stamp = row.get("timestamp")
        if stamp:
            try:
                stamps.append(dt.datetime.fromisoformat(
                    str(stamp).replace("Z", "+00:00")
                ).timestamp())
            except (TypeError, ValueError, OSError):
                pass
    if not grouped or not blocks:
        return 0
    block = max(blocks)
    sampled_at = max(stamps) if stamps else time.time()
    # The dashboard refreshes every epoch, but money history is daily.  Keep
    # replacing today's live close instead of appending hundreds of tiny,
    # reward-discrete intervals that would make the daily yield chart noisy
    # and inflate any later per-day aggregation.  The block guard preserves a
    # same-day historical backfill whose observations are at least one day old.
    sample_day = dt.datetime.fromtimestamp(
        sampled_at, dt.timezone.utc
    ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    connection.execute(
        "DELETE FROM balances WHERE sampled_at >= ? AND sampled_at < ? "
        "AND block > ?",
        (sample_day, sample_day + 86400, block - 7200),
    )
    values = []
    for netuid, amounts in grouped.items():
        alpha = amounts["alpha"]
        price = amounts["tao"] / alpha if alpha else 0.0
        values.append((block, netuid, alpha, price, free, sampled_at))
    connection.executemany(
        "INSERT OR REPLACE INTO balances "
        "(block, netuid, alpha, alpha_price_tao, free_tao, sampled_at) "
        "VALUES (?,?,?,?,?,?)",
        values,
    )
    connection.commit()
    return len(values)


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


def collect_current_balances_chain(connection: sqlite3.Connection,
                                   pi_host: str = "192.168.1.68",
                                   pi_user: str = "thomasg") -> int:
    """Persist a current per-subnet stake snapshot from the chain, via the Pi.

    Replaces the Taostats ``dtao/stake_balance/latest/v1`` endpoint.  The Pi
    runs the chain-backed collector (bittensor Subtensor) and emits one JSON
    line; only numeric fields reach SQLite.  The delegations/transfers tables
    are NOT populated here (their Taostats endpoints stay down until the quota
    resets), so the daily revenue degrades to the raw balance delta.
    """
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
         f"{pi_user}@{pi_host}",
         "/opt/nova-submit/.pixi/envs/default/bin/python "
         "/opt/nova-submit/tools/chain_money.py"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return 0
    try:
        data = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return 0
    block = int(data["block"])
    free_tao = float(data["free_tao"])
    now = time.time()
    rows = [
        (block, int(netuid), float(pos["alpha"]),
         float(pos["alpha_price_tao"]), free_tao, now)
        for netuid, pos in (data.get("positions") or {}).items()
    ]
    if not rows:
        return 0
    # Replace today's live close (one sample per day) so the daily yield chart
    # is not fragmented into hourly reward-discrete intervals.
    sample_day = dt.datetime.fromtimestamp(
        now, dt.timezone.utc
    ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    connection.execute(
        "DELETE FROM balances WHERE sampled_at >= ? AND sampled_at < ?",
        (sample_day, sample_day + 86400),
    )
    connection.executemany(
        "INSERT OR REPLACE INTO balances "
        "(block, netuid, alpha, alpha_price_tao, free_tao, sampled_at) "
        "VALUES (?,?,?,?,?,?)", rows)
    connection.commit()
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--coldkey", default=DEFAULT_COLDKEY)
    parser.add_argument("--pages", type=int, default=12)
    arguments = parser.parse_args()
    connection = connect(arguments.database)
    report = dict(
        delegations=collect_delegations(connection, arguments.coldkey),
        transfers=collect_transfers(connection, arguments.coldkey),
        balances=collect_current_balances_chain(connection))
    for label, query in (("netuids", "SELECT DISTINCT netuid FROM delegations"),
                         ("rows", "SELECT COUNT(*) FROM delegations")):
        report[label] = [r[0] for r in connection.execute(query)] if label == "netuids" \
            else connection.execute(query).fetchone()[0]
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
