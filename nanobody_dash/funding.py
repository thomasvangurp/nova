"""Build the non-secret daily infrastructure-funding instruction for the Pi.

The Mac already has the two inputs that must not be copied to the signing host:
read-only AWS/Vast billing access and the dashboard's historical economics.
It reduces those inputs to one bounded JSON instruction.  The Pi remains the
authority: it checks freshness, the hard TAO cap, the coldkey reserve and the
one-transfer-per-day ledger before signing anything.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

from .money import subnet_pnl_history


def daily_mining_earnings(connection: sqlite3.Connection, tao_usd: float) -> list[dict]:
    """Per UTC day: mining gross and registration burn, summed over subnets.

    Funding is sized from mining earnings net of the slot registration burn.
    It deliberately does not use the dashboard's portfolio delta, which moves
    with any TAO entering or leaving the coldkey, including the infra
    transfers this target itself authorises.
    """
    by_day: dict[str, dict] = {}
    for row in subnet_pnl_history(connection, tao_usd):
        bucket = by_day.setdefault(row["day"], dict(day=row["day"], gross_tao=0.0,
                                                    registration_tao=0.0))
        bucket["gross_tao"] += row["gross_tao_per_day"]
        bucket["registration_tao"] += row["registration_tao_per_day"]
    return [by_day[day] for day in sorted(by_day)]


SCHEMA_VERSION = 1


def build_target(
    connection: sqlite3.Connection,
    cloud_state: dict,
    tao_usd: float,
    *,
    now: dt.datetime | None = None,
    maximum_tao: float = 1.5,  # matches the Pi cap in /etc/nova/infra-sweep.json (2026-10-03)
    earnings_days: int = 7,
) -> dict:
    """Return today's cost-sized instruction, capped by measured earnings.

    AWS uses the forward operating projection because daily Cost Explorer rows
    arrive incomplete and would systematically underfund the card.  Explicitly
    excluded one-time migration traffic is already absent from that projection.
    Vast contributes only explicitly labelled production capacity.  Research
    remains a real, separately reported cost, but is not swept from mining
    earnings into the production card-funding budget.
    """
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    current = current.astimezone(dt.timezone.utc)
    if tao_usd <= 0:
        raise ValueError("TAO/USD must be positive")
    if maximum_tao <= 0:
        raise ValueError("maximum TAO must be positive")

    aws = cloud_state.get("aws") or {}
    vast = cloud_state.get("vast") or {}
    projected = float(aws.get("projected_month_end_usd") or 0)
    year, month = current.year, current.month
    days_in_month = calendar.monthrange(year, month)[1]
    aws_usd = float(aws.get("operating_usd_per_day")
                    or projected / days_in_month)
    # 2026-10-03 (user decision): the card must cover ALL running Vast spend --
    # the compound boxes and research included, not only labelled production
    # capacity -- so every per-category Vast cost is summed.
    vast_categories = ("sn68_production_usd_per_day", "sn68_compound_usd_per_day",
                       "sn68_research_usd_per_day", "research_usd_per_day",
                       "blueprint_usd_per_day", "bank_usd_per_day",
                       "bank_boxes_account_usd_per_day", "unattributed_usd_per_day")
    if any(vast.get(key) is not None for key in vast_categories):
        vast_usd = sum(float(vast.get(key) or 0) for key in vast_categories)
    else:
        vast_usd = float(vast.get("sn68_usd_per_day") or 0)
    cost_usd = max(0.0, aws_usd + vast_usd)
    cost_tao = cost_usd / tao_usd

    history = daily_mining_earnings(connection, tao_usd)
    complete = [
        row for row in history
        if str(row.get("day")) < current.date().isoformat()
    ][-max(1, int(earnings_days)):]
    earnings = [
        max(0.0, float(row.get("gross_tao") or 0)
            - float(row.get("registration_tao") or 0))
        for row in complete
    ]
    earnings_tao_per_day = sum(earnings) / len(earnings) if earnings else 0.0
    authorized = min(cost_tao, earnings_tao_per_day, float(maximum_tao))
    valid_until = current.replace(hour=23, minute=59, second=59, microsecond=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "target_day": current.date().isoformat(),
        "generated_at": int(current.timestamp()),
        "valid_until": int(valid_until.timestamp()),
        "operating_infra_usd": round(cost_usd, 2),
        "tao_usd": round(float(tao_usd), 4),
        "cost_tao": round(cost_tao, 9),
        "earnings_tao_per_day": round(earnings_tao_per_day, 9),
        "maximum_tao": round(float(maximum_tao), 9),
        "authorized_tao": round(max(0.0, authorized), 9),
        "breakdown_usd": {
            "aws_operating_projection_per_day": round(aws_usd, 2),
            "vast_all_running_per_day": round(vast_usd, 2),
        },
        "policy": ("all running Vast + AWS operating cost, capped by trailing "
                   "net mining earnings and hard daily limit"),
    }


def write_target(path: Path, target: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".next")
    staged.write_text(json.dumps(target, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(staged, 0o600)
    os.replace(staged, path)


def publish_target(
    target: dict,
    *,
    host: str,
    user: str,
    remote_path: str = "/home/thomasg/.nova/infra_funding_target.json",
) -> bool:
    """Atomically send the public instruction; never send a signing secret."""
    endpoint = f"{user}@{host}"
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", endpoint):
        raise ValueError("unsafe funding-target endpoint")
    if not remote_path.startswith("/home/") or ".." in remote_path:
        raise ValueError("unsafe funding-target path")
    program = (
        "import os,pathlib,sys;"
        f"p=pathlib.Path({remote_path!r});"
        "p.parent.mkdir(parents=True,exist_ok=True);"
        "q=p.with_suffix(p.suffix+'.next');"
        "fd=os.open(q,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);"
        "f=os.fdopen(fd,'wb');f.write(sys.stdin.buffer.read());"
        "f.flush();os.fsync(f.fileno());f.close();os.replace(q,p)"
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         endpoint, f"python3 -c {shlex.quote(program)}"],
        input=(json.dumps(target, sort_keys=True) + "\n").encode(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0
