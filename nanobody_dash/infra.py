"""Infrastructure cost: what the GPUs and the SN2 box actually cost per day.

vast.ai is read live -- every running instance reports ``dph_total``, so the
burn rate is exact rather than estimated, and the invoice endpoint gives what
has already been charged.  AWS is a configured figure: the Ohio box for SN2 is
a steady monthly cost and the local AWS profile is not always authenticated, so
a stale credential must not be allowed to silently zero the cost line.

The API key is read from the Keychain at call time and never leaves this
process.  Errors report a status code only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = "https://console.vast.ai/api/v0"
KEYCHAIN_CANDIDATES = ("vast.ai", "vastai-api")


class InfraError(RuntimeError):
    """Status codes only -- never a key, header or raw body."""


def _keychain(service: str) -> str | None:
    try:
        done = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return done.stdout.strip() or None


def vast_key(services=KEYCHAIN_CANDIDATES) -> tuple[str, str]:
    """Return (key, service_name). Tries each candidate service in turn."""
    for service in services:
        key = _keychain(service)
        if key:
            return key, service
    raise InfraError(f"no Keychain item among {services}")


def vast_get(path: str, key: str, timeout: int = 45):
    request = urllib.request.Request(
        f"{ROOT}/{path.lstrip('/')}",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise InfraError(f"vast {path} returned HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        raise InfraError(f"vast {path} unreachable") from None


def running_cost(key: str) -> dict:
    """Live burn rate from every instance the account is paying for."""
    body = vast_get("instances/", key)
    instances = body.get("instances") if isinstance(body, dict) else body
    rows, total = [], 0.0
    for machine in instances or []:
        state = machine.get("actual_status") or machine.get("cur_state")
        dph = float(machine.get("dph_total") or 0)
        if state in ("running", "loading"):
            total += dph
        rows.append(dict(
            id=machine.get("id"), state=state, dph=round(dph, 4),
            gpu=machine.get("gpu_name"), num_gpus=machine.get("num_gpus"),
            label=machine.get("label"),
            host=f"{machine.get('ssh_host')}:{machine.get('ssh_port')}"))
    return dict(usd_per_hour=round(total, 4), usd_per_day=round(total * 24, 2),
                instances=rows)


def invoices(key: str, days: int = 14) -> list[dict]:
    """Charges already billed, bucketed per day."""
    try:
        body = vast_get("users/current/invoices/", key)
    except InfraError:
        return []
    rows = body.get("invoices") if isinstance(body, dict) else body
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    daily: dict[str, float] = {}
    for row in rows or []:
        stamp = row.get("when") or row.get("timestamp")
        amount = row.get("amount")
        if stamp is None or amount is None:
            continue
        try:
            when = dt.datetime.fromtimestamp(float(stamp), dt.timezone.utc)
        except (TypeError, ValueError, OSError):
            continue
        if when < since:
            continue
        # vast records charges as negative amounts against the balance
        daily[when.strftime("%Y-%m-%d")] = round(
            daily.get(when.strftime("%Y-%m-%d"), 0.0) + abs(float(amount)), 4)
    return [dict(day=d, item="vast.ai", usd=v, note="invoiced")
            for d, v in sorted(daily.items())]


def load_static_costs(path: Path) -> list[dict]:
    """Costs that cannot be read from an API, e.g. the SN2 box in Ohio."""
    if not path.is_file():
        return []
    config = json.loads(path.read_text())
    out = []
    today = dt.date.today()
    for item, spec in (config.get("monthly_usd") or {}).items():
        per_day = float(spec) / 30.0
        for back in range(int(config.get("days", 14))):
            day = (today - dt.timedelta(days=back)).isoformat()
            out.append(dict(day=day, item=item, usd=round(per_day, 4),
                            note="from costs.json monthly figure"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--costs", type=Path, default=Path("config/costs.json"))
    parser.add_argument("--days", type=int, default=14)
    arguments = parser.parse_args()

    from .economics import connect, record_costs
    connection = connect(arguments.database)
    report: dict = {}
    try:
        key, service = vast_key()
        report["keychain_service"] = service
        live = running_cost(key)
        report["vast_usd_per_day_now"] = live["usd_per_day"]
        report["vast_instances"] = len(live["instances"])
        report["vast_running"] = sum(1 for i in live["instances"]
                                     if i["state"] in ("running", "loading"))
        billed = invoices(key, arguments.days)
        report["vast_invoice_days"] = len(billed)
        today = dt.date.today().isoformat()
        rows = billed or [dict(day=today, item="vast.ai",
                               usd=live["usd_per_day"], note="live burn rate")]
        report["cost_rows"] = record_costs(connection, rows)
        (Path("data")).mkdir(exist_ok=True)
        Path("data/vast_instances.json").write_text(json.dumps(live, indent=1))
    except InfraError as error:
        report["vast_error"] = str(error)
    static = load_static_costs(arguments.costs)
    if static:
        report["static_cost_rows"] = record_costs(connection, static)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#: Which project each vast box serves, matched against its label.  Cost is
#: meaningless in aggregate: the nanobody producer is one box, the reaction-5
#: bank is four, and anything else is neither and must not be charged against
#: either subnet's yield.
PROJECTS = (
    ("nanobody", ("nova-nanobody",)),
    ("compound", ("nova-rxn", "nova-nim19", "compound")),
    ("blueprint", ("blueprint",)),
)


def classify(label: str | None) -> str:
    text = (label or "").lower()
    for project, prefixes in PROJECTS:
        if any(prefix in text for prefix in prefixes):
            return project
    return "unattributed"


def cost_by_project(live: dict) -> dict:
    """Split the live burn rate by project, running instances only."""
    out: dict[str, dict] = {}
    for machine in live.get("instances", []):
        if machine.get("state") not in ("running", "loading"):
            continue
        project = classify(machine.get("label"))
        bucket = out.setdefault(project, dict(usd_per_hour=0.0, boxes=0, gpus=0,
                                              labels=[]))
        bucket["usd_per_hour"] += float(machine.get("dph") or 0)
        bucket["boxes"] += 1
        bucket["gpus"] += int(machine.get("num_gpus") or 0)
        bucket["labels"].append(machine.get("label"))
    for bucket in out.values():
        bucket["usd_per_hour"] = round(bucket["usd_per_hour"], 4)
        bucket["usd_per_day"] = round(bucket["usd_per_hour"] * 24, 2)
    return out
