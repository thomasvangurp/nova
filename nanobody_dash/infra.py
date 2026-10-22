"""Read-only infrastructure cost collection for SN68 and SN2.

vast.ai is read live -- every running instance reports ``dph_total``, so the
burn rate is exact rather than estimated, and the invoice endpoint gives what
has already been charged.  AWS Cost Explorer supplies daily account spend and
the month forecast for the Ohio SN2 workload.  AWS calls are explicitly
allowlisted to STS identity and Cost Explorer reads: this module cannot change
billing methods, create resources, make Marketplace purchases, or charge a
card.  A stale credential must not silently zero either cost line.

The API key is read from the Keychain at call time and never leaves this
process.  AWS credentials are resolved by the AWS CLI's normal provider chain
or an opaque named-profile locator.  Errors report a status only; raw provider
responses and authentication diagnostics are never logged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import statistics as st
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = "https://console.vast.ai/api/v0"
KEYCHAIN_CANDIDATES = ("vast.ai", "vastai-api")
# Confirmed 2026-10-02: these are two DIFFERENT Vast.ai accounts, not one
# account under two label patterns. "vastai-api" is production (SN68
# nanobody/compound/blueprint boxes); "vast.ai" is a separate account for
# unrelated bank/signal boxes. vast_key() picking only the first available
# (alphabetically "vast.ai") silently dropped production's ~$70/day entirely.
PRODUCTION_VAST_SERVICE = "vastai-api"
BANK_BOXES_VAST_SERVICE = "vast.ai"
AWS_ACCOUNT = "668100850015"
AWS_BILLING_REGION = "us-east-1"
AWS_WORKLOAD_REGION = "us-east-2"
AWS_ALLOWED_READS = {
    ("sts", "get-caller-identity"),
    ("ce", "get-cost-and-usage"),
    ("ce", "get-cost-forecast"),
}
CLOUD_STATE = Path("data/cloud_costs.json")


class InfraError(RuntimeError):
    """Status codes only -- never a key, header or raw body."""


def _safe_json(path: Path) -> dict:
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".next")
    staged.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.chmod(staged, 0o600)
    os.replace(staged, path)


def _aws_json(arguments: list[str], profile: str | None = None,
              timeout: int = 90) -> dict:
    """Run one allowlisted AWS read and return parsed JSON privately.

    The CLI resolves credentials itself.  stdout is parsed in this process and
    stderr is discarded, so neither credentials nor authentication responses
    can enter the dashboard log.  Only Cost Explorer reads are possible here.
    """
    if tuple(arguments[:2]) not in AWS_ALLOWED_READS:
        raise InfraError("AWS operation is not in the read-only allowlist")
    binary = os.environ.get("NOVA_AWS_BIN") or shutil.which("aws")
    if not binary and Path("/opt/homebrew/bin/aws").is_file():
        binary = "/opt/homebrew/bin/aws"
    if not binary:
        raise InfraError("AWS CLI is unavailable")
    command = [binary, *arguments, "--region", AWS_BILLING_REGION,
               "--output", "json", "--no-cli-pager"]
    environment = None
    if profile:
        command.extend(["--profile", profile])
        # A stale temporary shell session otherwise takes precedence over the
        # explicitly selected long-lived profile.  Remove only credential
        # variables from this child's private environment; never inspect or
        # log their values.
        environment = os.environ.copy()
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                     "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN",
                     "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
            environment.pop(name, None)
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=environment, check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise InfraError("AWS read was unavailable") from None
    if completed.returncode != 0:
        raise InfraError("AWS read was not authorized")
    try:
        body = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise InfraError("AWS read returned invalid data") from None
    if not isinstance(body, dict):
        raise InfraError("AWS read returned invalid data")
    return body


def _month_after(day: dt.date) -> dt.date:
    return (day.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def _aws_cost_category(service: str, usage_type: str) -> str:
    """Map Cost Explorer line items to stable dashboard categories."""
    service_lower = service.lower()
    usage_lower = usage_type.lower()
    if "tax" in service_lower:
        return "tax"
    if "datatransfer" in usage_lower or "data transfer" in service_lower:
        return "data-transfer"
    # Service ownership wins over generic usage names such as BoxUsage.
    if "lightsail" in service_lower:
        return "lightsail"
    if "virtual private cloud" in service_lower:
        return "vpc"
    if "elastic compute cloud - compute" in service_lower \
            or "boxusage" in usage_lower:
        return "compute"
    if any(token in usage_lower for token in
           ("ebs:", "volumeusage", "snapshotusage")):
        return "storage"
    if "elastic compute cloud" in service_lower or service_lower == "ec2 - other":
        return "ec2-other"
    return "other"


def _operational_exclusion(day: str, usage_type: str,
                           exclusions: list[dict]) -> str | None:
    """Return the reason when a billed line is intentionally non-recurring."""
    for exclusion in exclusions:
        days = {str(value) for value in exclusion.get("days") or []}
        needle = str(exclusion.get("usage_type_contains") or "")
        if days and day not in days:
            continue
        if needle and needle.lower() not in usage_type.lower():
            continue
        return str(exclusion.get("reason") or "one-time cost")
    return None


def aws_cost_explorer(account: str = AWS_ACCOUNT, days: int = 90,
                      profile: str | None = None, now: dt.date | None = None,
                      fetch=None,
                      operational_exclusions: list[dict] | None = None) -> dict:
    """Return allowlisted daily spend, accrued bill, and forecast for SN2."""
    today = now or dt.datetime.now(dt.timezone.utc).date()
    call = fetch or (lambda args: _aws_json(args, profile=profile))
    identity = call(["sts", "get-caller-identity"])
    if str(identity.get("Account") or "") != str(account):
        raise InfraError("AWS credential is not for the configured SN2 account")

    start = today - dt.timedelta(days=max(1, int(days)) - 1)
    end = today + dt.timedelta(days=1)
    usage = call([
        "ce", "get-cost-and-usage",
        "--time-period", f"Start={start.isoformat()},End={end.isoformat()}",
        "--granularity", "DAILY", "--metrics", "UnblendedCost", "UsageQuantity",
        "--group-by", "Type=DIMENSION,Key=SERVICE",
        "Type=DIMENSION,Key=USAGE_TYPE",
    ])
    exclusions = operational_exclusions or []
    daily = []
    billed_daily = []
    billed_breakdown: dict[str, float] = {}
    operating_breakdown: dict[str, float] = {}
    excluded_by_reason: dict[str, float] = {}
    usage_breakdown: dict[tuple[str, str, str], dict[str, float | str]] = {}
    operating_daily_by_category: dict[str, dict[str, float]] = {}
    month_start = today.replace(day=1).isoformat()
    for result in usage.get("ResultsByTime") or []:
        period = result.get("TimePeriod") or {}
        day = period.get("Start")
        if not day:
            continue
        billed_for_day: dict[str, float] = {}
        operating_for_day: dict[str, float] = {}
        groups = result.get("Groups") or []
        if groups:
            for group in groups:
                keys = group.get("Keys") or []
                service = str(keys[0]) if keys else ""
                usage_type = str(keys[1]) if len(keys) > 1 else ""
                metric = (group.get("Metrics") or {}).get("UnblendedCost") or {}
                try:
                    amount = float(metric.get("Amount"))
                except (TypeError, ValueError):
                    continue
                usage_metric = (group.get("Metrics") or {}).get("UsageQuantity") or {}
                try:
                    quantity = float(usage_metric.get("Amount") or 0)
                except (TypeError, ValueError):
                    quantity = 0.0
                usage_unit = str(usage_metric.get("Unit") or "")
                category = _aws_cost_category(service, usage_type)
                billed_for_day[category] = billed_for_day.get(category, 0) + amount
                reason = _operational_exclusion(str(day), usage_type, exclusions)
                if reason:
                    if str(day) >= month_start:
                        excluded_by_reason[reason] = \
                            excluded_by_reason.get(reason, 0) + amount
                else:
                    operating_for_day[category] = \
                        operating_for_day.get(category, 0) + amount
                if str(day) >= month_start and usage_type:
                    usage_key = (service, usage_type, usage_unit)
                    usage_row = usage_breakdown.setdefault(usage_key, {
                        "service": service, "usage_type": usage_type,
                        "unit": usage_unit, "quantity": 0.0, "usd": 0.0,
                    })
                    usage_row["quantity"] = \
                        float(usage_row["quantity"]) + quantity
                    usage_row["usd"] = float(usage_row["usd"]) + amount
        else:
            # Compatibility with older/cached Cost Explorer responses.  They
            # remain usable, but cannot provide a service breakdown.
            metric = (result.get("Total") or {}).get("UnblendedCost") or {}
            try:
                amount = float(metric.get("Amount"))
            except (TypeError, ValueError):
                continue
            billed_for_day["total"] = amount
            operating_for_day["total"] = amount

        for category, amount in billed_for_day.items():
            if str(day) >= month_start:
                billed_breakdown[category] = \
                    billed_breakdown.get(category, 0) + amount
            billed_daily.append(dict(
                day=str(day), item=f"aws:sn2:{category}", usd=round(amount, 6),
                note=("AWS Cost Explorer UnblendedCost; SN2 account; "
                      "billing API us-east-1; workload us-east-2"),
            ))
        for category, amount in operating_for_day.items():
            if str(day) >= month_start:
                operating_breakdown[category] = \
                    operating_breakdown.get(category, 0) + amount
            daily.append(dict(
                day=str(day), item=f"aws:sn2:{category}", usd=round(amount, 6),
                note=("AWS Cost Explorer recurring operations; one-time "
                      "costs excluded by explicit policy"),
            ))
        operating_daily_by_category[str(day)] = dict(operating_for_day)

    mtd_usd = sum(row["usd"] for row in billed_daily
                  if row["day"] >= month_start)
    operating_mtd_usd = sum(row["usd"] for row in daily
                            if row["day"] >= month_start)
    next_month = _month_after(today)
    forecast_usd = 0.0
    if end < next_month:
        forecast = call([
            "ce", "get-cost-forecast",
            "--time-period", f"Start={end.isoformat()},End={next_month.isoformat()}",
            "--metric", "UNBLENDED_COST", "--granularity", "MONTHLY",
        ])
        try:
            forecast_usd = float((forecast.get("Total") or {}).get("Amount") or 0)
        except (TypeError, ValueError):
            forecast_usd = 0.0
    # Cost Explorer's provider forecast is account-wide and cannot subtract the
    # explicitly identified one-time migration line.  Feeding it back into the
    # operating projection made the migration reappear as future spend.  Build
    # the forecast from the last seven complete, already-filtered operating
    # days instead.  Tax is posted in lumps, so apply the observed MTD tax ratio
    # to the recent pre-tax run rate rather than waiting for another tax row.
    complete_days = sorted(
        day for day in operating_daily_by_category
        if month_start <= day < today.isoformat()
    )
    recent_days = complete_days[-7:]
    recent_pre_tax = [
        sum(amount for category, amount
            in operating_daily_by_category[day].items() if category != "tax")
        for day in recent_days
    ]
    if recent_pre_tax:
        pre_tax_run_rate = st.mean(recent_pre_tax)
    else:
        current = operating_daily_by_category.get(today.isoformat(), {})
        pre_tax_run_rate = sum(
            amount for category, amount in current.items() if category != "tax"
        )
    mtd_tax = float(operating_breakdown.get("tax") or 0)
    mtd_pre_tax = sum(
        float(amount) for category, amount in operating_breakdown.items()
        if category != "tax"
    )
    tax_ratio = mtd_tax / mtd_pre_tax if mtd_pre_tax > 0 else 0.0
    # A malformed or provisional tax row must not multiply the run rate without
    # bound.  Fifty percent is already far above the observed account ratio.
    tax_ratio = min(max(tax_ratio, 0.0), 0.5)
    operating_run_rate = pre_tax_run_rate * (1.0 + tax_ratio)
    complete_mtd = sum(
        sum(operating_daily_by_category[day].values()) for day in complete_days
    )
    remaining_days = max(0, (next_month - today).days)
    operating_forecast = operating_run_rate * remaining_days
    operating_projection = complete_mtd + operating_forecast

    return dict(
        account_id=str(account), project="SN2", billing_region=AWS_BILLING_REGION,
        workload_region=AWS_WORKLOAD_REGION, status="live",
        collected_at=time.time(), period=today.strftime("%Y-%m"),
        # AWS Bills calls the amount accrued so far the "estimated bill".  It
        # is not the Cost Explorer forecast for the end of the month.
        bill_to_date_usd=round(mtd_usd, 2),
        mtd_usd=round(mtd_usd, 2),
        operating_mtd_usd=round(operating_mtd_usd, 2),
        one_time_excluded_usd=round(mtd_usd - operating_mtd_usd, 2),
        # Retain the raw provider value for audit, but never use it for
        # operations or card funding because it includes excluded history.
        provider_forecast_remaining_usd=round(forecast_usd, 2),
        operating_usd_per_day=round(operating_run_rate, 2),
        forecast_remaining_usd=round(operating_forecast, 2),
        forecast_remainder_usd=round(operating_forecast, 2),
        projected_month_end_usd=round(operating_projection, 2),
        # Retain the old key while existing cached state ages out.  Consumers
        # must label it as a projection, never as the AWS estimated bill.
        estimated_month_usd=round(operating_projection, 2),
        breakdown_mtd_usd={
            key: round(value, 2) for key, value in billed_breakdown.items()
        },
        operating_breakdown_mtd_usd={
            key: round(value, 2) for key, value in operating_breakdown.items()
        },
        usage_breakdown_mtd=[
            {
                **row,
                "quantity": round(float(row["quantity"]), 3),
                "usd": round(float(row["usd"]), 2),
            }
            for row in sorted(
                usage_breakdown.values(),
                key=lambda value: abs(float(value["usd"])), reverse=True,
            )
        ],
        excluded_by_reason_usd={
            key: round(value, 2) for key, value in excluded_by_reason.items()
        },
        daily=daily, billed_daily=billed_daily,
        source="AWS Cost Explorer (read-only)",
    )


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


def vast_keys(services=KEYCHAIN_CANDIDATES) -> dict[str, str]:
    """Return {service_name: key} for every candidate Keychain item that
    resolves. Unlike vast_key() (stops at the first hit), this reads ALL
    available accounts: confirmed 2026-10-02 "vastai-api" and "vast.ai" are
    two genuinely different Vast.ai accounts (production SN68 boxes vs.
    unrelated bank/signal boxes), not one account under two label patterns,
    so vast_key() picking only the first one silently dropped the other
    account's spend entirely (undercounted production by ~$70/day since the
    bank-boxes account happened to resolve first)."""
    found = {}
    for service in services:
        key = _keychain(service)
        if key:
            found[service] = key
    if not found:
        raise InfraError(f"no Keychain item among {services}")
    return found


def vast_get(path: str, key: str, timeout: int = 45):
    if os.environ.get("NOVA_DASH_LOCAL_PI") == "1":
        operation = {"instances/": "instances", "users/current/invoices/": "invoices"}.get(path)
        if operation is None:
            raise InfraError("Vast operation is not in the read-only allowlist")
        try:
            result = subprocess.run(
                ["/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai",
                 "show", operation, "--raw"], capture_output=True, timeout=timeout)
            if result.returncode:
                raise InfraError("Vast CLI read failed")
            return json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            raise InfraError("Vast CLI read unavailable") from None
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


def funding_events(key: str, days: int = 14) -> list[dict]:
    """Account top-ups, never compute cost.

    Vast's endpoint is named ``invoices`` but the rows returned for this
    account are Stripe funding payments.  Treating their absolute value as GPU
    spend created the old, badly inflated infrastructure bars.
    """
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
        # Vast records account funding payments as negative invoice amounts.
        daily[when.strftime("%Y-%m-%d")] = round(
            daily.get(when.strftime("%Y-%m-%d"), 0.0) + abs(float(amount)), 4)
    return [dict(day=d, funded_usd=v) for d, v in sorted(daily.items())]


#: SN68 compound: five dedicated per-reaction boxes, confirmed 2026-10-02
#: against the live Vast.ai instance list -- each is labelled
#: "nova-rxn{1-5}-nim-4x4090" (some with a "-v2" suffix from a later
#: redeploy). The label is Vast.ai account metadata, not visible from inside
#: the container itself, so this must match the API's own `label` field, not
#: anything read off the box. Deliberately anchored ("nova-rxn[1-5]-nim-"
#: at the start) rather than a loose substring match, so a one-off same-
#: numbered experiment box (e.g. "nova-rxn3-enrichment-control-20260924",
#: seen running alongside the real production boxes) does NOT get counted
#: as production compound cost.
import re as _re

_COMPOUND_LABEL = _re.compile(r"^(nova-rxn[1-5]-nim-4x4090(-v\d+)?|nova-sanqi-\d+)$")
_BLUEPRINT_LABEL = _re.compile(r"^blueprint-")


def cost_scope(label: str | None,
               production_labels: tuple[str, ...] = ("nova-nanobody-pixi",),
               nanobody_research_labels: tuple[str, ...] = ()) -> str:
    """Classify paid Vast capacity without charging research to SN68 prod."""
    text = (label or "").strip().lower()
    exact = {value.strip().lower() for value in production_labels}
    if text in exact:
        return "sn68-prod"
    research_exact = {
        value.strip().lower() for value in nanobody_research_labels
    }
    if text in research_exact:
        return "sn68-research"
    if _COMPOUND_LABEL.match(text):
        return "sn68-compound"
    if _BLUEPRINT_LABEL.match(text):
        return "blueprint"
    if text:
        return "research"
    return "unattributed"


def record_vast_runtime(connection, live: dict,
                        production_labels: tuple[str, ...],
                        nanobody_research_labels: tuple[str, ...] = (),
                        extra_research_live: dict | None = None,
                        sampled_at: float | None = None) -> dict:
    """Sample scoped burn and publish a per-day rate for each scope.

    Samples are the authoritative spend record going forward.  The daily
    figure is the mean observed hourly rate times 24, matching the dashboard's
    revenue-rate units.  Funding payments are deliberately not consulted.

    extra_research_live: a SECOND Vast.ai account's live instance list (the
    bank-boxes account, confirmed 2026-10-02 to be a genuinely different
    account from production) whose spend counts as the "bank" scope. Read
    directly rather than through cost_scope(): that classifier's label
    patterns are SN68/nanobody-specific, and a loose "bank" label match would
    misclassify these just because their names happen to also contain "bank".
    """
    stamp = int((sampled_at or time.time()) // 60 * 60)
    totals: dict[str, dict] = {
        scope: dict(usd_per_hour=0.0, running_instances=0, gpus=0)
        for scope in ("sn68-prod", "sn68-research", "sn68-compound",
                      "blueprint", "bank", "research", "unattributed")
    }
    for machine in live.get("instances") or []:
        if machine.get("state") not in ("running", "loading"):
            continue
        scope = cost_scope(
            machine.get("label"), production_labels, nanobody_research_labels,
        )
        bucket = totals[scope]
        bucket["usd_per_hour"] += float(machine.get("dph") or 0)
        bucket["running_instances"] += 1
        bucket["gpus"] += int(machine.get("num_gpus") or 0)
    for machine in (extra_research_live or {}).get("instances") or []:
        if machine.get("state") not in ("running", "loading"):
            continue
        bucket = totals["bank"]
        bucket["usd_per_hour"] += float(machine.get("dph") or 0)
        bucket["running_instances"] += 1
        bucket["gpus"] += int(machine.get("num_gpus") or 0)
    connection.executemany(
        "INSERT OR REPLACE INTO infrastructure_samples "
        "(sampled_at,provider,scope,usd_per_hour,running_instances,gpus) "
        "VALUES (?,?,?,?,?,?)",
        [(stamp, "vast.ai", scope, bucket["usd_per_hour"],
          bucket["running_instances"], bucket["gpus"])
         for scope, bucket in totals.items()],
    )
    day = dt.datetime.fromtimestamp(stamp, dt.timezone.utc).date().isoformat()
    rows = []
    for scope in totals:
        average = connection.execute(
            "SELECT AVG(usd_per_hour) FROM infrastructure_samples "
            "WHERE provider='vast.ai' AND scope=? "
            "AND date(sampled_at,'unixepoch')=?",
            (scope, day),
        ).fetchone()[0]
        rows.append(dict(
            day=day, item=f"vast.ai:{scope}", usd=round(float(average or 0) * 24, 4),
            note="mean observed runtime rate; funding top-ups excluded",
        ))
    connection.executemany(
        "INSERT OR REPLACE INTO costs (day,item,usd,note) VALUES (?,?,?,?)",
        [(row["day"], row["item"], row["usd"], row["note"]) for row in rows],
    )
    connection.commit()
    return totals


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


def load_cloud_config(path: Path) -> dict:
    """Load non-secret provider/account labels and safety policy."""
    config = _safe_json(path)
    aws = config.setdefault("aws", {})
    aws.setdefault("account_id", AWS_ACCOUNT)
    aws.setdefault("project", "SN2")
    aws.setdefault("billing_region", AWS_BILLING_REGION)
    aws.setdefault("workload_region", AWS_WORKLOAD_REGION)
    # Cap the Cost Explorer history window: get-cost-and-usage with
    # --group-by SERVICE,USAGE_TYPE truncates past ~80 days (the provider
    # group limit), silently dropping the most recent month's EC2 compute and
    # zeroing the bill cell. 60 days stays well under that ceiling.
    aws.setdefault("history_days", 60)
    aws.setdefault("refresh_hours", 6)
    payments = config.setdefault("payments", {})
    payments.setdefault("cloud_funding_source", "Krak virtual card")
    payments.setdefault("real_card_allowed", False)
    return config


def _manual_aws_snapshot(config: dict, today: dt.date) -> dict:
    period = today.strftime("%Y-%m")
    rows = config.get("manual_snapshots") or []
    match = next((row for row in rows if row.get("period") == period), None)
    if not match:
        return {}
    return dict(
        account_id=str(config.get("account_id") or AWS_ACCOUNT), project="SN2",
        billing_region=config.get("billing_region", AWS_BILLING_REGION),
        workload_region=config.get("workload_region", AWS_WORKLOAD_REGION),
        status="manual reference", period=period,
        bill_to_date_usd=float(
            match.get("bill_to_date_usd")
            if match.get("bill_to_date_usd") is not None
            else match.get("estimated_month_usd") or 0
        ),
        as_of=match.get("as_of"), source=match.get("source"), daily=[],
    )


def _aws_refresh_due(state: dict, hours: float, now: float) -> bool:
    attempted = float((state.get("aws") or {}).get("attempted_at") or 0)
    return now - attempted >= max(1.0, float(hours)) * 3600


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--costs", type=Path, default=Path("config/costs.json"))
    parser.add_argument("--cloud-costs", type=Path,
                        default=Path("config/cloud_costs.json"))
    parser.add_argument("--cloud-state", type=Path, default=CLOUD_STATE)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--force-aws", action="store_true",
                        help="refresh the read-only AWS cost cache immediately")
    arguments = parser.parse_args()

    from .economics import connect, record_costs
    connection = connect(arguments.database)
    config = load_cloud_config(arguments.cloud_costs)
    state = _safe_json(arguments.cloud_state)
    state["payments"] = config.get("payments") or {}
    report: dict = {}
    try:
        keys = {PRODUCTION_VAST_SERVICE: None} if os.environ.get("NOVA_DASH_LOCAL_PI") == "1" else vast_keys()
        production_service = (
            PRODUCTION_VAST_SERVICE if PRODUCTION_VAST_SERVICE in keys
            else next(iter(keys)))
        production_key = keys[production_service]
        report["keychain_service"] = production_service
        live = running_cost(production_key)
        report["vast_usd_per_day_now"] = live["usd_per_day"]
        report["vast_instances"] = len(live["instances"])
        report["vast_running"] = sum(1 for i in live["instances"]
                                     if i["state"] in ("running", "loading"))
        funding = funding_events(production_key, arguments.days)
        report["vast_funding_days"] = len(funding)

        # Bank-boxes account, tracked separately but folded into the
        # "research" scope (see record_vast_runtime's extra_research_live).
        # Missing/unreachable is non-fatal -- production tracking must not
        # break because the second account's Keychain item is absent.
        #
        # Verified 2026-10-02: after refreshing the stale "vastai-api" item,
        # the two Keychain items turned out to be byte-identical -- one
        # real account, not two. Guard against double-counting the SAME
        # instances into both the normal scope AND "research" if that is
        # ever true again (whether because they're the same key, or two
        # different keys that happen to resolve to the same account).
        bank_boxes_key = keys.get(BANK_BOXES_VAST_SERVICE)
        bank_boxes_live = None
        if bank_boxes_key and bank_boxes_key == production_key:
            report["bank_boxes_same_account_as_production"] = True
        elif bank_boxes_key:
            try:
                bank_boxes_live = running_cost(bank_boxes_key)
                report["bank_boxes_usd_per_day_now"] = bank_boxes_live["usd_per_day"]
                report["bank_boxes_running"] = sum(
                    1 for i in bank_boxes_live["instances"]
                    if i["state"] in ("running", "loading"))
            except InfraError as error:
                report["bank_boxes_error"] = str(error)[:200]
        # Remove only the derived rows created by the former incorrect
        # invoice-as-spend interpretation. The raw provider response was never
        # stored, and no payment or provider data is changed.
        removed = connection.execute(
            "DELETE FROM costs WHERE item='vast.ai' AND note='invoiced'"
        ).rowcount
        connection.commit()
        report["invalid_vast_cost_rows_removed"] = removed
        (Path("data")).mkdir(exist_ok=True)
        Path("data/vast_instances.json").write_text(json.dumps(live, indent=1))
        production_labels = tuple(
            (config.get("vast") or {}).get("production_labels")
            or ("nova-nanobody-pixi",)
        )
        nanobody_research_labels = tuple(
            (config.get("vast") or {}).get("nanobody_research_labels") or ()
        )
        scoped = record_vast_runtime(
            connection, live, production_labels, nanobody_research_labels,
            extra_research_live=bank_boxes_live,
        )
        by_project = cost_by_project(
            live, production_labels, nanobody_research_labels,
        )
        Path("data/vast_by_project.json").write_text(json.dumps(by_project, indent=1))
        month = dt.date.today().strftime("%Y-%m")
        state["vast"] = dict(
            provider="Vast.ai", project="SN68", status="live",
            collected_at=time.time(),
            sn68_usd_per_day=round(
                (float(scoped["sn68-prod"]["usd_per_hour"])
                 + float(scoped["sn68-research"]["usd_per_hour"])) * 24, 2
            ),
            sn68_production_usd_per_day=round(
                float(scoped["sn68-prod"]["usd_per_hour"]) * 24, 2
            ),
            sn68_research_usd_per_day=round(
                float(scoped["sn68-research"]["usd_per_hour"]) * 24, 2
            ),
            sn68_compound_usd_per_day=round(
                float(scoped["sn68-compound"]["usd_per_hour"]) * 24, 2
            ),
            research_usd_per_day=round(
                float(scoped["research"]["usd_per_hour"]) * 24, 2
            ),
            unattributed_usd_per_day=round(
                float(scoped["unattributed"]["usd_per_hour"]) * 24, 2
            ),
            blueprint_usd_per_day=round(
                float(scoped["blueprint"]["usd_per_hour"]) * 24, 2
            ),
            bank_usd_per_day=round(
                float(scoped["bank"]["usd_per_hour"]) * 24, 2
            ),
            # The bank-boxes ACCOUNT (separate from the "bank" label-scope
            # above, which is machines within the production account whose
            # label happens to contain "bank"). Already folded into
            # research_usd_per_day; kept here too so it's visible as its own
            # line per the 2026-10-02 "track both accounts separately"
            # decision.
            bank_boxes_account_usd_per_day=(
                round(float(bank_boxes_live["usd_per_day"]), 2)
                if bank_boxes_live is not None else 0.0
            ),
            funding_mtd_usd=round(sum(
                float(row["funded_usd"]) for row in funding
                if str(row["day"]).startswith(month)
            ), 2),
            running_instances=report["vast_running"],
            source="Vast.ai label-scoped runtime samples (read-only)",
        )
    except InfraError as error:
        report["vast_error"] = str(error)

    aws_config = config.get("aws") or {}
    now = time.time()
    today = dt.datetime.now(dt.timezone.utc).date()
    manual = _manual_aws_snapshot(aws_config, today)
    previous_aws = state.get("aws") if isinstance(state.get("aws"), dict) else {}
    if not previous_aws and manual:
        previous_aws = manual
    refresh_hours = float(aws_config.get("refresh_hours") or 6)
    if arguments.force_aws or _aws_refresh_due(
        {"aws": previous_aws}, refresh_hours, now
    ):
        profile = os.environ.get("NOVA_AWS_PROFILE") or aws_config.get("profile")
        try:
            aws = aws_cost_explorer(
                account=str(aws_config.get("account_id") or AWS_ACCOUNT),
                days=int(aws_config.get("history_days") or 90),
                profile=str(profile) if profile else None,
                operational_exclusions=list(
                    aws_config.get("operational_exclusions") or []
                ),
            )
            if manual and manual.get("period") == aws.get("period"):
                aws["bills_reference_usd"] = manual.get("bill_to_date_usd")
                aws["bills_reference_as_of"] = manual.get("as_of")
                aws["bills_reference_source"] = manual.get("source")
            aws["attempted_at"] = now
            state["aws"] = aws
            # Replace only dashboard-derived AWS rows.  The complete bill is
            # retained in cloud state; operating economics intentionally use
            # the recurring series after explicit one-time exclusions.
            connection.execute(
                "DELETE FROM costs WHERE item='aws:sn2' OR item LIKE 'aws:sn2:%'"
            )
            connection.commit()
            report["aws_cost_rows"] = record_costs(connection, aws["daily"])
            report["aws_status"] = "live"
            report["aws_bill_to_date_usd"] = aws["bill_to_date_usd"]
            report["aws_operating_mtd_usd"] = aws["operating_mtd_usd"]
            report["aws_projected_month_end_usd"] = \
                aws["projected_month_end_usd"]
        except InfraError:
            # Keep the last good live data or the explicitly supplied bill
            # reference.  Never replace cloud cost with a fictional zero.
            fallback = dict(previous_aws or manual)
            fallback["attempted_at"] = now
            fallback["status"] = "stale; AWS read unavailable"
            state["aws"] = fallback
            report["aws_status"] = "unavailable; retained last known cost"
    else:
        state["aws"] = previous_aws
        report["aws_status"] = "cached"

    static = load_static_costs(arguments.costs)
    if static:
        report["static_cost_rows"] = record_costs(connection, static)
    _atomic_json(arguments.cloud_state, state)
    print(json.dumps(report))
    return 0


#: Which project each vast box serves, matched against its label.  Cost is
#: meaningless in aggregate: the nanobody producer is one box, the reaction-5
#: bank is four, and anything else is neither and must not be charged against
#: either subnet's yield.
def classify(label: str | None) -> str:
    """Backward-compatible default classifier used by older callers."""
    return cost_scope(label)


def cost_by_project(live: dict,
                    production_labels: tuple[str, ...] = ("nova-nanobody-pixi",),
                    nanobody_research_labels: tuple[str, ...] = ()) -> dict:
    """Split the live burn rate by project, running instances only."""
    out: dict[str, dict] = {}
    for machine in live.get("instances", []):
        if machine.get("state") not in ("running", "loading"):
            continue
        project = cost_scope(
            machine.get("label"), production_labels, nanobody_research_labels,
        )
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


if __name__ == "__main__":
    raise SystemExit(main())
