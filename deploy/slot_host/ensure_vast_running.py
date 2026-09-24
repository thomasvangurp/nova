#!/usr/bin/env python3
"""Keep the uniquely labelled production Vast instance running.

The Vast CLI resolves its own API credential.  This program captures all raw
provider output in memory and emits only a small operational status record, so
credentials and provider responses never enter the journal.

An operator can intentionally suppress automatic recovery by creating the
configured disable marker.  Destruction is never automated: a missing instance
or an ambiguous label fails closed and requires a human decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def _instances(binary: str) -> list[dict]:
    completed = subprocess.run(
        [binary, "show", "instances", "--raw"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120,
    )
    if completed.returncode:
        raise RuntimeError("vast instance inventory unavailable")
    try:
        body = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("vast instance inventory was not valid JSON") from None
    if isinstance(body, dict):
        body = body.get("instances") or []
    if not isinstance(body, list):
        raise RuntimeError("vast instance inventory had an unexpected shape")
    return [row for row in body if isinstance(row, dict)]


def matching_instance(rows: list[dict], label: str) -> dict:
    matches = [row for row in rows if str(row.get("label") or "") == label]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one instance labelled {label!r}, found {len(matches)}"
        )
    return matches[0]


def producer_endpoints(instance: dict) -> list[tuple[str, str, int]]:
    """Return direct SSH endpoints first and the provider proxy last."""
    endpoints: list[tuple[str, str, int]] = []
    public_ip = str(instance.get("public_ipaddr") or "").strip()
    for mapping in (instance.get("ports") or {}).get("22/tcp") or []:
        if not isinstance(mapping, dict):
            continue
        try:
            port = int(mapping.get("HostPort"))
        except (TypeError, ValueError):
            continue
        if public_ip and port > 0:
            endpoints.append(("direct", public_ip, port))
    proxy_host = str(instance.get("ssh_host") or "").strip()
    try:
        proxy_port = int(instance.get("ssh_port"))
    except (TypeError, ValueError):
        proxy_port = 0
    if proxy_host and proxy_port > 0:
        proxy = ("proxy", proxy_host, proxy_port)
        if (proxy_host, proxy_port) not in {(h, p) for _, h, p in endpoints}:
            endpoints.append(proxy)
    return endpoints


def ensure_remote_runtime(instance: dict) -> str:
    """Ensure the retained box's watchdog and artifact publisher are alive.

    The captured argv is read only on the producer. It is never returned to the
    Pi or journal, and it is rejected if it could write to the chain.
    """
    python = (
        "import pathlib,subprocess;"
        "raw=pathlib.Path('/root/artifacts/publisher_argv.bin').read_bytes().decode();"
        "argv=[x for x in raw.split('\\0') if x];"
        "assert argv and argv[0]=='tmux' and '--commit-command' not in argv;"
        "assert '--minimum-sequence-pool-size' in argv;"
        "raise SystemExit(subprocess.run(argv,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL).returncode)"
    )
    command = (
        "set -eu; "
        "if ! pgrep -f '[s]ession_watchdog.sh' >/dev/null 2>&1; then "
        "nohup /root/session_watchdog.sh >/dev/null 2>&1 & fi; "
        "if ! tmux has-session -t nova-nanobody-publisher >/dev/null 2>&1; then "
        f"python3 -c {shlex.quote(python)}; fi; "
        "sleep 3; "
        "pgrep -f '[s]ession_watchdog.sh' >/dev/null 2>&1; "
        "tmux has-session -t nova-nanobody-publisher >/dev/null 2>&1"
    )
    failures = []
    for route, host, port in producer_endpoints(instance):
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "StrictHostKeyChecking=no", "-o",
             "UserKnownHostsFile=/dev/null", "-p", str(port),
             f"root@{host}", command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=60,
        )
        if completed.returncode == 0:
            return route
        failures.append(f"{route}:rc={completed.returncode}")
    if not failures:
        raise RuntimeError("production instance has no SSH endpoint")
    raise RuntimeError("production runtime unavailable (" + ", ".join(failures) + ")")





def producer_freshness(instance: dict) -> dict:
    """Read the producer's last-published age and the chain's current epoch.

    Wall-clock age of the publisher state file is the deciding signal: a frozen
    chain-state file keeps reporting a small epoch delta forever, which is how
    the 25315-25317 producer hang stayed invisible to the liveness checks.
    """
    program = (
        "import json,os,time\n"
        "def age(p):\n"
        "    try:\n"
        "        return int(time.time()-os.stat(p).st_mtime)\n"
        "    except Exception:\n"
        "        return 10**9\n"
        "def epoch(p):\n"
        "    try:\n"
        "        return int(json.load(open(p)).get('epoch',-1))\n"
        "    except Exception:\n"
        "        return -1\n"
        "print(age('/root/artifacts/nanobody_publisher_state.json'),\n"
        "      epoch('/root/artifacts/nanobody_publisher_state.json'),\n"
        "      epoch('/root/artifacts/chain_state.json'))\n"
    )
    command = f"python3 -c {shlex.quote(program)}"
    for route, host, port in producer_endpoints(instance):
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-p", str(port), f"root@{host}", command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=60,
        )
        if completed.returncode or not completed.stdout.strip():
            continue
        try:
            age, pub_epoch, cur_epoch = (
                int(part) for part in completed.stdout.split()
            )
        except (TypeError, ValueError):
            continue
        return {
            "route": route,
            "published_age_seconds": age,
            "published_epoch": pub_epoch,
            "current_epoch": cur_epoch,
        }
    raise RuntimeError("producer freshness unavailable")


def restart_producer_runtime(instance: dict) -> str:
    """Kill a hung producer + scorer; the box watchdog relaunches them clean."""
    command = (
        "tmux kill-session -t nova-nanobody-producer 2>/dev/null; "
        "tmux kill-session -t nova-nanobody-scorerd 2>/dev/null; "
        "echo restarted"
    )
    for route, host, port in producer_endpoints(instance):
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-p", str(port), f"root@{host}", command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=60,
        )
        if completed.returncode == 0:
            return route
    raise RuntimeError("producer restart unavailable")


def _marker_age(marker: Path) -> int:
    try:
        return int(time.time() - marker.stat().st_mtime)
    except OSError:
        return 10**9


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(time.time())) + "\n")


def ensure_running(binary: str, label: str, wait_seconds: int = 240,
                   poll_seconds: int = 10) -> dict:
    row = matching_instance(_instances(binary), label)
    instance_id = int(row["id"])
    state = str(row.get("actual_status") or "unknown")
    if state == "running":
        return {"event": "producer_healthy", "id": instance_id, "state": state}

    # A stopped rental retains its disk and can be recovered.  Anything else
    # (destroyed/missing/ambiguous) must not cause this guard to rent or choose a
    # new machine automatically.
    if state not in {"exited", "stopped", "offline"}:
        raise RuntimeError(f"production instance is in non-startable state {state!r}")
    started = subprocess.run(
        [binary, "start", "instance", str(instance_id)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120,
    )
    if started.returncode:
        raise RuntimeError("vast refused the production restart")

    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        row = matching_instance(_instances(binary), label)
        state = str(row.get("actual_status") or "unknown")
        if state == "running":
            return {"event": "producer_restarted", "id": instance_id, "state": state}
        if time.monotonic() >= deadline:
            raise RuntimeError("production instance did not become running in time")
        time.sleep(max(1, poll_seconds))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vastai", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--disable-marker", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=240)
    parser.add_argument("--max-producer-age", type=int, default=9000,
                        help="restart producer if its last publish is this old (s)")
    parser.add_argument("--restart-marker", type=Path,
                        default=Path("/home/thomasg/.nova/producer-last-restart"))
    arguments = parser.parse_args(argv)

    if arguments.disable_marker.exists():
        print(json.dumps({"event": "producer_guard_disabled"}, sort_keys=True))
        return 0
    try:
        result = ensure_running(
            arguments.vastai, arguments.label, wait_seconds=arguments.wait_seconds
        )
        row = matching_instance(_instances(arguments.vastai), arguments.label)
        result["runtime_route"] = ensure_remote_runtime(row)
        freshness = producer_freshness(row)
        result["producer"] = {
            "published_epoch": freshness["published_epoch"],
            "current_epoch": freshness["current_epoch"],
            "published_age_seconds": freshness["published_age_seconds"],
        }
        stale = freshness["published_age_seconds"] > arguments.max_producer_age
        if (freshness["current_epoch"] >= 0 and freshness["published_epoch"] >= 0
                and freshness["current_epoch"] - freshness["published_epoch"] >= 2):
            stale = True
        if stale:
            if _marker_age(arguments.restart_marker) < arguments.max_producer_age:
                result["producer"]["action"] = "stale_but_recently_restarted"
            else:
                result["producer"]["action"] = "restarting"
                result["producer_restart_route"] = restart_producer_runtime(row)
                _touch(arguments.restart_marker)
        else:
            result["producer"]["action"] = "fresh"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        # The exception strings above are deliberately provider-response-free.
        print(json.dumps({"event": "producer_guard_failed",
                          "reason": str(error)[:160]}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
