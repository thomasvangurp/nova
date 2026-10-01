#!/usr/bin/env python3
"""Guard, retrieve, analyze, and stop the isolated Hamming-2 GPU replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "analysis/nanobody-gap/hamming2-fast-20260922"
STATE = DEST / "supervisor_state.json"
INSTANCE_ID = 52073739
LABEL = "nova-nanobody-h2-fast-20260922"
GPU_COUNT = 8
SSH_HOST = "90.224.159.6"
SSH_PORT = 41774
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"
REMOTE = "/root/hamming2-fast-20260922"
VAST_API_SERVICE = "vast.ai"


def _write_state(**values: object) -> None:
    previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
    STATE.write_text(json.dumps({
        **previous,
        **values,
        "updated_unix": time.time(),
    }, indent=2, sort_keys=True) + "\n")


def _ssh() -> list[str]:
    return [
        "ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(SSH_PORT),
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
    ]


def _remote_status() -> dict:
    code = f"""
import json, os
base={REMOTE!r}
main=sum(os.path.isfile(os.path.join(base, f'pass_{{i:02d}}', 'exact_all.json')) for i in range(1,13))
extra=sum(os.path.isfile(os.path.join(base, f'easynano_extra_{{i:02d}}', 'exact_all.json')) for i in range(13,21))
runner=False
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try: args=open(f'/proc/{{pid}}/cmdline','rb').read().split(b'\\0')
    except OSError: continue
    if any(x.endswith(b'/run_remote.sh') for x in args): runner=True
print(json.dumps({{'main_complete':main,'extra_complete':extra,'complete':os.path.isfile(os.path.join(base,'COMPLETE')),'runner_alive':runner}}))
"""
    result = subprocess.run(
        [*_ssh(), "root@" + SSH_HOST, "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError("remote_status_failed")
    payload = json.loads(result.stdout)
    allowed = {
        key: payload.get(key)
        for key in ("main_complete", "extra_complete", "complete", "runner_alive")
    }
    if not isinstance(allowed["main_complete"], int):
        raise RuntimeError("invalid_remote_status")
    return allowed


def _remote_manifest() -> list[dict]:
    code = f"""
import hashlib, json
from pathlib import Path
base=Path({REMOTE!r}); rows=[]
for path in sorted(base.glob('pass_*/exact_all.json')) + sorted(base.glob('easynano_extra_*/exact_all.json')):
    data=path.read_bytes(); rows.append({{'file':str(path.relative_to(base)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}})
print(json.dumps(rows))
"""
    result = subprocess.run(
        [*_ssh(), "root@" + SSH_HOST, "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError("remote_manifest_failed")
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("invalid_remote_manifest")
    return rows


def fetch() -> dict:
    rows = _remote_manifest()
    for item in rows:
        relative = str(item.get("file", ""))
        if not re.fullmatch(
            r"(?:pass_\d{2}|easynano_extra_\d{2})/exact_all\.json", relative
        ):
            raise RuntimeError("unsafe_artifact_name")
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        transport = shlex.join(_ssh())
        result = subprocess.run(
            [
                "rsync", "-rlt", "--checksum", "--no-owner", "--no-group",
                "--no-perms", "-e", transport,
                f"root@{SSH_HOST}:{REMOTE}/{relative}", str(target),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        valid = (
            result.returncode == 0
            and target.is_file()
            and target.stat().st_size == int(item["bytes"])
            and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        )
        if not valid:
            raise RuntimeError("artifact_verification_failed")
    return {
        "artifacts": len(rows),
        "main_complete": sum(item["file"].startswith("pass_") for item in rows),
        "extra_complete": sum(
            item["file"].startswith("easynano_extra_") for item in rows
        ),
    }


def analyze() -> None:
    result = subprocess.run(
        [sys.executable, str(DEST / "analyze_results.py")],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    if result.returncode:
        raise RuntimeError("analysis_failed")
    if not (DEST / "analysis.json").is_file() or not (DEST / "REPORT.md").is_file():
        raise RuntimeError("analysis_artifacts_missing")


def _keychain(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError("credential_unavailable")
    return result.stdout.decode().strip()


def _api(path: str, body: object | None = None, method: str = "GET") -> dict:
    credential = _keychain(VAST_API_SERVICE)
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/" + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + credential,
            "Content-Type": "application/json",
        },
    )
    credential = ""
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def stop() -> None:
    instances = _api("instances/?owner=me").get("instances", [])
    matches = [item for item in instances if item.get("id") == INSTANCE_ID]
    if len(matches) != 1:
        raise RuntimeError("instance_not_found")
    host = matches[0]
    if host.get("label") != LABEL or host.get("num_gpus") != GPU_COUNT:
        raise RuntimeError("instance_guard_failed")
    if host.get("actual_status") in {"stopped", "exited"}:
        _write_state(stage="stopped", instance_status=host.get("actual_status"))
        return
    result = _api(
        "instances/" + str(INSTANCE_ID) + "/", {"state": "stopped"}, "PUT"
    )
    if result.get("success") is not True:
        raise RuntimeError("stop_failed")
    _write_state(stage="stop_requested")


def status() -> int:
    try:
        remote = _remote_status()
    except Exception as error:
        remote = {"error_class": type(error).__name__}
    state = json.loads(STATE.read_text()) if STATE.is_file() else {}
    print(json.dumps({"remote": remote, "supervisor": state}, sort_keys=True))
    return 0


def monitor(*, keep_running: bool = False) -> int:
    deadline = time.time() + 4 * 60 * 60
    _write_state(stage="monitoring", deadline_unix=deadline)
    while time.time() < deadline:
        try:
            remote = _remote_status()
            _write_state(stage="monitoring", **remote)
            if remote["complete"]:
                summary = fetch()
                if summary["main_complete"] != 12 or summary["extra_complete"] != 8:
                    raise RuntimeError("incomplete_artifact_set")
                analyze()
                _write_state(stage="analyzed", **summary)
                if keep_running:
                    _write_state(stage="held_for_followup", **summary)
                else:
                    stop()
                return 0
            if not remote["runner_alive"]:
                summary = fetch()
                if keep_running:
                    _write_state(stage="runner_stopped_for_followup", **summary)
                    return 0
                _write_state(stage="runner_failed", **summary)
                stop()
                return 1
        except Exception as error:
            _write_state(stage="monitor_error", error_class=type(error).__name__)
        time.sleep(60)
    try:
        summary = fetch()
        _write_state(stage="deadline", **summary)
    finally:
        stop()
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("status", "fetch", "stop", "monitor"))
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="retrieve and analyze a completed run without stopping its instance",
    )
    arguments = parser.parse_args()
    operation = arguments.operation
    if operation == "status":
        return status()
    if operation == "fetch":
        print(json.dumps(fetch(), sort_keys=True))
        return 0
    if operation == "stop":
        stop()
        return 0
    return monitor(keep_running=arguments.keep_running)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _write_state(stage="failed", error_class=type(error).__name__)
        print(json.dumps({"event": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1)
