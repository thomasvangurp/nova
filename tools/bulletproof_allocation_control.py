#!/usr/bin/env python3
"""Monitor, retrieve, verify, analyze, and stop the fixed research box."""

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


REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "analysis/nanobody-gap/bulletproof-allocation-20260923"
STATE = DEST / "supervisor_state.json"
REMOTE = "/root/bulletproof-allocation-20260923"
SSH_HOST = "90.224.159.6"
SSH_PORT = 41774
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"


def write_state(**values: object) -> None:
    previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
    STATE.write_text(json.dumps({
        **previous, **values, "updated_unix": time.time(),
    }, indent=2, sort_keys=True) + "\n")


def ssh() -> list[str]:
    return [
        "ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(SSH_PORT),
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",
    ]


def remote_json(code: str, timeout: int = 60) -> object:
    result = subprocess.run(
        [*ssh(), "root@" + SSH_HOST, "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("remote_command_failed")
    return json.loads(result.stdout)


def status() -> dict:
    code = f"""
import json,os
from pathlib import Path
p=Path({REMOTE!r})
dev=sum((p/f'development_{{i:02d}}'/'exact_all.json').is_file() for i in range(1,13))
fresh=sum((p/f'fresh_{{i:02d}}'/'exact_all.json').is_file() for i in range(1,33))
runner=False
for pid in os.listdir('/proc'):
 if not pid.isdigit(): continue
 try: args=open(f'/proc/{{pid}}/cmdline','rb').read().split(b'\\0')
 except OSError: continue
 if any(x.endswith(b'/bulletproof-allocation-20260923/run_remote.sh') for x in args): runner=True
print(json.dumps({{'development':dev,'fresh':fresh,'fleets_frozen':(p/'FLEET_FREEZE.sha256').is_file(),'complete':(p/'COMPLETE').is_file(),'runner_alive':runner}}))
"""
    return dict(remote_json(code))


def manifest() -> list[dict]:
    code = f"""
import hashlib,json
from pathlib import Path
p=Path({REMOTE!r}); paths=sorted(p.glob('development_*/exact_all.json'))+sorted(p.glob('fresh_*/exact_all.json'))
paths += [x for x in (p/'selected_fleets.json',p/'fresh_sequences.json',p/'FLEET_FREEZE.sha256') if x.is_file()]
print(json.dumps([{{'file':str(x.relative_to(p)),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}} for x in paths]))
"""
    value = remote_json(code)
    if not isinstance(value, list):
        raise RuntimeError("invalid_manifest")
    return value


def fetch() -> dict:
    rows = manifest()
    allowed = re.compile(r"(?:(?:development|fresh)_\d{2}/exact_all\.json|selected_fleets\.json|fresh_sequences\.json|FLEET_FREEZE\.sha256)")
    for item in rows:
        relative = str(item["file"])
        if not allowed.fullmatch(relative):
            raise RuntimeError("unsafe_artifact_name")
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            "rsync", "-rlt", "--checksum", "--no-owner", "--no-group", "--no-perms",
            "-e", shlex.join(ssh()), f"root@{SSH_HOST}:{REMOTE}/{relative}", str(target),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if not (result.returncode == 0 and target.is_file()
                and target.stat().st_size == int(item["bytes"])
                and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]):
            raise RuntimeError("artifact_verification_failed")
    if (DEST / "FLEET_FREEZE.sha256").is_file():
        expected = {}
        for line in (DEST / "FLEET_FREEZE.sha256").read_text().splitlines():
            digest, raw = line.split(maxsplit=1)
            expected[Path(raw).name] = digest
        for name in ("selected_fleets.json", "fresh_sequences.json"):
            if name in expected and hashlib.sha256((DEST / name).read_bytes()).hexdigest() != expected[name]:
                raise RuntimeError("fleet_freeze_mismatch")
    return {
        "development": sum(item["file"].startswith("development_") for item in rows),
        "fresh": sum(item["file"].startswith("fresh_") and "/" in item["file"] for item in rows),
        "files": len(rows),
    }


def analyze() -> None:
    result = subprocess.run(
        [sys.executable, str(DEST / "analyze_fresh.py")], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600,
    )
    if result.returncode:
        raise RuntimeError("analysis_failed")


def stop_box() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "tools/h2_selector_proof_box_control.py"), "stop"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90,
    )
    if result.returncode:
        raise RuntimeError("box_stop_failed")


def monitor() -> int:
    deadline = time.time() + 8 * 60 * 60
    write_state(stage="monitoring", deadline_unix=deadline)
    while time.time() < deadline:
        try:
            remote = status()
            write_state(stage="monitoring", **remote)
            if remote["complete"]:
                summary = fetch()
                if summary["development"] != 12 or summary["fresh"] != 32:
                    raise RuntimeError("incomplete_artifact_set")
                analyze()
                write_state(stage="analyzed", **summary)
                stop_box()
                write_state(stage="stop_requested", **summary)
                return 0
            if not remote["runner_alive"]:
                summary = fetch()
                write_state(stage="runner_failed_box_held", **summary)
                return 1
        except Exception as error:
            write_state(stage="monitor_error", error_class=type(error).__name__)
        time.sleep(30)
    summary = fetch()
    write_state(stage="deadline_box_held", **summary)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("status", "fetch", "monitor", "stop"))
    args = parser.parse_args()
    if args.operation == "status":
        try:
            remote = status()
        except Exception as error:
            remote = {"error_class": type(error).__name__}
        print(json.dumps(remote, sort_keys=True))
        return 0
    if args.operation == "fetch":
        print(json.dumps(fetch(), sort_keys=True))
        return 0
    if args.operation == "stop":
        stop_box()
        return 0
    return monitor()


if __name__ == "__main__":
    sys.exit(main())
