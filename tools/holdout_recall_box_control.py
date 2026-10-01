#!/usr/bin/env python3
"""Guard, retrieve, analyze, and stop the full-cohort holdout replay."""

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
DEST = REPO / "analysis/nanobody-gap/holdout-recall-20260922"
STATE = DEST / "supervisor_state.json"
INSTANCE_ID = 52073739
LABEL = "nova-nanobody-h2-fast-20260922"
GPU_COUNT = 8
HOURLY_COST = 4.14111111111111
SSH_HOST = "90.224.159.6"
SSH_PORT = 41774
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"
REMOTE = "/root/holdout-recall-20260922"
VAST_API_SERVICE = "vast.ai"


def _write_state(**values: object) -> None:
    previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
    STATE.write_text(json.dumps({
        **previous, **values, "updated_unix": time.time(),
    }, indent=2, sort_keys=True) + "\n")


def _ssh() -> list[str]:
    return [
        "ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(SSH_PORT),
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
    ]


def _remote_json(code: str, timeout: int = 60) -> dict | list:
    result = subprocess.run(
        [*_ssh(), "root@" + SSH_HOST, "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("remote_failed")
    return json.loads(result.stdout)


def remote_status() -> dict:
    code = f"""
import json, os, time
from pathlib import Path
base=Path({REMOTE!r}); meta=json.loads((base/'run_metadata.json').read_text())
rows=[]
for i in range(5,9):
 d=base/f'heldout_{{i:02d}}'; wave=d/'wave_000_input.json'; exact=d/'exact_all.json'; row={{'draw':i,'started':wave.is_file(),'complete':exact.is_file()}}
 if wave.is_file(): row['started_unix']=wave.stat().st_mtime
 if exact.is_file():
  data=json.loads(exact.read_text()); row.update(n=data.get('n'),protocol=data.get('scorer_protocol'),completed_unix=exact.stat().st_mtime)
  if wave.is_file(): row['duration_seconds']=exact.stat().st_mtime-wave.stat().st_mtime
 rows.append(row)
runner=False
for pid in os.listdir('/proc'):
 if not pid.isdigit(): continue
 try: args=open(f'/proc/{{pid}}/cmdline','rb').read().split(b'\\0')
 except OSError: continue
 if any(x.endswith(b'/holdout-recall-20260922/run_remote.sh') for x in args): runner=True
print(json.dumps({{'draws':rows,'complete':(base/'COMPLETE').is_file(),'runner_alive':runner,'started_unix':meta.get('started_unix'),'draw_05_started_unix':meta.get('draw_05_started_unix'),'elapsed_s':round(time.time()-meta.get('draw_05_started_unix',time.time()),1)}}))
"""
    return dict(_remote_json(code))


def remote_manifest() -> list[dict]:
    code = f"""
import hashlib,json,re
from pathlib import Path
base=Path({REMOTE!r});paths=[]
paths += sorted(base.glob('heldout_*/exact_all.json'))
paths += [p for p in (base/'run_metadata.json',base/'source_manifest.json',base/'sequences.json') if p.is_file()]
rows=[]
for path in paths:
 data=path.read_bytes(); rows.append({{'file':str(path.relative_to(base)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime':path.stat().st_mtime}})
print(json.dumps(rows))
"""
    rows = _remote_json(code)
    if not isinstance(rows, list):
        raise RuntimeError("invalid_remote_manifest")
    return rows


def fetch() -> dict:
    status = remote_status()
    rows = remote_manifest()
    allowed = re.compile(
        r"(?:heldout_0[5-8]/exact_all\.json|run_metadata\.json|source_manifest\.json|sequences\.json)"
    )
    for item in rows:
        relative = str(item.get("file", ""))
        if not allowed.fullmatch(relative):
            raise RuntimeError("unsafe_artifact_name")
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            "rsync", "-rlt", "--checksum", "--no-owner", "--no-group",
            "--no-perms", "-e", shlex.join(_ssh()),
            f"root@{SSH_HOST}:{REMOTE}/{relative}", str(target),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if not (
            result.returncode == 0
            and target.is_file()
            and target.stat().st_size == int(item["bytes"])
            and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        ):
            raise RuntimeError("artifact_verification_failed")
    exact_rows = [item for item in rows if item["file"].startswith("heldout_")]
    metadata = json.loads((DEST / "run_metadata.json").read_text())
    if exact_rows:
        last = max(float(item["mtime"]) for item in exact_rows)
        timings = status["draws"]
        start = float(timings[0]["started_unix"])
        metadata.update({
            "first_wave_started_unix": start,
            "last_draw_completed_unix": last,
            "scoring_wall_seconds": last - start,
            "hourly_cost": HOURLY_COST,
            "estimated_scoring_cost": (last - start) / 3600 * HOURLY_COST,
            "draw_timings": timings,
        })
        (DEST / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return {
        "artifacts": len(rows),
        "verified_exact": len(exact_rows),
        "verified_total": len(rows),
    }


def analyze() -> None:
    result = subprocess.run(
        [sys.executable, str(DEST / "analyze_recall.py")],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600,
    )
    if result.returncode:
        raise RuntimeError("analysis_failed")
    if not (DEST / "analysis.json").is_file() or not (DEST / "REPORT.md").is_file():
        raise RuntimeError("analysis_artifacts_missing")


def _keychain(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
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
        remote = remote_status()
    except Exception as error:
        remote = {"error_class": type(error).__name__}
    state = json.loads(STATE.read_text()) if STATE.is_file() else {}
    print(json.dumps({"remote": remote, "supervisor": state}, sort_keys=True))
    return 0


def monitor() -> int:
    deadline = time.time() + 3 * 60 * 60
    _write_state(stage="monitoring", deadline_unix=deadline)
    while time.time() < deadline:
        try:
            remote = remote_status()
            _write_state(stage="monitoring", **remote)
            if remote["complete"]:
                summary = fetch()
                if summary["verified_exact"] != 4:
                    raise RuntimeError("incomplete_artifact_set")
                for index in range(5, 9):
                    payload = json.loads(
                        (DEST / f"heldout_{index:02d}" / "exact_all.json").read_text()
                    )
                    if payload.get("scorer_protocol") != "validator-dual-sample-v1" or int(payload.get("n", -1)) != 888:
                        raise RuntimeError("invalid_exact_artifact")
                analyze()
                _write_state(stage="analyzed", **summary)
                stop()
                return 0
            if not remote["runner_alive"]:
                summary = fetch()
                _write_state(stage="runner_failed_box_held", **summary)
                return 1
        except Exception as error:
            _write_state(stage="monitor_error", error_class=type(error).__name__)
        time.sleep(30)
    summary = fetch()
    _write_state(stage="deadline_box_held", **summary)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("status", "fetch", "stop", "monitor"))
    operation = parser.parse_args().operation
    if operation == "status":
        return status()
    if operation == "fetch":
        print(json.dumps(fetch(), sort_keys=True))
        return 0
    if operation == "stop":
        stop()
        return 0
    return monitor()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _write_state(stage="failed_box_held", error_class=type(error).__name__)
        print(json.dumps({"event": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1)
