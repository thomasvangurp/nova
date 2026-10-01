#!/usr/bin/env python3
"""Guarded control/retrieval for the preregistered H2 selector proof."""

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
DEST = REPO / "analysis/nanobody-gap/h2-selector-proof-20260922"
STATE = DEST / "supervisor_state.json"
INSTANCE_ID = 52073739
LABEL = "nova-nanobody-h2-fast-20260922"
GPU_COUNT = 8
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"
REMOTE = "/root/h2-selector-proof-20260922"
VAST_API_SERVICE = "vast.ai"
_SSH_ENDPOINT: tuple[str, int] | None = None


def write_state(**values: object) -> None:
    previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
    STATE.write_text(json.dumps({
        **previous, **values, "updated_unix": time.time(),
    }, indent=2, sort_keys=True) + "\n")


def ssh_endpoint() -> tuple[str, int]:
    global _SSH_ENDPOINT
    if _SSH_ENDPOINT is not None:
        return _SSH_ENDPOINT
    instance = guarded_instance()
    if instance.get("actual_status") != "running":
        raise RuntimeError("instance_not_running")
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if not host or not port:
        host = instance.get("public_ipaddr")
        entries = (instance.get("ports") or {}).get("22/tcp") or []
        port = next(
            (entry.get("HostPort") for entry in entries if entry.get("HostPort")),
            None,
        )
    if not host or not port:
        raise RuntimeError("ssh_endpoint_unavailable")
    _SSH_ENDPOINT = (str(host), int(port))
    return _SSH_ENDPOINT


def ssh() -> list[str]:
    _, port = ssh_endpoint()
    return [
        "ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(port),
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
    ]


def remote_json(code: str, timeout: int = 60) -> dict | list:
    host, _ = ssh_endpoint()
    result = subprocess.run(
        [*ssh(), "root@" + host, "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("remote_failed")
    return json.loads(result.stdout)


def remote_status() -> dict:
    code = f"""
import json,os,time
from pathlib import Path
base=Path({REMOTE!r}); rows=[]
for i in range(9,17):
 d=base/f'fresh_{{i:02d}}'; wave=d/'wave_000_input.json'; exact=d/'exact_all.json'; row={{'draw':i,'started':wave.is_file(),'complete':exact.is_file()}}
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
 if any(x.endswith(b'/h2-selector-proof-20260922/run_remote.sh') for x in args): runner=True
print(json.dumps({{'draws':rows,'complete':(base/'COMPLETE').is_file(),'runner_alive':runner}}))
"""
    return dict(remote_json(code))


def remote_manifest() -> list[dict]:
    code = f"""
import hashlib,json
from pathlib import Path
base=Path({REMOTE!r}); paths=sorted(base.glob('fresh_*/exact_all.json'))
paths += [p for p in (base/'run_metadata.json',base/'preregistration.json',base/'sequences.json') if p.is_file()]
rows=[]
for path in paths:
 data=path.read_bytes(); rows.append({{'file':str(path.relative_to(base)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime':path.stat().st_mtime}})
print(json.dumps(rows))
"""
    result = remote_json(code)
    if not isinstance(result, list):
        raise RuntimeError("invalid_remote_manifest")
    return result


def fetch() -> dict:
    host, _ = ssh_endpoint()
    status = remote_status()
    rows = remote_manifest()
    allowed = re.compile(
        r"(?:fresh_(?:0[9]|1[0-6])/exact_all\.json|run_metadata\.json|preregistration\.json|sequences\.json)"
    )
    frozen_prereg = hashlib.sha256((DEST / "preregistration.json").read_bytes()).hexdigest()
    for item in rows:
        relative = str(item.get("file", ""))
        if not allowed.fullmatch(relative):
            raise RuntimeError("unsafe_artifact_name")
        if relative in {"preregistration.json", "sequences.json"}:
            target = DEST / relative
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("remote_frozen_input_mismatch")
            continue
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            "rsync", "-rlt", "--checksum", "--no-owner", "--no-group",
            "--no-perms", "-e", shlex.join(ssh()),
            f"root@{host}:{REMOTE}/{relative}", str(target),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if not (
            result.returncode == 0
            and target.is_file()
            and target.stat().st_size == int(item["bytes"])
            and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        ):
            raise RuntimeError("artifact_verification_failed")
    if hashlib.sha256((DEST / "preregistration.json").read_bytes()).hexdigest() != frozen_prereg:
        raise RuntimeError("local_preregistration_changed")
    exact = [item for item in rows if item["file"].startswith("fresh_")]
    if exact:
        # The remote watcher can observe COMPLETE between its final update and
        # loop-exit check.  Reconstruct the authoritative timings from the same
        # exact/wave mtimes returned by remote_status before frozen analysis.
        timing_rows = status["draws"]
        if len(timing_rows) != 8 or not all(row.get("complete") for row in timing_rows):
            raise RuntimeError("incomplete_timing_set")
        metadata = json.loads((DEST / "run_metadata.json").read_text())
        first = float(timing_rows[0]["started_unix"])
        last = float(timing_rows[-1]["completed_unix"])
        metadata.update({
            "draw_timings": timing_rows,
            "first_wave_started_unix": first,
            "last_draw_completed_unix": last,
            "scoring_wall_seconds": last - first,
            "estimated_scoring_cost": (
                (last - first) / 3600.0 * float(metadata["hourly_cost"])
            ),
        })
        (DEST / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return {"verified_exact": len(exact), "verified_total": len(rows)}


def analyze() -> None:
    result = subprocess.run(
        [sys.executable, str(DEST / "analyze_fresh.py")],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800,
    )
    if result.returncode:
        raise RuntimeError("analysis_failed")
    if not (DEST / "analysis.json").is_file() or not (DEST / "REPORT.md").is_file():
        raise RuntimeError("analysis_artifacts_missing")


def keychain(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    if result.returncode:
        raise RuntimeError("credential_unavailable")
    return result.stdout.decode().strip()


def api(path: str, body: object | None = None, method: str = "GET") -> dict:
    credential = keychain(VAST_API_SERVICE)
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


def guarded_instance() -> dict:
    instances = api("instances/?owner=me").get("instances", [])
    matches = [item for item in instances if item.get("id") == INSTANCE_ID]
    if len(matches) != 1:
        raise RuntimeError("instance_not_found")
    instance = matches[0]
    if instance.get("label") != LABEL or instance.get("num_gpus") != GPU_COUNT:
        raise RuntimeError("instance_guard_failed")
    return instance


def resume() -> None:
    instance = guarded_instance()
    if instance.get("actual_status") == "running":
        write_state(stage="running")
        return
    result = api(
        "instances/" + str(INSTANCE_ID) + "/", {"state": "running"}, "PUT"
    )
    if result.get("success") is not True:
        raise RuntimeError("resume_failed")
    write_state(stage="resume_requested")


def stop() -> None:
    instance = guarded_instance()
    if instance.get("actual_status") in {"stopped", "exited"}:
        write_state(stage="stopped", instance_status=instance.get("actual_status"))
        return
    result = api(
        "instances/" + str(INSTANCE_ID) + "/", {"state": "stopped"}, "PUT"
    )
    if result.get("success") is not True:
        raise RuntimeError("stop_failed")
    write_state(stage="stop_requested")


def monitor() -> int:
    deadline = time.time() + 4 * 60 * 60
    write_state(stage="monitoring", deadline_unix=deadline)
    while time.time() < deadline:
        try:
            remote = remote_status()
            write_state(stage="monitoring", **remote)
            if remote["complete"]:
                summary = fetch()
                if summary["verified_exact"] != 8:
                    raise RuntimeError("incomplete_artifact_set")
                analyze()
                write_state(stage="analyzed", **summary)
                stop()
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
    parser.add_argument("operation", choices=("resume", "stop", "monitor", "status"))
    operation = parser.parse_args().operation
    if operation == "resume":
        resume()
    elif operation == "stop":
        stop()
    elif operation == "status":
        try:
            remote = remote_status()
        except Exception as error:
            remote = {"error_class": type(error).__name__}
        state = json.loads(STATE.read_text()) if STATE.is_file() else {}
        print(json.dumps({"remote": remote, "supervisor": state}, sort_keys=True))
    else:
        return monitor()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_state(stage="failed_box_held", error_class=type(error).__name__)
        print(json.dumps({"event": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1)
