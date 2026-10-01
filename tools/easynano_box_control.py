#!/usr/bin/env python3
"""Guarded control for one non-production EasyNano P20809 GPU rental."""

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
DEST = REPO / "analysis/easynano-p20809/results"
MANIFEST = DEST / "instance-retry1.json"
LABEL = "nova-easynano-p20809-20260922-retry1"
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"
IMAGE = "vastai/base-image:cuda-13.2.1-auto"
LEASE_SECONDS = 4 * 60 * 60
GPU_NAMES = {"H100 SXM", "H100 PCIe", "A100 SXM4", "A100 PCIE", "RTX PRO 6000 WS"}


def keychain(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    if result.returncode:
        raise RuntimeError("credential_unavailable")
    return result.stdout.decode().strip()


def api(path: str, body: object | None = None, method: str = "GET") -> tuple[int, dict]:
    credential = keychain("vast.ai")
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/" + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json"},
    )
    credential = ""
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.load(response)
        status = response.status
    return status, data


def _saved() -> dict:
    data = json.loads(MANIFEST.read_text())
    if data.get("label") != LABEL or not isinstance(data.get("id"), int) or data.get("gpu_count") != 1:
        raise RuntimeError("wrong_manifest")
    return data


def instance() -> dict:
    saved = _saved()
    _, data = api("instances/?owner=me")
    matches = [item for item in data.get("instances", []) if item.get("id") == saved["id"]]
    if len(matches) != 1:
        raise RuntimeError("instance_not_found")
    host = matches[0]
    if host.get("label") != LABEL or host.get("num_gpus") != 1:
        raise RuntimeError("instance_guard_failed")
    return host


def create() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        host = instance()
        print(json.dumps({"event": "existing", "id": host["id"], "status": host.get("actual_status")}))
        return 0
    _, current = api("instances/?owner=me")
    if any(item.get("label") == LABEL for item in current.get("instances", [])):
        raise RuntimeError("existing_label_requires_review")
    query = {
        "limit": 100, "type": "on-demand", "num_gpus": {"eq": 1},
        "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
        "gpu_ram": {"gte": 80000}, "cpu_cores_effective": {"gte": 16},
        "cpu_ram": {"gte": 96000}, "disk_space": {"gte": 220},
    }
    _, result = api("bundles/", query, "POST")
    offers = []
    for offer in result.get("offers", []):
        driver = str(offer.get("driver_version", ""))
        if not re.fullmatch(r"[0-9.]+", driver):
            continue
        if (offer.get("gpu_name") in GPU_NAMES and offer.get("num_gpus") == 1
                and offer.get("gpu_ram", 0) >= 80000
                and offer.get("cpu_cores_effective", 0) >= 16
                and offer.get("cpu_ram", 0) >= 96000
                and offer.get("direct_port_count", 0) > 0
                and offer.get("dph_total", 999) <= 3.5
                and tuple(map(int, driver.split("."))) >= (570, 0, 0)):
            offers.append(offer)
    if not offers:
        raise RuntimeError("no_guarded_offer")
    offer = min(offers, key=lambda item: item["dph_total"])
    status, result = api("asks/" + str(offer["id"]) + "/", {
        "image": IMAGE, "label": LABEL, "disk": 220, "runtype": "ssh_direct",
        "target_state": "running", "cancel_unavail": True,
    }, "PUT")
    if result.get("success") is not True or not isinstance(result.get("new_contract"), int):
        print(json.dumps({"event": "create_failed", "http_status": status}))
        return 1
    record = {
        "id": result["new_contract"], "label": LABEL, "offer_id": offer["id"],
        "gpu_count": 1, "gpu_name": offer["gpu_name"],
        "gpu_ram_mb": offer["gpu_ram"], "quoted_dph": offer["dph_total"],
        "created_unix": time.time(), "lease_safety_seconds": LEASE_SECONDS,
        "off_chain": True, "live_submission_authorized": False,
    }
    MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record[k] for k in ("id", "label", "gpu_name", "gpu_ram_mb", "quoted_dph")}))
    return 0


def ssh(host: dict) -> list[str]:
    mapping = host.get("ports", {}).get("22/tcp") or []
    if not mapping:
        raise RuntimeError("ssh_not_ready")
    return [
        "ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(mapping[0]["HostPort"]),
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",
    ]


def remote(code: str, timeout: int = 60) -> dict:
    host = instance()
    result = subprocess.run(
        [*ssh(host), "root@" + host["public_ipaddr"], "python3 -c " + shlex.quote(code)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("remote_failed")
    data = json.loads(result.stdout)
    if not isinstance(data, dict):
        raise RuntimeError("invalid_remote_result")
    return data


def status() -> int:
    host = instance()
    safe = {key: host.get(key) for key in ("id", "label", "actual_status", "cur_state", "gpu_name", "num_gpus")}
    mapping = host.get("ports", {}).get("22/tcp") or []
    if mapping and host.get("actual_status") == "running":
        code = "import json;from pathlib import Path;p=Path('/root/easynano-run/output/progress.json');print(p.read_text() if p.is_file() else json.dumps({'stage':'not_started'}))"
        try:
            progress = remote(code)
            safe["progress"] = {key: progress.get(key) for key in (
                "stage", "completed", "expected", "last_run", "runs", "full_evaluations", "error_class"
            ) if key in progress}
        except Exception as error:
            safe["progress"] = {"stage": "unavailable", "error_class": type(error).__name__}
    print(json.dumps(safe, sort_keys=True))
    return 0


def start() -> int:
    code = r'''
import json, os, subprocess
from pathlib import Path
root=Path('/root/easynano-run');out=root/'output';out.mkdir(parents=True,exist_ok=True)
progress=out/'progress.json'
if progress.is_file() and json.loads(progress.read_text()).get('stage') in ['designing','evaluating','complete']:
 print(json.dumps({'event':'already_started','stage':json.loads(progress.read_text()).get('stage')}));raise SystemExit(0)
log=(out/'run.log').open('ab',buffering=0)
process=subprocess.Popen(['/usr/local/bin/pixi','run','--manifest-path',str(root/'pixi.toml'),'run-seeds'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(out/'runner.pid').write_text(str(process.pid)+'\n')
print(json.dumps({'event':'started','stage':'designing'}))
'''
    print(json.dumps(remote(code), sort_keys=True))
    return 0


def start_rerank() -> int:
    code = r'''
import json, subprocess, time
from pathlib import Path
root=Path('/root/easynano-run');out=root/'output';out.mkdir(parents=True,exist_ok=True)
(out/'progress.json').write_text(json.dumps({'stage':'mc-starting','updated_unix':time.time()})+'\n')
log=(out/'rerank.log').open('ab',buffering=0)
process=subprocess.Popen(['/usr/local/bin/pixi','run','--manifest-path',str(root/'pixi.toml'),'rerank-mc'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(out/'rerank.pid').write_text(str(process.pid)+'\n')
print(json.dumps({'event':'rerank_started','stage':'mc-starting'}))
'''
    print(json.dumps(remote(code), sort_keys=True))
    return 0


def fetch() -> int:
    host = instance()
    destination = DEST / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    code = r'''
import hashlib,json,re
from pathlib import Path
base=Path('/root/easynano-run/output');items=[]
for path in base.rglob('*.json'):
 rel=str(path.relative_to(base))
 if not re.fullmatch(r'[A-Za-z0-9_.\-/]+',rel):continue
 data=path.read_bytes();items.append({'file':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'files':items}))
'''
    manifest = remote(code, timeout=90)["files"]
    transport = shlex.join(ssh(host))
    for item in manifest:
        relative = item["file"]
        if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", relative) or ".." in Path(relative).parts:
            raise RuntimeError("unsafe_artifact_name")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            "rsync", "-rlt", "--checksum", "--no-owner", "--no-group", "--no-perms",
            "-e", transport, "root@" + host["public_ipaddr"] + ":/root/easynano-run/output/" + relative,
            str(target),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        verified = (result.returncode == 0 and target.is_file()
                    and target.stat().st_size == item["bytes"]
                    and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"])
        print(json.dumps({"event": "artifact", **item, "verified": verified}), flush=True)
        if not verified:
            return 1
    return 0


def stop() -> int:
    host = instance()
    if host.get("cur_state") == "stopped" and host.get("actual_status") in {"stopped", "exited"}:
        print(json.dumps({"event": "stopped", "id": host["id"]}))
        return 0
    status_code, result = api("instances/" + str(host["id"]) + "/", {"state": "stopped"}, "PUT")
    passed = result.get("success") is True
    print(json.dumps({"event": "stop_requested", "id": host["id"], "http_status": status_code, "passed": passed}))
    return 0 if passed else 1


def resume() -> int:
    host = instance()
    if host.get("cur_state") == "running" and host.get("actual_status") == "running":
        print(json.dumps({"event": "running", "id": host["id"]}))
        return 0
    status_code, result = api("instances/" + str(host["id"]) + "/", {"state": "running"}, "PUT")
    passed = result.get("success") is True
    print(json.dumps({"event": "resume_requested", "id": host["id"], "http_status": status_code, "passed": passed}))
    return 0 if passed else 1


def monitor() -> int:
    saved = _saved()
    deadline = saved["created_unix"] + saved["lease_safety_seconds"] - 120
    previous = None
    while time.time() < deadline:
        host = instance()
        if host.get("actual_status") in {"stopped", "exited"}:
            return 1
        try:
            progress = remote("import json;from pathlib import Path;p=Path('/root/easynano-run/output/progress.json');print(p.read_text() if p.is_file() else json.dumps({'stage':'not_started'}))")
        except Exception as error:
            progress = {"stage": "unavailable", "error_class": type(error).__name__}
        signature = (progress.get("stage"), progress.get("completed"), progress.get("last_run"))
        if signature != previous:
            print(json.dumps({"event": "progress", **{key: progress.get(key) for key in (
                "stage", "completed", "expected", "last_run", "runs", "full_evaluations", "error_class"
            ) if key in progress}}), flush=True)
            previous = signature
        if progress.get("stage") == "complete":
            try:
                return fetch()
            finally:
                stop()
        if progress.get("stage") == "failed":
            try:
                fetch()
            finally:
                stop()
            return 1
        time.sleep(20)
    try:
        fetch()
    finally:
        stop()
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("create", "status", "resume", "start", "start-rerank", "monitor", "fetch", "stop"))
    operation = parser.parse_args().operation
    return {"create": create, "status": status, "resume": resume, "start": start,
            "start-rerank": start_rerank, "monitor": monitor, "fetch": fetch,
            "stop": stop}[operation]()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"event": "failed", "error_class": type(error).__name__}))
        sys.exit(1)
