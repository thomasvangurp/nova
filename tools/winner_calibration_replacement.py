#!/usr/bin/env python3
"""Provision and destroy one guarded replacement 8xRTX4090 research host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "analysis/nanobody-gap/winner-calibration-ten-ticket-20260922"
MANIFEST = DEST / "replacement_instance.json"
INVENTORY = DEST / "replacement_inventory.json"
LABEL = "nova-winner-calibration-8x4090-20260922"
GPU_COUNT = 8
GPU_NAME = "RTX 4090"
MAXIMUM_DPH = 4.75
MINIMUM_RELIABILITY = 0.99
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"
IMAGE = "vastai/base-image:cuda-13.2.1-auto"


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
        return response.status, json.load(response)


def acceptable(offer: dict) -> bool:
    driver = str(offer.get("driver_version") or "")
    return (
        offer.get("gpu_name") == GPU_NAME
        and offer.get("num_gpus") == GPU_COUNT
        and offer.get("gpu_ram", 0) >= 23000
        # The bundles query enforces verified=true, but the compact offer row
        # omits that field (None) rather than echoing it.
        and offer.get("verified") in {None, True}
        and offer.get("rentable") is True
        and offer.get("rented") is False
        and offer.get("direct_port_count", 0) > 0
        and offer.get("cpu_cores_effective", 0) >= 64
        and offer.get("cpu_ram", 0) >= 120000
        and offer.get("disk_space", 0) >= 220
        and offer.get("reliability2", 0) >= MINIMUM_RELIABILITY
        and offer.get("dph_total", 999) <= MAXIMUM_DPH
        and re.fullmatch(r"[0-9.]+", driver) is not None
        and tuple(map(int, driver.split("."))) >= (570, 0, 0)
    )


def shortlist() -> list[dict]:
    _, result = api("bundles/", {
        "limit": 400,
        "type": "on-demand",
        "num_gpus": {"eq": GPU_COUNT},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_ram": {"gte": 23000},
        "disk_space": {"gte": 220},
    }, "POST")
    offers = [offer for offer in result.get("offers", []) if acceptable(offer)]
    offers.sort(key=lambda offer: (-offer["reliability2"], offer["dph_total"]))
    return offers


def saved() -> dict:
    payload = json.loads(MANIFEST.read_text())
    if payload.get("label") != LABEL or payload.get("gpu_count") != GPU_COUNT:
        raise RuntimeError("replacement_manifest_guard_failed")
    return payload


def instance() -> dict:
    manifest = saved()
    _, result = api("instances/?owner=me")
    matches = [item for item in result.get("instances", []) if item.get("id") == manifest["id"]]
    if len(matches) != 1:
        raise RuntimeError("replacement_instance_not_found")
    host = matches[0]
    if host.get("label") != LABEL or host.get("num_gpus") != GPU_COUNT or host.get("gpu_name") != GPU_NAME:
        raise RuntimeError("replacement_instance_guard_failed")
    return host


def create() -> int:
    if MANIFEST.exists():
        host = instance()
        print(json.dumps({"event": "existing", "id": host["id"], "status": host.get("actual_status")}))
        return 0
    _, current = api("instances/?owner=me")
    if any(item.get("label") == LABEL for item in current.get("instances", [])):
        raise RuntimeError("existing_label_requires_review")
    for offer in shortlist():
        status, result = api("asks/" + str(offer["id"]) + "/", {
            "image": IMAGE,
            "label": LABEL,
            "disk": 250,
            "runtype": "ssh_direct",
            "target_state": "running",
            "cancel_unavail": True,
        }, "PUT")
        if result.get("success") is not True or not isinstance(result.get("new_contract"), int):
            print(json.dumps({"event": "offer_declined", "offer": offer["id"], "http_status": status}))
            continue
        record = {
            "id": result["new_contract"],
            "label": LABEL,
            "offer_id": offer["id"],
            "gpu_count": GPU_COUNT,
            "gpu_name": GPU_NAME,
            "quoted_dph": offer["dph_total"],
            "reliability": offer["reliability2"],
            "created_unix": time.time(),
            "off_chain": True,
            "live_submission_authorized": False,
            "purpose": "frozen offline historical calibration and ten-ticket A/B",
        }
        MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: record[key] for key in ("id", "label", "gpu_count", "gpu_name", "quoted_dph", "reliability")}))
        return 0
    print(json.dumps({"event": "no_guarded_offer"}))
    return 1


def inventory() -> int:
    host = instance()
    ports = (host.get("ports") or {}).get("22/tcp") or []
    port = next((row.get("HostPort") for row in ports if row.get("HostPort")), None)
    if host.get("actual_status") != "running" or not port:
        raise RuntimeError("replacement_ssh_not_ready")
    hostvars = {
            "ansible_host": host.get("public_ipaddr"),
            "ansible_port": int(port),
            "ansible_user": "root",
            "ansible_ssh_private_key_file": SSH_KEY,
            "ansible_ssh_common_args": (
                "-F /dev/null -o ProxyCommand=none -o ProxyJump=none "
                "-o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                "-o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o LogLevel=ERROR"
            ),
    }
    payload = {"all": {"children": {
        "winner_calibration_worker": {"hosts": {"winner-calibration-replacement": hostvars}},
        "hamming2_worker": {"hosts": {"winner-calibration-replacement": hostvars}},
    }}}
    INVENTORY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "inventory_written", "id": host["id"], "status": host.get("actual_status"), "ssh_ready": True}))
    return 0


def status() -> int:
    host = instance()
    manifest = saved()
    print(json.dumps({
        "id": host.get("id"), "label": host.get("label"),
        "actual_status": host.get("actual_status"), "cur_state": host.get("cur_state"),
        "gpu_name": host.get("gpu_name"), "num_gpus": host.get("num_gpus"),
        "quoted_dph": manifest.get("quoted_dph"),
        "ssh_ready": bool((host.get("ports") or {}).get("22/tcp")),
    }, sort_keys=True))
    return 0


def destroy() -> int:
    host = instance()
    status, result = api("instances/" + str(host["id"]) + "/", None, "DELETE")
    passed = result.get("success") is True
    print(json.dumps({"event": "destroy_requested", "id": host["id"], "http_status": status, "passed": passed}))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("create", "status", "inventory", "destroy"))
    operation = parser.parse_args().operation
    return {"create": create, "status": status, "inventory": inventory, "destroy": destroy}[operation]()


if __name__ == "__main__":
    raise SystemExit(main())
