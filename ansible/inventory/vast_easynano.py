#!/usr/bin/env python3
"""Credential-safe dynamic inventory for the one isolated EasyNano rental."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import easynano_box_control as control


def main() -> int:
    host = control.instance()
    mapping = host.get("ports", {}).get("22/tcp") or []
    if not mapping:
        raise RuntimeError("ssh_not_ready")
    name = f"easynano-{host['id']}"
    inventory = {
        "easynano_worker": {"hosts": [name]},
        "_meta": {"hostvars": {name: {
            "ansible_host": host["public_ipaddr"],
            "ansible_port": int(mapping[0]["HostPort"]),
            "ansible_user": "root",
            "ansible_ssh_private_key_file": control.SSH_KEY,
            "vast_label": host["label"],
            "vast_num_gpus": host["num_gpus"],
            "vast_gpu_name": host["gpu_name"],
        }}},
    }
    print(json.dumps(inventory))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"inventory failed: {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
