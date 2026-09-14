#!/usr/bin/env python3
"""Print the hotkey name the miner should sign with, discovered from the chain.

Hardcoding the name is the stale-configuration failure this repository exists
to avoid: nova68-slot-1 was deregistered and the miner then spent every cycle
failing publish_metadata_extrinsic with "Custom error: 19".
"""
from __future__ import annotations

import json
import subprocess
import sys

REMOTE = r"""
import json, glob, os
state = json.load(open("/root/artifacts/chain_state.json"))
ours = {r["hotkey"] for r in state["neurons"] if r.get("coldkey") == "%s"}
for path in sorted(glob.glob("/root/.bittensor/wallets/mining_ops/hotkeys/*")):
    if path.endswith("pub.txt") or not os.path.isfile(path):
        continue
    try:
        if json.load(open(path)).get("ss58Address") in ours:
            print(os.path.basename(path))
            break
    except Exception:
        pass
"""


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: resolve_chain_writer.py <inventory> <coldkey>", file=sys.stderr)
        return 2
    inventory, coldkey = sys.argv[1], sys.argv[2]

    listing = subprocess.run(
        [inventory, "--list"], capture_output=True, text=True, check=True
    )
    hosts = json.loads(listing.stdout)["_meta"]["hostvars"]
    endpoint = next(
        (v for name, v in hosts.items() if name.startswith("vast")), None
    )
    if endpoint is None:
        print("no labelled vast instance in the inventory", file=sys.stderr)
        return 2

    probe = subprocess.run(
        [
            "ssh", "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(endpoint["ansible_port"]),
            f"root@{endpoint['ansible_host']}",
            "python3 -",
        ],
        input=REMOTE % coldkey,
        capture_output=True,
        text=True,
    )
    name = probe.stdout.strip().splitlines()[-1].strip() if probe.stdout.strip() else ""
    if not name:
        print("no local hotkey is registered to this coldkey", file=sys.stderr)
        return 1
    print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
