#!/usr/bin/env python3
"""Ansible dynamic inventory for the SN68 nanobody boxes, keyed on a vast.ai label.

Why this exists rather than an ``-i 'ssh1.vast.ai,' -e ansible_port=21300``
invocation: vast.ai rotates the SSH endpoint whenever an instance restarts, and
on 2026-09-12 the port moved from 21300 to 20868 with no notification. Every
command aimed at the old port failed with a connection timeout, which is
indistinguishable from a dead box, and the wrong conclusion -- "every box is
down" -- was reached and acted on. A written-down port is therefore not a
convenience, it is a latent false outage. The endpoint is only ever true at the
moment ``vastai show instances`` is read, so that is where it is read from.

The instance is selected by its **label**, not by its id or its endpoint,
because the label is the one identifier an operator chooses and a rebuild
preserves. Set it when renting:

    vastai create instance <offer> --label nova-nanobody-pixi ...
    vastai label instance <id> nova-nanobody-pixi     # on an existing box

Usage::

    ANSIBLE_CONFIG=ansible/ansible.cfg ansible-playbook \\
      -i ansible/inventory/vast_nanobody.py ansible/site.yml

Groups produced:

``nanobody_producer``
    Every matching instance. This is the GPU box: the design/scoring producer,
    the exact scorer, and the chain publisher all run here.
``nanobody_publisher``
    The same hosts under a role name, so a play can say what it is acting on.
``slot_manager``
    Exactly one host, ``operator``, with ``ansible_connection=local``. The slot
    purchase agent needs the coldkey and a macOS Keychain, so it runs on the
    operator's own machine and never on rented hardware. It is in the inventory
    unconditionally -- keeping >= 9 hotkeys registered is exactly the work that
    must continue while no GPU box exists, which was the situation on the day
    this was written.

Environment overrides, all optional:

``NOVA_VAST_LABEL``
    Instance label to select. Default ``nova-nanobody-pixi``.
``VASTAI_BIN``
    Path to the CLI. Default ``/opt/homebrew/bin/vastai``, then ``vastai`` on
    ``PATH``.
``NOVA_VAST_INSTANCES_FILE``
    Read a saved ``vastai show instances --raw`` document instead of calling the
    CLI. This is how the unit tests exercise endpoint rotation without an
    account, and how an operator can reproduce a past inventory.
``NOVA_VAST_ANY_STATUS``
    ``1`` keeps instances whose ``actual_status`` is not ``running``. Off by
    default: a loading or exited instance has no usable endpoint, and including
    it turns one broken host into a failed play for all of them.

Only an allowlist of instance fields becomes host variables. The vast.ai
instance document also carries ``jupyter_token`` and ssh key material, and host
variables are printed by ``ansible-inventory``, echoed in ``-vvv`` output, and
templated into task arguments. None of that may carry a credential, so the
allowlist is the mechanism rather than a redaction pass afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any

DEFAULT_LABEL = "nova-nanobody-pixi"
DEFAULT_VASTAI_BIN = "/opt/homebrew/bin/vastai"

# Instance fields that are safe to publish as host variables, mapped to the
# variable name. Anything absent from this mapping never leaves this script.
SAFE_FIELDS = {
    "id": "vast_instance_id",
    "label": "vast_label",
    "actual_status": "vast_actual_status",
    "num_gpus": "vast_num_gpus",
    "gpu_name": "vast_gpu_name",
    "machine_id": "vast_machine_id",
}


def _vastai_binary() -> str:
    override = os.environ.get("VASTAI_BIN")
    if override:
        return override
    if os.path.isfile(DEFAULT_VASTAI_BIN) and os.access(DEFAULT_VASTAI_BIN, os.X_OK):
        return DEFAULT_VASTAI_BIN
    found = shutil.which("vastai")
    if found:
        return found
    raise SystemExit(
        "vastai CLI not found; install it or set VASTAI_BIN to its path "
        f"(looked at {DEFAULT_VASTAI_BIN} and on PATH)"
    )


def read_instances() -> list[dict[str, Any]]:
    """Return the raw instance documents, from the CLI or a saved document.

    The CLI is asked for ``--raw`` JSON and its stdout is parsed here rather
    than piped through a shell filter: a truncated or non-JSON body has to be a
    diagnosable error, not a silently empty filter result.
    """
    saved = os.environ.get("NOVA_VAST_INSTANCES_FILE")
    if saved:
        with open(saved, encoding="utf-8") as handle:
            body = handle.read()
    else:
        completed = subprocess.run(
            [_vastai_binary(), "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            # stderr is reported, not swallowed: an expired or missing API key
            # otherwise looks exactly like "you rent nothing".
            raise SystemExit(
                "vastai show instances failed with exit status "
                f"{completed.returncode}: {completed.stderr.strip()[:400]}"
            )
        body = completed.stdout
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"vastai did not return JSON ({error}); first 200 bytes: {body[:200]!r}"
        ) from None
    if isinstance(payload, dict):
        payload = payload.get("instances", [])
    if not isinstance(payload, list):
        raise SystemExit(f"unexpected vastai payload type: {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def endpoint(instance: dict[str, Any]) -> tuple[str | None, int | None]:
    """Resolve the instance's current SSH host and port.

    ``ssh_host``/``ssh_port`` is the vast.ai proxy pair and is what the CLI
    prints; it is preferred because it works for instances without a public
    address. A directly addressable instance publishes its mapped port under
    ``ports`` instead, so that is the documented fallback rather than a guess.
    """
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if host and port:
        return str(host), int(port)
    host = instance.get("public_ipaddr")
    mapped = (instance.get("ports") or {}).get("22/tcp") or []
    for entry in mapped:
        if isinstance(entry, dict) and entry.get("HostPort"):
            return (str(host) if host else None), int(entry["HostPort"])
    return (str(host) if host else None), None


def select(instances: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    any_status = os.environ.get("NOVA_VAST_ANY_STATUS", "0") == "1"
    chosen = []
    for instance in instances:
        if str(instance.get("label") or "") != label:
            continue
        if not any_status and str(instance.get("actual_status") or "") != "running":
            print(
                f"skipping instance {instance.get('id')}: actual_status="
                f"{instance.get('actual_status')!r}; set NOVA_VAST_ANY_STATUS=1 "
                "to include it",
                file=sys.stderr,
            )
            continue
        chosen.append(instance)
    return chosen


def build_inventory() -> dict[str, Any]:
    label = os.environ.get("NOVA_VAST_LABEL", DEFAULT_LABEL)
    matching = select(read_instances(), label)

    hostvars: dict[str, dict[str, Any]] = {}
    producers: list[str] = []
    for instance in matching:
        host, port = endpoint(instance)
        identifier = instance.get("id")
        name = f"vast{identifier}" if identifier is not None else "vast-unlabelled"
        if not host or not port:
            # A running instance with no resolvable endpoint is reported and
            # dropped rather than added with a null port, which would surface
            # much later as an unreadable connection error.
            print(
                f"skipping instance {identifier}: no resolvable SSH endpoint "
                f"(ssh_host={instance.get('ssh_host')!r} "
                f"ssh_port={instance.get('ssh_port')!r})",
                file=sys.stderr,
            )
            continue
        variables: dict[str, Any] = {
            "ansible_host": host,
            "ansible_port": port,
            "ansible_user": "root",
        }
        for field, variable in SAFE_FIELDS.items():
            if field in instance:
                variables[variable] = instance[field]
        # The publisher fails closed on an unexpected GPU count and the producer
        # sizes its shards from it, so it is derived from the rental rather than
        # written into a vars file where it can disagree with the hardware.
        if instance.get("num_gpus") is not None:
            variables["gpu_count"] = int(instance["num_gpus"])
        hostvars[name] = variables
        producers.append(name)

    if not producers:
        print(
            f"no running vast.ai instance is labelled {label!r}; the GPU plays "
            "will fail their host assertion. Label the box with "
            f"`vastai label instance <id> {label}` or set NOVA_VAST_LABEL.",
            file=sys.stderr,
        )

    hostvars["operator"] = {
        "ansible_connection": "local",
        "ansible_host": "127.0.0.1",
        "ansible_python_interpreter": sys.executable,
    }

    # The purchase host. It is an always-on node rather than the operator's
    # laptop because the recycle price halves every 360 blocks, so buying cheaply
    # means being awake for the quiet intervals -- a machine that sleeps buys at
    # the top of the decay curve. NOVA_SLOT_HOST selects it; setting it to
    # "operator" moves the agent back onto this machine, which is the right
    # answer only while the always-on node is unavailable.
    slot_host = os.environ.get("NOVA_SLOT_HOST", DEFAULT_SLOT_HOST).strip()
    if slot_host != "operator":
        hostvars[slot_host] = {
            "ansible_host": os.environ.get("NOVA_SLOT_ADDRESS", slot_host).strip(),
            "ansible_user": os.environ.get("NOVA_SLOT_USER", "root").strip(),
            "ansible_python_interpreter": os.environ.get(
                "NOVA_SLOT_PYTHON", "/usr/bin/python3"
            ).strip(),
        }

    return {
        "_meta": {"hostvars": hostvars},
        "all": {"children": ["nanobody_producer", "nanobody_publisher", "slot_manager"]},
        "nanobody_producer": {"hosts": producers},
        "nanobody_publisher": {"hosts": list(producers)},
        "slot_manager": {"hosts": [slot_host]},
    }


# The Raspberry Pi on the tailnet. Reachable by tailscale name, so it follows the
# device rather than an address, and overridable for a different purchase host.
DEFAULT_SLOT_HOST = "raspberrypi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="emit the whole inventory")
    group.add_argument("--host", help="emit one host's variables")
    arguments = parser.parse_args()

    inventory = build_inventory()
    if arguments.host:
        print(json.dumps(inventory["_meta"]["hostvars"].get(arguments.host, {}), indent=2))
        return
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
