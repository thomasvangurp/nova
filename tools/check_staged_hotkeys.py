#!/usr/bin/env python3
"""Does the staging directory hold exactly the hotkeys the chain says are ours?

ansible/nanobody_services.yml globs the staging directory and asserts
``set(staged) == set(owned)`` on the box, after the repository has been synced
and the runtime built. Finding out there that one keyfile is missing wastes the
deploy; worse, the three keys currently staged are DEREGISTERED, so the failure
message would be a count mismatch rather than anything that names the problem.

This answers the same question here, in a second, and names every uid.

It never reads a private keyfile. Addresses come from the ``*pub.txt`` beside
each key and from ``coldkeypub.txt`` -- both are public halves -- and a private
key is only ever tested for existence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_STAGING = Path.home() / '.nova-hotkeys'
DEFAULT_WALLET = Path.home() / '.bittensor/wallets/mining_ops'


def address_of(path: Path) -> str | None:
    """The ss58 out of a bittensor public-key file, or None if it is not one."""
    try:
        return str(json.loads(path.read_text())['ss58Address'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def staged(directory: Path) -> dict[str, str]:
    """{address: hotkey name} for every staged key whose private half is present.

    A ``*pub.txt`` with no key beside it is a leftover, not a stageable hotkey,
    and would make the box's own count assert fail for a reason the operator
    cannot see from here.
    """
    found: dict[str, str] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob('*pub.txt')):
        name = path.name[: -len('pub.txt')]
        address = address_of(path)
        if address and (directory / name).is_file():
            found[address] = name
    return found


def registered(coldkey: str, netuid: int, network: str) -> dict[str, int]:
    """{hotkey address: uid} for the neurons this coldkey holds, from the chain."""
    import bittensor as bt  # noqa: PLC0415 -- absent on hosts that only stage

    metagraph = bt.Subtensor(network=network).metagraph(netuid, lite=True)
    return {str(hotkey): int(uid)
            for uid, hotkey, holder in zip(metagraph.uids, metagraph.hotkeys,
                                           metagraph.coldkeys)
            if str(holder) == coldkey}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--staging', default=str(DEFAULT_STAGING))
    parser.add_argument('--wallet', default=str(DEFAULT_WALLET),
                        help='wallet holding coldkeypub.txt; the PUBLIC half only')
    parser.add_argument('--netuid', type=int, default=68)
    parser.add_argument('--network', default='finney')
    arguments = parser.parse_args(argv)

    coldkey = address_of(Path(arguments.wallet) / 'coldkeypub.txt')
    if coldkey is None:
        print(f'no readable coldkeypub.txt under {arguments.wallet}', file=sys.stderr)
        return 2
    here = staged(Path(arguments.staging))
    try:
        chain = registered(coldkey, arguments.netuid, arguments.network)
    except ImportError:
        print('bittensor is not importable here; run this with an interpreter '
              'that has it', file=sys.stderr)
        return 2

    missing = sorted(uid for address, uid in chain.items() if address not in here)
    extra = sorted(name for address, name in here.items() if address not in chain)

    print(f'coldkey holds {len(chain)} registered hotkey(s) on netuid '
          f'{arguments.netuid}: {sorted(chain.values())}')
    print(f'staged in {arguments.staging}: {len(here)}')
    if missing:
        print(f'MISSING keyfiles for {len(missing)} registered uid(s): {missing}')
    if extra:
        # Deregistered slots. Left in place they make the box's assert fail on a
        # count, which reads as "you staged too few" rather than "these are dead".
        print(f'STAGED BUT NOT REGISTERED (remove these): {extra}')
    if missing or extra:
        print('NOT READY: the playbook asserts staged == registered')
        return 1
    print(f'READY: all {len(chain)} registered hotkeys are staged')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
