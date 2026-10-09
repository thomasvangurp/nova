#!/usr/bin/env python3
"""Print a wallet hotkey name that is registered on the live metagraph.

The miner looks its own uid up with ``metagraph.hotkeys.index(...)`` and the
setup handler re-raises, so a deregistered signing hotkey does not degrade the
miner -- it stops it from starting at all. On 2026-09-21 nova68-slot-19 had
deregistered while the miner was running; the process survived only because it
had resolved its uid once, at startup, against an older metagraph. Any restart
would have been fatal, and a supervisor that restarted it would have produced a
crash loop instead of a recovery.

Resolving here against the live chain makes a dereg survivable: the miner comes
back signing with whatever hotkey is currently registered.

Deliberately NOT resolved from /root/artifacts/chain_state.json. That file is
written by the miner, so consulting it to decide how to start the miner is
circular, and a frozen roster would hand back the very hotkey that cannot sign.

Only the hotkey NAME is ever written to stdout. Wallet files hold private key
material; this parses one field out of them in-process and emits nothing else.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys


def _local_hotkeys(wallet_path: str, wallet_name: str) -> dict[str, str]:
    """Map hotkey name -> ss58 address, reading nothing else out of the file."""
    found: dict[str, str] = {}
    pattern = os.path.join(wallet_path, wallet_name, "hotkeys", "*")
    for path in sorted(glob.glob(pattern)):
        if not os.path.isfile(path) or path.endswith("pub.txt"):
            continue
        try:
            with open(path) as handle:
                address = json.load(handle).get("ss58Address")
        except Exception:
            continue
        if isinstance(address, str) and address:
            found[os.path.basename(path)] = address
    return found


async def _registered(network: str, netuid: int) -> set[str]:
    import bittensor as bt

    # Same accessor shim the miner uses: the attribute was renamed between
    # bittensor releases and picking the wrong one fails at import time, which
    # would leave this returning "chain unreachable" for a local problem.
    factory = getattr(bt, "AsyncSubtensor", None) or getattr(bt, "async_subtensor", None)
    if factory is None:
        raise RuntimeError("bittensor exposes neither AsyncSubtensor nor async_subtensor")

    subtensor = factory(network=network)
    await subtensor.initialize()
    try:
        metagraph = await subtensor.metagraph(netuid)
        await metagraph.sync()
        return set(metagraph.hotkeys)
    finally:
        close = getattr(subtensor, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--wallet-path", default="/root/.bittensor/wallets")
    parser.add_argument("--wallet-name", default="mining_ops")
    parser.add_argument(
        "--prefer",
        default="",
        help="hotkey name to keep using when it is still registered, so a "
        "restart does not change signer for no reason",
    )
    args = parser.parse_args()

    local = _local_hotkeys(args.wallet_path, args.wallet_name)
    if not local:
        print(
            f"no hotkeys under {args.wallet_path}/{args.wallet_name}/hotkeys",
            file=sys.stderr,
        )
        return 3

    try:
        registered = asyncio.run(_registered(args.network, args.netuid))
    except Exception as error:  # chain unreachable: say so, do not guess
        print(f"could not read the metagraph: {error}", file=sys.stderr)
        return 4

    usable = [name for name, address in local.items() if address in registered]
    if not usable:
        print(
            f"none of the {len(local)} local hotkeys are registered on netuid "
            f"{args.netuid}; the miner cannot sign and must not be restarted "
            "into a crash loop",
            file=sys.stderr,
        )
        return 5

    if args.prefer in usable:
        print(args.prefer)
    else:
        if args.prefer:
            print(
                f"{args.prefer} is not registered; falling back to {usable[0]}",
                file=sys.stderr,
            )
        print(usable[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
