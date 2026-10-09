#!/usr/bin/env python3
"""Print the name of a wallet hotkey that OUR COLDKEY currently holds on-chain.

The miner looks its own uid up with ``metagraph.hotkeys.index(...)`` and the
setup handler re-raises, so a deregistered signing hotkey does not degrade the
miner -- it stops it from starting at all. On 2026-09-21 nova68-slot-19 had
deregistered while the miner was running; the process survived only because it
had resolved its uid once, at startup, against an older metagraph. Any restart
would have been fatal, and a supervisor that restarted it would have produced a
crash loop instead of a recovery.

Identity comes from the coldkey, never from a slot name. Until 2026-10-09 the
relaunch script passed a hardcoded ``--prefer nova68-slot-330``; slot-330 then
deregistered and the script kept "preferring" it, silently falling back to
whichever registered keyfile sorted first by name. Slot names are disposable --
every registration burns one -- while the coldkey is the one identifier that
does not change, so candidates are the local keyfiles whose address the LIVE
metagraph lists under ``--coldkey``.

Selection among several valid candidates, in order:

1. ``--prefer`` -- an explicit operator override (NOVA_MINER_HOTKEY). Honoured
   only while it is still ours and registered; otherwise ignored, loudly.
2. ``--sticky-file`` -- the hotkey this script chose last time, so a routine
   restart does not change signer for no reason. Re-validated against the live
   chain on every call, so it cannot go stale the way a hardcoded name did.
3. Otherwise the candidate with the highest emission, tie-broken by lowest uid.
   Subtensor prunes the lowest-emission non-immune neuron when a registration
   needs a slot, so the highest-emission key is the one least likely to be the
   next deregistration -- i.e. the choice that minimises future signer changes.

Deliberately NOT resolved from /root/artifacts/chain_state.json. That file is
written by the miner, so consulting it to decide how to start the miner is
circular, and a frozen roster would hand back the very hotkey that cannot sign.

Only the hotkey NAME is ever written to stdout. Addresses are read from the
public ``<name>pub.txt`` beside each key; the private keyfile is parsed for its
ss58Address field only when no public half exists, and nothing else is emitted.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from typing import NamedTuple

DEFAULT_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"


class Neuron(NamedTuple):
    uid: int
    coldkey: str
    emission: float


def _address(path: str) -> str | None:
    try:
        with open(path) as handle:
            address = json.load(handle).get("ss58Address")
    except Exception:
        return None
    return address if isinstance(address, str) and address else None


def _local_hotkeys(wallet_path: str, wallet_name: str) -> dict[str, str]:
    """Map hotkey name -> ss58 address, preferring the public half of each key."""
    found: dict[str, str] = {}
    pattern = os.path.join(wallet_path, wallet_name, "hotkeys", "*")
    for path in sorted(glob.glob(pattern)):
        if not os.path.isfile(path) or path.endswith("pub.txt"):
            continue
        public = path + "pub.txt"
        address = _address(public) if os.path.isfile(public) else _address(path)
        if address:
            found[os.path.basename(path)] = address
    return found


async def _metagraph(network: str, netuid: int) -> dict[str, Neuron]:
    """{hotkey address: Neuron} for every registered neuron on the live chain."""
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
        coldkeys = list(metagraph.coldkeys)
        if len(coldkeys) != len(metagraph.hotkeys):
            # Without coldkeys nothing can be attributed to us; refuse rather
            # than return an empty fleet that reads as "all deregistered".
            raise RuntimeError("metagraph returned hotkeys without matching coldkeys")
        raw_emission = getattr(metagraph, "emission", None)
        emission = [] if raw_emission is None else list(raw_emission)
        return {
            str(hotkey): Neuron(
                uid=int(uid),
                coldkey=str(coldkey),
                emission=float(emission[i]) if i < len(emission) else 0.0,
            )
            for i, (uid, hotkey, coldkey) in enumerate(
                zip(metagraph.uids, metagraph.hotkeys, coldkeys)
            )
        }
    finally:
        close = getattr(subtensor, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def candidates(
    local: dict[str, str], chain: dict[str, Neuron], coldkey: str
) -> dict[str, Neuron]:
    """Local hotkey name -> Neuron, for keys registered under ``coldkey`` now."""
    return {
        name: chain[address]
        for name, address in local.items()
        if address in chain and chain[address].coldkey == coldkey
    }


def choose(usable: dict[str, Neuron], prefer: str, sticky: str) -> tuple[str, str]:
    """(hotkey name, reason). ``usable`` must be non-empty."""
    if prefer and prefer in usable:
        return prefer, "operator override"
    if sticky and sticky in usable:
        return sticky, "previous signer still registered"
    name = min(usable, key=lambda n: (-usable[n].emission, usable[n].uid))
    return name, "highest emission (least likely to be pruned next)"


def _read_sticky(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _write_sticky(path: str, name: str) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            handle.write(name + "\n")
        os.replace(tmp, path)
    except OSError as error:
        print(f"could not record signer in {path}: {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--wallet-path", default="/root/.bittensor/wallets")
    parser.add_argument("--wallet-name", default="mining_ops")
    parser.add_argument(
        "--coldkey",
        default=os.environ.get("NOVA_COLDKEY_SS58") or DEFAULT_COLDKEY,
        help="ss58 of the coldkey whose registered hotkeys may sign",
    )
    parser.add_argument(
        "--prefer",
        default="",
        help="explicit override; used only while it is ours and registered",
    )
    parser.add_argument(
        "--sticky-file",
        default="",
        help="remembers the last choice so restarts keep the same signer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not update --sticky-file; report candidates on stderr",
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
        chain = asyncio.run(_metagraph(args.network, args.netuid))
    except Exception as error:  # chain unreachable: say so, do not guess
        print(f"could not read the metagraph: {error}", file=sys.stderr)
        return 4

    usable = candidates(local, chain, args.coldkey)
    if not usable:
        print(
            f"none of the {len(local)} local hotkeys are registered under coldkey "
            f"{args.coldkey} on netuid {args.netuid}; the miner cannot start and "
            "must not be restarted into a crash loop",
            file=sys.stderr,
        )
        return 5

    sticky = _read_sticky(args.sticky_file) if args.sticky_file else ""
    for label, name in (("--prefer", args.prefer), ("sticky", sticky)):
        if name and name not in usable:
            print(
                f"{label} {name} is not registered under our coldkey; ignoring it",
                file=sys.stderr,
            )
    name, reason = choose(usable, args.prefer, sticky)
    print(
        f"signer {name} uid={usable[name].uid} ({reason}); "
        f"{len(usable)} candidates: "
        + ", ".join(f"{n}=uid{usable[n].uid}" for n in sorted(usable, key=lambda n: usable[n].uid)),
        file=sys.stderr,
    )
    if args.sticky_file and not args.dry_run:
        _write_sticky(args.sticky_file, name)
    print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
