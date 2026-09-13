"""Commit one uid's submission, signed by that uid's own hotkey.

The publisher selects a distinct sequence per uid the coldkey holds, but a
commitment is signed by a wallet, and a wallet is one hotkey -- so committing for
uid N requires N's own key. A single writer process therefore submits for exactly
one uid, however many the fleet holds, and until this existed we prepared nine
submissions and sent one. Eight ninths of the benefit of a fleet, discarded at the
last step.

This is the per-uid half, invoked once per uid through the publisher's
``--commit-command`` seam.

Two outcomes are deliberately NOT failures, because a fleet is a moving target
and one uid's problem must not abort the others:

* a hotkey that no longer holds a uid -- deregistration is routine on a full
  subnet, and a stale local keyfile is the normal residue of it;
* "too soon to commit again" -- the chain rate-limits commitments per hotkey, and
  hitting that means a previous submission stands, not that this one broke.

Both exit zero with a reason on stdout. Anything else exits non-zero, because
silently not submitting is the failure this whole fleet exists to avoid.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Printed on stdout so a caller can tell a skip from a submission without
#: parsing prose.
RESULT_COMMITTED = "COMMIT_OK"
RESULT_STALE = "COMMIT_SKIPPED_STALE_HOTKEY"
RESULT_TOO_SOON = "COMMIT_SKIPPED_RATE_LIMITED"


def encoded_payload(message: str, uid: int, block: int, timelock) -> tuple[str, str]:
    """Timelock-encrypt one submission, returning (filename, base64 content).

    The encryption is keyed on the uid, which is why this cannot be done once for
    the whole fleet: each uid's payload is distinct even for an identical
    sequence.
    """
    encrypted = timelock.encrypt(uid, message, block)
    content = str(encrypted)
    encoded = base64.b64encode(content.encode()).decode()
    filename = hashlib.sha256(content.encode()).hexdigest()[:20]
    return filename, encoded


def holds_uid(neurons, hotkey_ss58: str, uid: int) -> bool:
    """Whether this hotkey still holds this uid, from a chain roster.

    Checked against the roster rather than trusted from the command line: uids are
    recycled, so a uid that was ours an epoch ago can belong to another operator
    now, and committing for it would sign someone else's slot.
    """
    for row in neurons:
        if int(row.get("uid", -1)) == int(uid):
            return str(row.get("hotkey")) == hotkey_ss58
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--hotkey", required=True, help="hotkey NAME in the wallet")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--wallet-name", default="mining_ops")
    parser.add_argument("--wallet-path", default="/root/.bittensor/wallets")
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--network", default="finney")
    parser.add_argument(
        "--github-path",
        default=None,
        help="repo path prefix for the uploaded payload; defaults to load_github_path()",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate everything, then stop before the extrinsic",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Imported here, not at module import time, so --help works in any
    # environment: the deploy gate runs it where bittensor may be absent.
    import bittensor as bt
    from bittensor.core.errors import MetadataError

    from btdr import QuicknetBittensorDrandTimelock
    from files import load_github_path, upload_file_to_github

    wallet_factory = getattr(bt, "Wallet", None) or bt.wallet
    wallet = wallet_factory(
        name=args.wallet_name, hotkey=args.hotkey, path=args.wallet_path
    )
    hotkey_ss58 = wallet.hotkey.ss58_address

    subtensor_factory = getattr(bt, "Subtensor", None) or bt.subtensor
    subtensor = subtensor_factory(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)
    neurons = [
        {"uid": index, "hotkey": str(hotkey)}
        for index, hotkey in enumerate(list(metagraph.hotkeys))
    ]

    if not holds_uid(neurons, hotkey_ss58, args.uid):
        # Routine, not broken: report and let the other uids proceed.
        print(f"{RESULT_STALE} uid={args.uid} hotkey={args.hotkey}")
        return 0

    block = subtensor.get_current_block()
    filename, encoded = encoded_payload(
        args.sequence, args.uid, block, QuicknetBittensorDrandTimelock()
    )
    github_path = args.github_path or load_github_path()
    commit_content = f"{github_path}/{filename}.txt"

    if args.dry_run:
        print(f"COMMIT_DRY_RUN uid={args.uid} block={block} content={commit_content}")
        return 0

    try:
        committed = subtensor.set_commitment(
            wallet=wallet, netuid=args.netuid, data=commit_content
        )
    except MetadataError:
        print(f"{RESULT_TOO_SOON} uid={args.uid} hotkey={args.hotkey}")
        return 0

    if not committed:
        print(f"COMMIT_REFUSED uid={args.uid} hotkey={args.hotkey}", file=sys.stderr)
        return 1

    # Upload only after the chain accepted it. The reverse order publishes a
    # payload no commitment points at, which is indistinguishable from spam.
    if not upload_file_to_github(filename, encoded):
        print(f"COMMIT_UPLOAD_FAILED uid={args.uid} content={commit_content}", file=sys.stderr)
        return 1

    print(f"{RESULT_COMMITTED} uid={args.uid} hotkey={args.hotkey} content={commit_content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
