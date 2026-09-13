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


# neurons/validator/commitments.py: SEPARATOR "," separates items within a
# section, NULL_TOKEN "~" is the empty list, and the two sections are joined by
# exactly one "|".
NULL_TOKEN = "~"
SUBMISSION_DELIMITER = "|"


def submission_message(sequence: str) -> str:
    """Format one nanobody submission the way the validator parses it.

    parse_decrypted_submission refuses anything that does not hold exactly one
    "|", separating the molecule section from the sequence section, and refuses
    whitespace and non-ascii outright. This driver used to encrypt the bare
    sequence -- zero delimiters -- so every uid we committed was discarded at
    parse time, before scoring, similarity or any validity check. It looked
    like a healthy fleet from our side: COMMIT_OK for every uid, and not one
    row on the leaderboard in any epoch. The miner has always sent "~|<seq>";
    the fleet driver did not.
    """
    text = str(sequence).strip().upper()
    if not text:
        raise ValueError("refusing to commit an empty sequence")
    if any(character.isspace() for character in text):
        raise ValueError("a submission may not contain whitespace")
    if not text.isascii():
        raise ValueError("a submission must be ascii")
    if SUBMISSION_DELIMITER in text:
        raise ValueError("a sequence may not contain the section delimiter")
    return f"{NULL_TOKEN}{SUBMISSION_DELIMITER}{text}"


def encoded_payload(message: str, uid: int, block: int, timelock) -> tuple[str, str]:
    """Timelock-encrypt one submission, returning (filename, base64 content).

    ``message`` must already be in the validator's wire format; build it with
    ``submission_message``.

    The encryption is keyed on the uid, which is why this cannot be done once for
    the whole fleet: each uid's payload is distinct even for an identical
    sequence.
    """
    encrypted = timelock.encrypt(uid, message, block)
    content = str(encrypted)
    encoded = base64.b64encode(content.encode()).decode()
    filename = hashlib.sha256(content.encode()).hexdigest()[:20]
    return filename, encoded


def _looks_like_address(value: str) -> str | bool:
    """Whether this is an ss58 address rather than a wallet key-file name.

    Addresses on this chain start with "5" and run to 48 characters; key files
    are named things like ``nova68-slot-1``. The length bound comes from
    ``fleet.MINIMUM_SS58_LENGTH``, which exists because a TRUNCATED address --
    the kind that appears in a log line or a copied-out summary -- matches no
    neuron row and would otherwise be treated as a name.
    """
    from nanobody_opt import fleet

    candidate = str(value).strip()
    return candidate.startswith("5") and len(candidate) >= fleet.MINIMUM_SS58_LENGTH


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
    # Not argparse-required: --self-test must run with no submission to make.
    # main() enforces them for every other invocation.
    parser.add_argument("--uid", type=int)
    parser.add_argument(
        "--hotkey",
        help="hotkey wallet NAME or ss58 address; an address is resolved to the "
        "local key file that holds it",
    )
    parser.add_argument("--sequence")
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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "import every dependency this program needs, then exit. The "
            "deploy runs it: --help proves nothing, because the imports are "
            "deferred into main()"
        ),
    )
    return parser


SELF_TEST_OK = "COMMIT_SELF_TEST_OK"


def _add_repository_to_path() -> None:
    """Put the deployed root and its utils/ on sys.path, as the miner does.

    btdr and files are not installed packages -- they are modules under
    /root/utils, and neurons/miner/miner.py appends both paths itself before
    importing them. This program did not, so every commit died on
    ``ModuleNotFoundError: No module named 'btdr'`` while the miner beside it
    worked, and the fleet committed nothing.
    """
    root = Path(__file__).resolve().parent.parent
    for entry in (root, root / "utils"):
        text = str(entry)
        if text not in sys.path:
            sys.path.append(text)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            name
            for name in ("uid", "hotkey", "sequence")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            parser.error(
                "the following arguments are required: "
                + ", ".join(f"--{name}" for name in missing)
            )
    _add_repository_to_path()

    # Imported here, not at module import time, so --help works in any
    # environment: the deploy gate runs it where bittensor may be absent.
    import bittensor as bt
    from bittensor.core.errors import MetadataError

    from btdr import QuicknetBittensorDrandTimelock
    from files import load_github_path, upload_file_to_github

    from nanobody_opt import fleet

    if args.self_test:
        # Resolving the path, not merely importing it: the import gate passed
        # on a box where GITHUB_REPO_* was unset, and every uid then died on
        # the resolution instead. Not a secret -- owner/repo/branch/path.
        print(f"github_path={load_github_path()}")
        print(SELF_TEST_OK)
        return 0

    # The publisher substitutes {hotkey} from the chain, so it arrives as an
    # ss58 address, while a wallet is constructed from a key FILE NAME. Accept
    # either: resolving here is the difference between submitting and a
    # confusing "keyfile not found" at the last step of an epoch.
    hotkey_name = args.hotkey
    if _looks_like_address(args.hotkey):
        addresses = fleet.local_hotkey_addresses(args.wallet_path, args.wallet_name)
        hotkey_name = addresses.get(args.hotkey)
        if hotkey_name is None:
            # An on-chain hotkey with no local key is the stale case seen from
            # the other side: the fleet moved and this box has not caught up.
            print(f"{RESULT_STALE} uid={args.uid} hotkey={args.hotkey} (no local key file)")
            return 0

    wallet_factory = getattr(bt, "Wallet", None) or bt.wallet
    wallet = wallet_factory(
        name=args.wallet_name, hotkey=hotkey_name, path=args.wallet_path
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
        print(f"{RESULT_STALE} uid={args.uid} hotkey={hotkey_name}")
        return 0

    block = subtensor.get_current_block()
    filename, encoded = encoded_payload(
        submission_message(args.sequence),
        args.uid,
        block,
        QuicknetBittensorDrandTimelock(),
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
        print(f"{RESULT_TOO_SOON} uid={args.uid} hotkey={hotkey_name}")
        return 0

    if not committed:
        print(f"COMMIT_REFUSED uid={args.uid} hotkey={hotkey_name}", file=sys.stderr)
        return 1

    # Upload only after the chain accepted it. The reverse order publishes a
    # payload no commitment points at, which is indistinguishable from spam.
    if not upload_file_to_github(filename, encoded):
        print(f"COMMIT_UPLOAD_FAILED uid={args.uid} content={commit_content}", file=sys.stderr)
        return 1

    print(f"{RESULT_COMMITTED} uid={args.uid} hotkey={hotkey_name} content={commit_content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
