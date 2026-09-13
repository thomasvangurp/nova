"""Publish from every uid our coldkey holds, one distinct sequence each.

Two measured facts shape this module.

The validator dedupes by sequence.  Identical sequences committed from different
hotkeys receive an identical ``final_score`` -- observed 211/211, 120/120 and
119/119 across ten of thirty-three epochs on subnet 68.  So k hotkeys are worth
something only if they carry k DISTINCT molecules; copying one winner k times
buys exactly one ticket in the validator's single draw.

uid numbers are RECYCLED on deregistration.  The miner published into "our uid
8" for twenty-one days after that slot had become somebody else's, and every
diagnostic keyed on the remembered uid dutifully followed the new owner.  The
identity that survives deregistration is the COLDKEY, so the fleet is discovered
from the chain's own neuron rows every epoch and never remembered, and no
configured hotkey list is accepted even as a fallback -- a stale list is the same
failure as the stale uid, with the same silence.

When the discovered set is empty that is not a quiet no-op: it is the
twenty-one-day bug happening again, and it must be impossible to miss.

This module deliberately has NO third-party imports, for the same reason
``publication_gates`` has none: the publisher runs in a different Pixi
environment from the scorer, and a shared source of truth that only one side can
import is not shared at all.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The validator's ordering, and the only one a publisher may use.  Only rank one
# pays and the validator draws once, so the quantity to maximize is the chance of
# a winning draw, not the best average.  See nanobody_opt/win_probability.py.
REQUIRED_SELECTION_ORDER = "win-probability"

# ss58 addresses are 47-49 characters.  A truncated key pasted from a dashboard
# ("5EkCwcNTfxeN...") would match nothing, and "matched nothing" is the failure
# this module exists to make loud, so it is rejected as a configuration error
# rather than reported as an empty fleet.
MINIMUM_SS58_LENGTH = 40

# 2 and 3 are the single-sequence checkpoint schemas this artifact wraps, so the
# fleet artifact takes the next number rather than a colliding one.
FLEET_ARTIFACT_SCHEMA_VERSION = 4

EMPTY_FLEET_BANNER = (
    "\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "NANOBODY_FLEET_EMPTY: no uid on netuid {netuid} belongs to coldkey {coldkey}.\n"
    "NOTHING WAS PUBLISHED. Every hotkey of this coldkey is deregistered, so\n"
    "there is no slot to commit into. Do not read a uid out of any cached file:\n"
    "uid numbers are recycled on deregistration, and publishing into a recycled\n"
    "uid is what let this miner run for 21 days scoring another miner's slot.\n"
    "Register a hotkey on netuid {netuid} and the next epoch will publish.\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
)

MISSING_KEYFILE_BANNER = (
    "\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "NANOBODY_FLEET_KEYFILE_MISSING: the chain says these uids are ours, but\n"
    "wallet {wallet} on this host holds no key for their hotkeys, so nothing can\n"
    "sign a commitment for them:\n"
    "{detail}\n"
    "This is a different failure from an empty fleet: the slot exists and is\n"
    "earning nothing. Install the hotkey or stop paying for the slot.\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
)


class EmptyFleetError(RuntimeError):
    """Raised when no registered uid belongs to our coldkey."""

    def __init__(self, coldkey: str, netuid: int, observed: int) -> None:
        self.coldkey = coldkey
        self.netuid = netuid
        self.observed = observed
        super().__init__(
            EMPTY_FLEET_BANNER.format(netuid=netuid, coldkey=coldkey)
            + f"scanned {observed} neuron rows and matched none.\n"
        )


class MissingHotkeyError(RuntimeError):
    """Raised when a uid we hold on chain has no signing key on this host."""

    def __init__(self, wallet: str, missing: Sequence["FleetMember"]) -> None:
        self.wallet = wallet
        self.missing = list(missing)
        detail = "\n".join(
            f"  uid {member.uid} hotkey {member.hotkey}" for member in missing
        )
        super().__init__(MISSING_KEYFILE_BANNER.format(wallet=wallet, detail=detail))


class SelectionOrderError(ValueError):
    """Raised when a ranking was not ordered by win probability."""


@dataclass(frozen=True)
class FleetMember:
    """A uid we currently hold, as the chain reports it this epoch."""

    uid: int
    hotkey: str


@dataclass(frozen=True)
class Assignment:
    """One distinct sequence bound to one uid, and the block it commits in."""

    uid: int
    hotkey: str
    sequence: str
    commit_block: int


def resolve_coldkey(
    value: str | None,
    *,
    wallet_root: Path | str | None = None,
    wallet_name: str | None = None,
) -> str:
    """Return our full coldkey ss58, from the flag or the local coldkeypub.

    ``coldkeypub.txt`` holds public key material only, and reading it locally
    avoids hand-copying a truncated address into a config file.  Only the
    ``ss58Address`` field is read, and nothing else from the file is retained.
    """
    candidate = (value or "").strip()
    if not candidate and wallet_name:
        root = Path(wallet_root or Path.home() / ".bittensor" / "wallets")
        candidate = _public_ss58(root / wallet_name / "coldkeypub.txt") or ""
    if not candidate:
        raise ValueError("our coldkey ss58 address is not configured")
    if "…" in candidate or "..." in candidate or len(candidate) < MINIMUM_SS58_LENGTH:
        raise ValueError(
            f"coldkey {candidate!r} is truncated; a partial address matches no "
            "neuron row and would be reported as an empty fleet"
        )
    return candidate


def _public_ss58(path: Path) -> str | None:
    """Extract only the public ss58 address from a wallet key file.

    Nothing else in the file is read out, returned, logged or kept.  The private
    field of a bittensor key file must never reach a log line, a command line or
    a model context, so this function is the only place that opens one and it
    returns exactly one public value.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    address = payload.get("ss58Address")
    if not isinstance(address, str) or len(address.strip()) < MINIMUM_SS58_LENGTH:
        return None
    return address.strip()


def local_hotkey_addresses(
    wallet_root: Path | str, wallet_name: str
) -> dict[str, str]:
    """Map each locally installed hotkey ss58 to its key file name.

    A hotkey file is named for the hotkey, not its address, so the address is
    taken from the file's public ``ss58Address`` field; a file whose name is
    itself an ss58 address is accepted without being opened at all.
    """
    directory = Path(wallet_root) / wallet_name / "hotkeys"
    addresses: dict[str, str] = {}
    try:
        entries = sorted(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return addresses
    for entry in entries:
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if len(entry.name) >= MINIMUM_SS58_LENGTH and entry.name.startswith("5"):
            addresses.setdefault(entry.name, entry.name)
            continue
        address = _public_ss58(entry)
        if address:
            addresses.setdefault(address, entry.name)
    return addresses


def require_local_hotkeys(
    fleet: Sequence[FleetMember], wallet_root: Path | str, wallet_name: str
) -> dict[str, str]:
    """Resolve every discovered hotkey to a local key file, or fail loudly.

    Chain discovery answers "which uids are ours"; this answers "which of them
    can this host actually sign for".  They are separate failures and are
    reported separately, because an unsignable registered slot is a paid-for
    hole rather than an absence of slots.
    """
    addresses = local_hotkey_addresses(wallet_root, wallet_name)
    missing = [member for member in fleet if member.hotkey not in addresses]
    if missing:
        raise MissingHotkeyError(wallet_name, missing)
    return {member.hotkey: addresses[member.hotkey] for member in fleet}


def neuron_rows(source: Any) -> list[dict[str, Any]]:
    """Normalize a chain view into ``uid``/``hotkey``/``coldkey`` rows.

    Accepts what the miner already has: a bittensor metagraph (parallel ``uids``,
    ``hotkeys``, ``coldkeys`` sequences), the ``neurons`` list the miner exports
    to its chain-state file, or that whole chain-state mapping.
    """
    if source is None:
        return []
    if isinstance(source, Mapping):
        return neuron_rows(source.get("neurons"))
    if hasattr(source, "hotkeys") and hasattr(source, "coldkeys"):
        hotkeys = list(source.hotkeys)
        coldkeys = list(source.coldkeys)
        uids = list(getattr(source, "uids", range(len(hotkeys))))
        if not len(uids) == len(hotkeys) == len(coldkeys):
            raise ValueError("metagraph uid/hotkey/coldkey columns disagree in length")
        return [
            {"uid": int(uid), "hotkey": str(hotkey), "coldkey": str(coldkey)}
            for uid, hotkey, coldkey in zip(uids, hotkeys, coldkeys)
        ]
    rows: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, Mapping):
            raise ValueError("neuron rows must be mappings")
        coldkey = item["coldkey"]
        # The exporter writes ``None`` when the metagraph's coldkeys column is
        # shorter than its hotkeys column, and documents that so the consumer can
        # tell a missing coldkey from somebody else's.  ``str()`` would turn the
        # absence into the literal "None", which matches nothing and therefore
        # silently DROPS the uid: a partly populated column shrinks the fleet we
        # publish from, and an absent one raises EmptyFleetError whose banner
        # tells the operator to register hotkeys they already hold.  The
        # metagraph branch above already refuses a column-length mismatch; so
        # does this one.
        if coldkey is None or not str(coldkey).strip():
            raise ValueError(
                f"neuron row for uid {item.get('uid')!r} carries no coldkey; the "
                "chain view is incomplete, which is not the same as the uid "
                "belonging to another miner"
            )
        rows.append({
            "uid": int(item["uid"]),
            "hotkey": str(item["hotkey"]).strip(),
            "coldkey": str(coldkey).strip(),
        })
    return rows


def discover_fleet(source: Any, coldkey: str, *, netuid: int = 68) -> list[FleetMember]:
    """Return every uid the chain currently attributes to ``coldkey``.

    Discovery is by coldkey and happens every epoch, because that is the only
    identity a deregistration cannot silently transfer to another miner.
    """
    address = resolve_coldkey(coldkey)
    rows = neuron_rows(source)
    members = sorted(
        (
            FleetMember(uid=int(row["uid"]), hotkey=str(row["hotkey"]))
            for row in rows
            if str(row["coldkey"]) == address
        ),
        key=lambda member: member.uid,
    )
    if not members:
        raise EmptyFleetError(address, netuid, len(rows))
    seen: set[int] = set()
    for member in members:
        if member.uid in seen:
            raise ValueError(f"chain view reports uid {member.uid} twice")
        seen.add(member.uid)
    return members


# Share of the fleet spent on the best EXPECTED score rather than on the best
# chance of a lucky draw. OFF by default, because variance seeking is the right
# objective here and buying the mean actively discards the candidates most
# likely to win: only rank one pays and the validator takes a single draw.
#
# Half, on measured evidence rather than theory. Across our five scored
# tickets in epoch 25099 the predicted p90 correlated +0.807 with the actual
# score while the win probability correlated only -0.526, and epoch 25101's
# pool held a candidate at p90 108.2 with confidence rank 82.2 that
# win-probability order ranked 27 times lower than its own top pick at p90
# 217.2 and confidence 175.1.
#
# Confidence is the whole of our gap to the winners -- we beat them on
# developability every epoch and usually on interaction -- and the best
# expected-score candidates are also the best-confidence ones. Half the fleet
# buys them; half still buys the tail, because only rank one pays and the
# validator draws once.
DEFAULT_VALUE_SHARE = 0.5

# Two tickets closer than this are one ticket. The validator discards a
# submission too similar to another, and measured over epochs 25093-25095 that
# is exactly what happened to us: every dropped uid was 0.9685-0.9843 identical
# to a uid that scored, while every surviving pair sat at 0.9449-0.9606. We
# committed seven and played three.
#
# 0.93 rather than the 0.95 the rule states, because this is a positional
# proxy: the validator aligns, and an alignment can only find MORE identity
# than position-for-position comparison of equal-length sequences. The margin
# is what keeps a proxy conservative.
MAXIMUM_FLEET_IDENTITY = 0.93


def sequence_identity(left: str, right: str) -> float:
    """Position-for-position identity, normalised by the longer sequence.

    Deliberately dependency-free: this module is imported in the chain
    environment, which has no alignment stack.
    """
    if not left or not right:
        return 0.0
    shared = sum(1 for a, b in zip(left, right) if a == b)
    return shared / max(len(left), len(right))


def _closest(sequence: str, chosen: Sequence[str]) -> float:
    return max((sequence_identity(sequence, other) for other in chosen), default=0.0)


#: Identity to a sequence the FIELD has recently submitted at which the
#: validator is likely to treat ours as the duplicate and score theirs instead.
FIELD_COLLISION_IDENTITY = 0.95


def uncrowded(
    candidates: Sequence[str],
    crowd: Sequence[str],
    limit: float = FIELD_COLLISION_IDENTITY,
) -> list[str]:
    """Candidates that no recent field submission sits on top of.

    Separation among our own tickets is not enough. In epoch 25099 our uid 12
    was 0.9528 identical to another miner's submission in the same epoch: the
    validator scored theirs and dropped ours, and we paid for a registration
    that produced nothing. Everyone is mutating the same winner, so the crowded
    region is precisely where our neighbourhood search wants to go.

    We cannot see the field's current epoch -- it is timelocked -- but what
    they submitted recently is public, and it is where they are still standing.
    """
    if not crowd:
        return list(candidates)
    return [
        sequence
        for sequence in candidates
        if _closest(sequence, crowd) < limit
    ]
VALUE_KEY = "score_p90"


def _best_by_expected_score(
    candidates: Sequence[Any], wanted: int, taken: set[str], limit: float
) -> list[str]:
    """Lowest predicted p90 rank sum first; lower is better, ties keep order."""
    scored: list[tuple[float, int, str]] = []
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        sequence = str(candidate.get("sequence") or "").strip().upper()
        value = candidate.get(VALUE_KEY)
        if not sequence or sequence in taken or not isinstance(value, (int, float)):
            continue
        scored.append((float(value), position, sequence))
    scored.sort()
    chosen: list[str] = []
    for _, _, sequence in scored:
        if sequence in taken:
            continue
        taken.add(sequence)
        chosen.append(sequence)
        if len(chosen) == wanted:
            break
    return chosen


def distinct_sequences(
    rankings: Mapping[str, Any],
    count: int,
    *,
    required_order: str = REQUIRED_SELECTION_ORDER,
    value_share: float = DEFAULT_VALUE_SHARE,
    maximum_identity: float = MAXIMUM_FLEET_IDENTITY,
    crowd: Sequence[str] = (),
) -> list[str]:
    """Take the best ``count`` pairwise-distinct sequences from a ranking.

    The ranking file's own order is the selection; this only refuses to trust a
    ranking that was not produced by the win-probability ordering, and drops
    repeats so that k uids never carry the same molecule.
    """
    if count < 1:
        raise ValueError("a fleet publication needs at least one uid")
    order = str(rankings.get("selection_order") or "")
    if order != required_order:
        raise SelectionOrderError(
            f"ranking selection_order is {order!r}, not {required_order!r}; only "
            "rank one pays and the validator draws once, so candidates must be "
            "ordered by the chance of a winning draw"
        )
    candidates = rankings.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("ranking contains no candidates")
    _assert_non_increasing_win_probability(candidates)
    if not 0.0 <= value_share <= 1.0:
        raise ValueError("value_share is a fraction of the fleet")
    chosen: list[str] = []
    seen: set[str] = set()
    # The value half first, so the fleet always holds the best expected score
    # it can see even when the pool is smaller than the fleet.
    eligible: list[str] = []
    crowd = [str(item).strip().upper() for item in crowd if item]
    value_wanted = int(count * value_share)
    if value_wanted:
        chosen.extend(
            _best_by_expected_score(
                candidates, value_wanted, seen, maximum_identity
            )
        )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("ranking candidates must be mappings")
        sequence = str(candidate.get("sequence") or "").strip().upper()
        if not sequence or sequence in seen:
            continue
        eligible.append(sequence)
        seen.add(sequence)
    # Spread the remaining tickets. A hard similarity floor is the wrong tool
    # here: on the epoch-25095 pool it left ONE assignable candidate out of
    # thirty, because local neighbourhood search produces one molecule thirty
    # times. So take the count we need and make it as separated as this pool
    # allows -- the validator discards a submission too close to another, and
    # over 25093-25095 that cost us four uids of seven every epoch.
    roomy = uncrowded(eligible, crowd)
    if crowd and len(roomy) < len(eligible):
        print(
            f"field crowding: {len(roomy)} of {len(eligible)} candidate(s) are "
            "clear of what the field submitted recently",
            flush=True,
        )
    for pool in (roomy, eligible):
        for sequence in pool:
            if len(chosen) == count:
                break
            if sequence in chosen:
                continue
            if _closest(sequence, chosen) < maximum_identity:
                chosen.append(sequence)
    if len(chosen) < count:
        # Nothing far enough left: fill by farthest-first, best rank breaking
        # ties, so the tickets we do spend are the most separated available.
        remaining = [item for item in eligible if item not in chosen]
        while remaining and len(chosen) < count:
            pick = min(remaining, key=lambda item: _closest(item, chosen))
            chosen.append(pick)
            remaining.remove(pick)
    if not chosen:
        raise ValueError("ranking yielded no usable sequence")
    return chosen


def _assert_non_increasing_win_probability(candidates: Sequence[Any]) -> None:
    """Verify the file really is ordered best-first by win probability."""
    previous: float | None = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get("win_probability")
        if value is None:
            continue
        probability = float(value)
        if not math.isfinite(probability):
            raise SelectionOrderError("candidate win_probability is not finite")
        if previous is not None and probability > previous + 1e-12:
            raise SelectionOrderError(
                "ranking is not ordered best-first by win probability "
                f"({probability} follows {previous})"
            )
        previous = probability


def commitment_blocks(
    *,
    epoch: int,
    epoch_length: int,
    reserve_blocks: int,
    count: int,
    block: int | None = None,
) -> list[int]:
    """One block per commitment, all inside the end-of-epoch reserve.

    ``max_regs_per_block`` is 1, so one extrinsic per block is the pace the chain
    accepts; the reserve is the only part of the epoch where scoring work has
    stopped, so commitments placed there are not overwritten by a later refresh.
    """
    if epoch_length < 1:
        raise ValueError("epoch length must be positive")
    if reserve_blocks < 1:
        raise ValueError("the commitment reserve must be at least one block")
    if count < 1:
        raise ValueError("a fleet publication needs at least one commitment")
    epoch_end = (epoch + 1) * epoch_length
    first = epoch_end - reserve_blocks
    if block is not None and block >= first:
        first = block + 1
    available = epoch_end - first
    if count > available:
        raise ValueError(
            f"{count} commitments do not fit in {available} block(s) left of the "
            f"{reserve_blocks}-block reserve for epoch {epoch}"
        )
    return [first + index for index in range(count)]


def assign_sequences(
    fleet: Sequence[FleetMember],
    sequences: Sequence[str],
    blocks: Sequence[int],
) -> list[Assignment]:
    """Bind one distinct sequence and one block to each uid, lowest uid first."""
    covered = min(len(fleet), len(sequences))
    if len(blocks) < covered:
        raise ValueError("fewer commitment blocks than sequences to commit")
    assignments = [
        Assignment(
            uid=fleet[index].uid,
            hotkey=fleet[index].hotkey,
            sequence=str(sequences[index]).strip().upper(),
            commit_block=int(blocks[index]),
        )
        for index in range(covered)
    ]
    published = [item.sequence for item in assignments]
    if len(set(published)) != len(published):
        raise ValueError(
            "two uids were assigned the same sequence; the validator dedupes by "
            "sequence, so that publication would waste a slot"
        )
    return assignments


def uncovered(
    fleet: Sequence[FleetMember], assignments: Sequence[Assignment]
) -> list[FleetMember]:
    """Registered uids that no distinct qualified sequence was available for."""
    assigned = {item.uid for item in assignments}
    return [member for member in fleet if member.uid not in assigned]


def fleet_fingerprint(fleet: Sequence[FleetMember]) -> str:
    """Identify the held uid set, so a registration change invalidates caches."""
    return json.dumps(
        [[member.uid, member.hotkey] for member in fleet], separators=(",", ":")
    )


def read_ledger(ledger: Path | str) -> list[str]:
    """Read the published-sequence ledger, ignoring comments and blanks."""
    try:
        lines = Path(ledger).read_text().splitlines()
    except (FileNotFoundError, NotADirectoryError, UnicodeDecodeError):
        return []
    return [
        line.strip().upper()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def record_published(ledger: Path | str, sequences: Iterable[str]) -> list[str]:
    """Append every newly published sequence, once, durably.

    All k go in.  The novelty gate consumes this file through
    ``rank_exact --exclude-file``, so a sequence missing from it will be ranked
    and published again later, and the validator's archive drops a resubmission.
    """
    path = Path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set(read_ledger(path))
    appended: list[str] = []
    for sequence in sequences:
        normalized = str(sequence).strip().upper()
        if not normalized or normalized in existing:
            continue
        existing.add(normalized)
        appended.append(normalized)
    if not appended:
        return []
    with path.open("a") as handle:
        for sequence in appended:
            handle.write(sequence + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return appended


def write_exclusion_file(
    destination: Path | str, ledger: Path | str, keep: Iterable[str]
) -> Path:
    """Write the novelty exclusion list, minus the sequences we hold right now.

    The ledger is permanent retirement, and the k sequences of the live epoch are
    in it as soon as they are published.  If those were also excluded from their
    own epoch's re-ranking, a mid-epoch refresh would drop the best k it just
    committed and replace them with the next k -- strictly worse molecules, for
    no reason.  They are excluded from the next epoch, not from this one.
    """
    retained = {str(item).strip().upper() for item in keep}
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [item for item in read_ledger(ledger) if item not in retained]
    path.write_text("".join(f"{item}\n" for item in lines))
    return path


def fleet_artifact(
    *,
    target: str,
    epoch: int,
    stage: str,
    assignments: Sequence[Assignment],
    checkpoints: Mapping[str, Mapping[str, Any]],
    unassigned: Sequence[FleetMember] = (),
) -> dict[str, Any]:
    """Build the single fleet artifact: one file, one entry per uid.

    One file rather than k, because a reader that opens k files can see a torn
    fleet -- some uids from this epoch, some from the last -- with no way to
    detect it.  One atomically replaced file is either wholly old or wholly new.
    """
    entries = []
    for item in assignments:
        checkpoint = checkpoints.get(item.sequence) or {}
        validation = checkpoint.get("validation") or {}
        boltzgen = validation.get("boltzgen") or {}
        selection = checkpoint.get("selection") or {}
        exact = checkpoint.get("exact") or {}
        entries.append({
            "uid": item.uid,
            "hotkey": item.hotkey,
            "sequence": item.sequence,
            "commit_block": item.commit_block,
            "metrics": dict(boltzgen.get("metrics") or {}),
            "selection": {
                "score": selection.get("score"),
                "score_median": selection.get("score_median"),
                "score_p90": selection.get("score_p90"),
                "win_probability": selection.get("win_probability"),
            },
            "observation_count": exact.get("observation_count"),
            "publication_stage": checkpoint.get("publication_stage", stage),
            # The whole per-uid checkpoint travels with the entry so the consumer
            # can run its own fail-closed validation on the exact thing it is
            # about to commit, rather than trusting a summary of it.
            "checkpoint": dict(checkpoint),
        })
    return {
        "schema_version": FLEET_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "nanobody-fleet",
        "status": stage,
        "publication_stage": stage,
        "target": target,
        "epoch": epoch,
        "submission_count": len(entries),
        "submissions": entries,
        "unassigned_uids": [
            {"uid": member.uid, "hotkey": member.hotkey} for member in unassigned
        ],
    }


def artifact_sequences(payload: Any) -> list[str]:
    """Every sequence a fleet artifact published, in uid order.

    Also reads the single-sequence artifact this publisher used to write, so an
    upgrade retires the sequence the previous format published instead of
    silently allowing it to be selected again.
    """
    if not isinstance(payload, Mapping):
        return []
    submissions = payload.get("submissions")
    if isinstance(submissions, list):
        sequences = [
            str(item.get("sequence") or "").strip().upper()
            for item in submissions
            if isinstance(item, Mapping)
        ]
        return [sequence for sequence in sequences if sequence]
    legacy = payload.get("sequences")
    if isinstance(legacy, list):
        return [
            str(item).strip().upper()
            for item in legacy
            if isinstance(item, str) and item.strip()
        ]
    return []


def assignments_from_artifact(payload: Any) -> list[Assignment]:
    """Recover the live uid/sequence/block binding from the fleet artifact."""
    if not isinstance(payload, Mapping):
        return []
    submissions = payload.get("submissions")
    if not isinstance(submissions, list):
        return []
    recovered: list[Assignment] = []
    for entry in submissions:
        if not isinstance(entry, Mapping):
            continue
        sequence = str(entry.get("sequence") or "").strip().upper()
        if not sequence:
            continue
        recovered.append(Assignment(
            uid=int(entry["uid"]),
            hotkey=str(entry["hotkey"]),
            sequence=sequence,
            commit_block=int(entry.get("commit_block") or 0),
        ))
    return recovered


def write_atomic_json(path: Path | str, payload: Any) -> None:
    """Replace a file in one rename, so no reader ever sees half a fleet."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
