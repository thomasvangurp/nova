# Pure allocation/planning functions from installed Pi consumer ea9677c4.
# No authentication, chain or submission implementation is included.
import json
import time


NULL = "~"

STALE_AGE_LIMIT = 3

MAX_ARTIFACT_AGE_SECONDS = 3600

def sequences_by_uid(artifact: dict, epoch: int,
                     allow_stale: bool = False,
                     crew=None) -> dict[int, str]:
    """The nanobody half. A stale artifact is refused, not silently reused.

    Its epoch is checked because the producer rewrites it every epoch and a
    commit carrying last epoch's sequence is a wasted slot, not a near miss.

    But refusing used to abort the WHOLE commit, molecules included, and once
    the Pi became the sole writer that turned a late producer into an empty
    epoch -- five of them between 25243 and 25247. So the caller may allow a
    stale artifact once it has waited as long as the window permits: last
    epoch's sequence is a worse submission than this epoch's, and a far better
    one than nothing.
    """
    stale = int(artifact.get("epoch", -1)) != epoch
    artifact_epoch = int(artifact.get("epoch", -1))
    # Freshness (bug 4, format-robust): prefer the writer's updated_at_unix
    # stamp, but a producer that does not stamp it must NOT hard-fail the
    # commit. A missing/non-finite stamp falls back to the always-present
    # epoch field, and only a genuinely stale epoch (age > STALE_AGE_LIMIT)
    # is rejected. A VALID-but-out-of-window stamp is still rejected.
    updated = artifact.get("updated_at_unix")
    stamp_is_valid = (
        isinstance(updated, (int, float))
        and not isinstance(updated, bool)
        and updated == updated
    )
    if stamp_is_valid and abs(time.time() - updated) > MAX_ARTIFACT_AGE_SECONDS:
        raise ValueError("nanobody artifact updated_at_unix is stale")
    if stale and not allow_stale:
        raise ValueError(
            f"nanobody artifact is for epoch {artifact.get('epoch')}, not {epoch}")
    if stale and artifact_epoch >= 0 and epoch - artifact_epoch > STALE_AGE_LIMIT:
        # A producer outage must never silently recommit the same pool forever:
        # an artifact this old is guaranteed to already be in the HF archive,
        # so recommitting it just re-earns the "not unique" rejection.
        raise ValueError(
            f"nanobody artifact is {epoch - artifact_epoch} epochs stale "
            f"(epoch {artifact_epoch}, limit {STALE_AGE_LIMIT}); refusing to "
            "recommit a duplicate")
    if stale:
        print(json.dumps(dict(event="stale_nanobody", have=artifact.get("epoch"),
                              want=epoch)), flush=True)
    out = {}
    pool = []

    # New artifacts carry an explicitly UID-independent pool.  The GPU host
    # does not own registrations and therefore cannot authoritatively bind a
    # sequence to a uid: the Pi may have bought a replacement since the
    # producer's last chain snapshot.  Keep accepting the old per-uid shape so
    # a rolling deploy remains safe, but treat every distinct sequence in it as
    # a pool entry too.
    sequence_pool = artifact.get("sequence_pool")
    for item in sequence_pool or []:
        raw = item.get("sequence") if isinstance(item, dict) else item
        if raw is None:
            continue
        sequence = str(raw).strip().upper()
        if sequence and sequence not in pool:
            pool.append(sequence)
    # Bug 10: an explicit-but-empty pool is no evidence the artifact can fill
    # a freshly acquired uid. Legacy artifacts (no sequence_pool key) still
    # fall back to their per-uid submissions below.
    if sequence_pool is not None and not pool:
        raise ValueError("nanobody artifact has no usable sequence pool")
    for entry in artifact.get("submissions") or []:
        sequence = str(entry.get("sequence") or "").strip().upper()
        if sequence:
            out[int(entry["uid"])] = sequence
            if sequence not in pool:
                pool.append(sequence)

    if crew is None:
        return out

    # Preserve an assignment when that uid is still ours, then deterministically
    # fill every newly acquired/recycled uid from the unused ranked pool.  This
    # is the ownership boundary: only the submit host's fresh crew snapshot is
    # allowed to decide which uid carries which sequence.
    allocated = {}
    used = set()
    for uid, _hotkey in crew:
        sequence = out.get(int(uid))
        if sequence and sequence not in used:
            allocated[int(uid)] = sequence
            used.add(sequence)
    # Fill with the best pool entry that is NOT a validator duplicate of a
    # sequence already allocated: the validator keeps one of any pair at or
    # above 0.95 identity, so a near-copy scores nothing. Epoch 25400 filled
    # five uids in pool order with 0.976-0.984 copies. Only when no distinct
    # entry is left does a uid take a near-copy, so every uid still commits.
    for uid, _hotkey in crew:
        uid = int(uid)
        if uid in allocated:
            continue
        remaining = [sequence for sequence in pool if sequence not in used]
        if not remaining:
            break
        taken = list(allocated.values())
        sequence = next(
            (candidate for candidate in remaining
             if all(_identity(candidate, other) < DUPLICATE_IDENTITY
                    for other in taken)),
            remaining[0])
        allocated[uid] = sequence
        used.add(sequence)
    return allocated

def plan(crew, molecules, sequences, value=None):
    """One row per uid, with whichever halves are actually available.

    A uid with neither half is dropped rather than committed as "~|~", which
    scores nothing and still spends the commitment slot.
    """
    rows = []
    for uid, hotkey in crew:
        mols = molecules.get(uid, "")
        seq = sequences.get(uid, "")
        if not mols and not seq:
            continue
        if not seq:
            # The VALIDATOR accepts "molecules|~" -- its parser returns 20
            # molecules and 0 sequences -- but commit_submission's own
            # validate_plaintext demands a real sequence and rejects it. Until
            # that over-strictness is relaxed on the signing path (deliberately,
            # not under time pressure), such a uid is skipped and said so,
            # rather than sent to a driver that will refuse it.
            print(json.dumps(dict(event="skipped_no_sequence", uid=uid)), flush=True)
            continue
        rows.append(dict(uid=uid, hotkey=hotkey,
                         plaintext=f"{mols or NULL}|{seq}",
                         has_molecules=bool(mols), has_sequence=True))
    # Best first. Commits are serial and the epoch can run out before the list
    # does, so crew order decides which uids get dropped -- and crew order is
    # arbitrary. If anything is going to be lost to the blackout, it must be
    # the least valuable portfolio, not whichever uid happens to sort last.
    value = value or {}
    rows.sort(key=lambda r: (-float(value.get(r["uid"], 0.0)),
                             -r["has_molecules"], r["uid"]))
    return rows


# Exact installed main I/O statements, isolated from signing and CLI setup.
from pathlib import Path

def main(arguments):
    if arguments.pull or arguments.pull_only:
        pull_producer(arguments.vastai, arguments.producer_label,
                      arguments.producer_path, Path(arguments.nanobody))
        if arguments.pull_only:
            return 0
    crew = [(int(u), str(h)) for u, h in _load(Path(arguments.crew))]
    sequences = sequences_by_uid(_load(Path(arguments.nanobody)), arguments.epoch,
                                 allow_stale=arguments.allow_stale_nanobody,
                                 crew=crew)
    # Empty stand-ins for the independent molecule I/O omitted from this fixture.
    molecules, value = {}, {}
    rows = plan(crew, molecules, sequences, value)
    return sequences
