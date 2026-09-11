"""Keep proposals inside the neighbourhood where winners are actually found.

Every one of the six winners in the leakage-free replay was Hamming distance one
from a sequence a different hotkey had already published.  Measured against the
live proposal bank, about forty per cent of each generation sits at distance two
or more from anything in the public archive: whole lanes -- ``local_substitution``
at 0% within distance one, ``whole_cdr_motif`` at 3%, the terminal and framework
haplotype lanes around 30% -- explore space where no winner has ever been
observed.

At a 320-candidate budget that gamble was unavoidable, because a 127-residue
sequence has 2,413 single mutants and only a sample of them fits.  At several
thousand observations per epoch it is no longer necessary: the budget can cover
the distance-one shell instead of sampling beyond it.

This filter is deliberately a post-generation step.  It does not retune the
generator, so every lane keeps producing what it produces and the change is one
comparison against the archive, reversible by dropping a flag.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SEQUENCE_LENGTH = 127


def _encode(sequences: list[str], length: int) -> np.ndarray:
    return np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(
        len(sequences), length
    )


def archive_sequences(path: Path, length: int) -> list[str]:
    """Every public sequence of the expected length, deduplicated."""
    seen: dict[str, None] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            sequence = (row.get("sequence") or "").strip().upper()
            if len(sequence) == length:
                seen.setdefault(sequence, None)
    return list(seen)


def minimum_distances(
    queries: list[str], archive: list[str], *, block: int = 64
) -> np.ndarray:
    """Hamming distance from each query to the nearest archive sequence."""
    if not queries:
        return np.zeros(0, dtype=int)
    if not archive:
        # Without an archive every proposal is equally unsupported; returning
        # zero keeps the filter inert rather than discarding the whole bank.
        return np.zeros(len(queries), dtype=int)
    length = len(queries[0])
    reference = _encode(archive, length)
    encoded = _encode(queries, length)
    closest = np.empty(len(queries), dtype=int)
    for start in range(0, len(queries), block):
        chunk = encoded[start : start + block]
        distance = (chunk[:, None, :] != reference[None, :, :]).sum(2)
        closest[start : start + block] = distance.min(1)
    return closest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--maximum-distance",
        type=int,
        default=1,
        help="keep proposals within this Hamming distance of a public sequence",
    )
    parser.add_argument(
        "--minimum-keep",
        type=int,
        default=400,
        help=(
            "never hand the screen fewer proposals than this; if the filter is "
            "that aggressive the nearest proposals are kept instead, because "
            "screening something beats screening nothing"
        ),
    )
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()

    with args.input.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader]
    usable = [
        row for row in rows
        if len(str(row.get("sequence", "")).strip()) == args.sequence_length
    ]
    if not usable:
        raise SystemExit(f"no usable proposals in {args.input}")

    archive = archive_sequences(args.archive, args.sequence_length)
    distances = minimum_distances(
        [str(row["sequence"]).strip().upper() for row in usable], archive
    )
    order = np.argsort(distances, kind="stable")
    kept_indices = [
        index for index in order if distances[index] <= args.maximum_distance
    ]
    if len(kept_indices) < args.minimum_keep:
        kept_indices = list(order[: args.minimum_keep])
    kept = [usable[index] for index in kept_indices]

    if "public_distance" not in fieldnames:
        fieldnames.append("public_distance")
    for row, index in zip(kept, kept_indices, strict=True):
        row["public_distance"] = int(distances[index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    within = int((distances <= args.maximum_distance).sum())
    print(
        f"PUBLIC_NEIGHBOURHOOD_FILTER input={len(usable)} "
        f"within_distance_{args.maximum_distance}={within} kept={len(kept)}",
        flush=True,
    )
    if within < len(usable):
        print(
            f"  discarded {len(usable) - within} proposals at distance "
            f"> {args.maximum_distance} from every public sequence",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
