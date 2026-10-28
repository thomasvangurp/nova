#!/usr/bin/env python3
"""Generate the exhaustive single-edit sweep children for the current target.

Derives the target from ``rules.target_for_epoch(epoch)`` and the sweep length
from the modal length of that target's measured parents (no P20809/127
hardcode).  The swept positions default to the framework (non-CDR) positions of
a modal-length parent, so a Q9NZQ7 sweep does not inherit P20809's position
set.  Run against the production mutation DB (``/root/artifacts/
nanobody_mutations.sqlite``); emits ``exhaustive_children.json``.
"""
import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt.rules import target_for_epoch

DB = "/root/artifacts/nanobody_mutations.sqlite"
AA = "ACDEFGHIKLMNPQRSTVWY"

# 0-indexed P20809 framework positions from the original sweep, kept only as a
# legacy default when re-deriving that historical target.  Never applied to
# Q9NZQ7 (its positions are parsed from the parents' framework regions).
LEGACY_P20809_POSITIONS = [0, 2, 113, 121, 120, 123, 115, 122, 125, 110, 72, 114, 126]


def chain_epoch() -> int | None:
    try:
        return int(json.load(open("/root/artifacts/chain_state.json"))["epoch"])
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def modal_length(sequences) -> int:
    return Counter(map(len, sequences)).most_common(1)[0][0]


def framework_positions(sequences) -> list[int]:
    """0-indexed non-CDR positions, parsed from the first VHH-shaped parent."""
    from nanobody_opt.regions import find_regions
    for sequence in sequences:
        try:
            regions = find_regions(sequence)
        except ValueError:
            continue
        cdr = set()
        for start, end in (regions.cdr1, regions.cdr2, regions.cdr3):
            cdr.update(range(start, end))
        return [i for i in range(len(sequence)) if i not in cdr]
    raise SystemExit("no VHH-shaped parent to derive framework positions from; pass --positions")


def main(parents_n: int, epoch: int, positions: list[int] | None) -> None:
    target = target_for_epoch(epoch)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    acc: dict[str, list[float]] = {}
    for seq, sc in conn.execute(
        "SELECT n.sequence, o.score FROM nodes n JOIN observations o "
        "ON o.node_id=n.id AND o.score_kind='local_validator_replay_rank_sum' "
        "WHERE n.target=?", (target,)):
        acc.setdefault(seq, []).append(sc)
    if not acc:
        conn.close()
        raise SystemExit(f"no measured parents for target {target} (epoch {epoch})")

    length = modal_length(list(acc))
    modal = [(s, v) for s, v in acc.items() if len(s) == length]
    parents = [s for s, _ in sorted(modal, key=lambda x: (statistics.mean(x[1]), x[0]))[:parents_n]]
    if not parents:
        conn.close()
        raise SystemExit(f"no parents at modal length {length} for target {target}")

    if positions is None:
        positions = LEGACY_P20809_POSITIONS if target == "P20809" else framework_positions(parents)
    positions = [p for p in positions if 0 <= p < length]

    existing = set(r[0] for r in conn.execute("SELECT sequence FROM nodes WHERE target=?", (target,)))
    conn.close()

    children: dict[str, dict] = {}
    for p in parents:
        for pos in positions:
            for to in AA:
                if pos >= len(p) or p[pos] == to:
                    continue
                child = p[:pos] + to + p[pos + 1:]
                if child in existing:
                    continue
                children.setdefault(child, {"sequence": child, "parent": p,
                                            "position": pos, "to": to})

    work = [{"sequence": c["sequence"], "epoch": epoch, "parent": c["parent"],
             "position": c["position"], "to": c["to"]} for c in children.values()]
    spec = {"target": target, "epoch": epoch, "length": length,
            "parents": parents, "positions": positions, "children": work}
    json.dump(spec, open("exhaustive_children.json", "w"))

    per = Counter(w["position"] for w in work)
    print(f"target={target} epoch={epoch} length={length} "
          f"parents={len(parents)} children={len(work)}")
    print("per-position:", dict(sorted(per.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=None,
                        help="target epoch (default: current chain epoch)")
    parser.add_argument("--positions", type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                        default=None, help="0-indexed positions to sweep (default: framework positions)")
    args = parser.parse_args()
    epoch = args.epoch if args.epoch is not None else chain_epoch()
    if epoch is None:
        parser.error("--epoch is required when /root/artifacts/chain_state.json is unavailable")
    main(args.parents, epoch, args.positions)
