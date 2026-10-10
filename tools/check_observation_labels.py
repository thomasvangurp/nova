#!/usr/bin/env python3
"""Read-only DB trip-wire for the observation label registry (P4 / R3).

For each ``(source, score_kind)`` it checks, per epoch:

* the label is registered in ``nanobody_opt/observation_labels.py``;
* the source is registered for that label (or documented as known off-scale);
* the epoch mean lies inside the label's ``typical_range``;
* the epoch mean does not jump more than ``--max-jump`` x against the previous
  epoch with data (the 37k -> 53k -> 120k pattern of 2026-10-09).

Exit 0 when clean, 1 when anything is flagged. Default scope is the
``public_multipass_rivals`` source, which uses the (source, score_kind) index
and is cheap. ``--all-pairs`` also lists every (source, score_kind) pair in
the table, which scans that index.

Usage (a host with the central-DB creds)::

    /root/bgen_python.sh /root/check_observation_labels.py
    /root/bgen_python.sh /root/check_observation_labels.py --all-pairs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _root in (HERE, HERE.parent):
    if (_root / "nanobody_opt").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from nanobody_opt.observation_labels import LABELS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default="public_multipass_rivals")
    parser.add_argument("--max-jump", type=float, default=1.5)
    parser.add_argument("--all-pairs", action="store_true")
    args = parser.parse_args()

    from nanobody_opt import db

    findings: list[str] = []
    with db.connect() as connection:
        if args.all_pairs:
            for source, kind, n in connection.execute(
                    "SELECT source, score_kind, COUNT(*) FROM observations "
                    "GROUP BY source, score_kind").fetchall():
                label = LABELS.get(kind)
                state = ("UNREGISTERED" if label is None else
                         "ok" if source in label.sources else
                         "KNOWN-OFF-SCALE" if source in label.known_foreign else
                         "SOURCE-NOT-REGISTERED")
                print(f"pair {source} / {kind}: {n} rows [{state}]")
                if state in ("UNREGISTERED", "SOURCE-NOT-REGISTERED"):
                    findings.append(f"{source}/{kind}: {state} ({n} rows)")
        for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
            rows = connection.execute(
                "SELECT score_kind, epoch, COUNT(*), AVG(score) FROM observations "
                "WHERE source=%s GROUP BY score_kind, epoch ORDER BY score_kind, epoch",
                (source,)).fetchall()
            previous: dict[str, tuple[int, float]] = {}
            for kind, epoch, n, mean in rows:
                mean = float(mean)
                label = LABELS.get(kind)
                where = f"{source}/{kind} epoch={epoch} n={n} mean={mean:.0f}"
                if label is None:
                    findings.append(f"{where}: unregistered score_kind")
                    continue
                if source in label.known_foreign:
                    print(f"known  {where}: documented off-scale rows "
                          f"({label.known_foreign[source][:80]})")
                    continue
                if source not in label.sources:
                    findings.append(f"{where}: source not registered for label")
                if label.typical_range and not (
                        label.typical_range[0] <= mean <= label.typical_range[1]):
                    findings.append(f"{where}: outside registered range "
                                    f"{label.typical_range}")
                if epoch is not None and kind in previous:
                    p_epoch, p_mean = previous[kind]
                    ratio = max(mean, p_mean) / max(min(mean, p_mean), 1e-9)
                    if ratio > args.max_jump:
                        findings.append(f"{where}: mean jumped {ratio:.2f}x vs "
                                        f"epoch {p_epoch} ({p_mean:.0f})")
                if epoch is not None:
                    previous[kind] = (int(epoch), mean)
            kinds = sorted({r[0] for r in rows})
            print(f"source {source}: {len(rows)} (score_kind, epoch) groups, "
                  f"kinds={kinds}")
    for finding in findings:
        print("FLAG " + finding)
    print(f"LABEL_CHECK {'CLEAN' if not findings else 'FLAGGED'} findings={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
