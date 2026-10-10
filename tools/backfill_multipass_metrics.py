#!/usr/bin/env python3
"""Back-fill raw per-pass metrics for multipass passes still on disk (no GPU).

ARCHITECTURE_REDESIGN P2: until 2026-10-09 ``multipass_all_rivals.py`` stored
only a field-relative rank-sum per pass (``metrics_json='{}'``), so any formula
or field change forced a full re-fold. The raw BoltzGen metrics of every pass
it ran are still on the dashboard box in
``/root/multipass_all_rivals/<epoch>/score<run>/exact_all.json``. This reads
them and upserts the same observation ids the scorer wrote
(``public_multipass:epoch<E>:<seq>:pass<run>``) under the explicit
``multipass_sum10_pass`` label, with:

* ``metrics_json``  -- the raw per-pass metrics (formula-independent);
* ``score``         -- sum-of-10 re-derived against the field given by
                       ``--archive`` (recorded as field_rows + field_sha256);
* ``provenance``    -- writer, formula, pass index/count, cohort, the source
                       file, and the row's ORIGINAL score/score_kind, so the
                       relabel destroys nothing (R3 "repair without deleting").

Rows that already carry metrics are left alone (idempotent). Epoch dirs
written to in the last ``--settle-minutes`` are skipped (a run in progress).

Usage (dashboard box, bgen env)::

    /root/bgen_python.sh /root/backfill_multipass_metrics.py --dry-run
    /root/bgen_python.sh /root/backfill_multipass_metrics.py --epochs 25613-25618
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, Path("/root")):
    if (candidate / "nanobody_opt").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
# The scorer module that sits next to this file wins (never a stale /root copy).
_MP_PATH = next(c / "multipass_all_rivals.py"
                for c in (HERE, HERE.parent / "tools", Path("/root"))
                if (c / "multipass_all_rivals.py").is_file())
_spec = importlib.util.spec_from_file_location("multipass_all_rivals", _MP_PATH)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

BACKFILL_WRITER = "backfill_multipass_metrics"


def parse_epochs(spec: str | None, root: Path) -> list[int]:
    available = sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())
    if not spec:
        return available
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        elif part:
            wanted.add(int(part))
    return [e for e in available if e in wanted]


def cohort_map(epoch: int, epoch_dir: Path) -> dict[str, str]:
    """Our sequences for ``epoch`` (everything else is a rival): the Pi's
    intended drop plus our coldkey's leaderboard rows, as the scorer does."""
    ours = {seq for _, seq in mp.fetch_our_intended(epoch)}
    board = mp.fetch_leaderboard(epoch) or []
    ours |= {seq for ck, seq in board if ck == mp.OUR_COLDKEY}
    return {seq: "ours" for seq in ours}


def existing_rows(connection, ids: list[str]) -> dict[str, tuple]:
    found: dict[str, tuple] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join(["%s"] * len(chunk))
        for row in connection.execute(
            "SELECT observation_id, score, score_kind, metrics_json "
            f"FROM observations WHERE observation_id IN ({marks})",
            tuple(chunk),
        ).fetchall():
            found[str(row[0])] = (row[1], row[2], row[3])
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("/root/multipass_all_rivals"))
    parser.add_argument("--archive", type=Path, default=Path("/root/hf_archive.csv"))
    parser.add_argument("--epochs", default=None, help="e.g. 25594-25618,25620")
    parser.add_argument("--settle-minutes", type=float, default=15.0)
    parser.add_argument("--force", action="store_true",
                        help="rewrite rows that already carry metrics")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from nanobody_opt import db
    from nanobody_opt.mutation_network import update_observation_store

    archive = pd.read_csv(args.archive).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive["sequence"].astype(str).str.upper()
    field = mp.field_fingerprint(args.archive, archive)
    print(f"field rows={field['field_rows']} sha256={field['field_sha256'][:12]}", flush=True)

    totals = {"epochs": 0, "passes": 0, "rows": 0, "written": 0, "had_metrics": 0,
              "new_rows": 0, "skipped_busy": 0}
    now = time.time()
    for epoch in parse_epochs(args.epochs, args.root):
        epoch_dir = args.root / str(epoch)
        pass_files = sorted(
            (int(p.name[len("score"):]), p / "exact_all.json")
            for p in epoch_dir.glob("score*")
            if p.name[len("score"):].isdigit() and (p / "exact_all.json").is_file()
        )
        if not pass_files:
            continue
        newest = max(p.stat().st_mtime for p in epoch_dir.rglob("*") if p.is_file())
        if now - newest < args.settle_minutes * 60:
            print(f"epoch {epoch}: SKIP, written {int((now - newest) / 60)} min ago "
                  "(run in progress?)", flush=True)
            totals["skipped_busy"] += 1
            continue
        cohorts = cohort_map(epoch, epoch_dir)
        observations: list[dict] = []
        for run, path in pass_files:
            payload = json.loads(path.read_text())
            ranks, structs, raw = mp.rank_pass_payload(payload, archive)
            observations.extend(mp.pass_observations(
                epoch=epoch, run=run, passes=len(pass_files), ranks=ranks, raw=raw,
                structs=structs, cohort_of=cohorts, field=field,
                extra_provenance={"writer": BACKFILL_WRITER,
                                  "backfilled_from": str(path),
                                  "backfilled_from_mtime": int(path.stat().st_mtime),
                                  "backfilled_at": int(now)}))
        with db.connect() as connection:
            before = existing_rows(connection, [o["observation_id"] for o in observations])
        keep = []
        for obs in observations:
            old = before.get(obs["observation_id"])
            if old is None:
                totals["new_rows"] += 1
                obs["original_score"] = None
                obs["original_score_kind"] = None
            else:
                metrics_json = old[2]
                if metrics_json and metrics_json not in ("{}", "null") and not args.force:
                    totals["had_metrics"] += 1
                    continue
                obs["original_score"] = None if old[0] is None else float(old[0])
                obs["original_score_kind"] = old[1]
            keep.append(obs)
        with_metrics = sum(1 for o in keep if o["metrics"])
        print(f"epoch {epoch}: passes={len(pass_files)} rows={len(observations)} "
              f"to_write={len(keep)} with_metrics={with_metrics} "
              f"ours={sum(1 for o in keep if o['cohort'] == 'ours')}", flush=True)
        totals["epochs"] += 1
        totals["passes"] += len(pass_files)
        totals["rows"] += len(observations)
        if keep and not args.dry_run:
            # Small batches with retries: the central DB is shared with the
            # live producer, and one 500-row transaction deadlocked (1213).
            for start in range(0, len(keep), 100):
                batch = keep[start:start + 100]
                for attempt in range(1, 6):
                    try:
                        totals["written"] += update_observation_store(
                            Path("/root/artifacts/nanobody_mutations.sqlite"), batch)
                        break
                    except Exception as error:  # noqa: BLE001
                        print(f"  epoch {epoch} batch {start}: attempt {attempt} "
                              f"{type(error).__name__}: {error}", flush=True)
                        if attempt == 5:
                            raise
                        time.sleep(5 * attempt)
    print("BACKFILL_DONE " + json.dumps({**totals, "dry_run": args.dry_run}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
