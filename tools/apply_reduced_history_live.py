#!/usr/bin/env python3
"""Make live compound pool generation exclude prior reduced-Boltz screens."""
from __future__ import annotations

import argparse
import errno
import os
import shutil
import time
from pathlib import Path


MARKER = "reduced_history_upstream_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def write(path: Path, text: str) -> None:
    if path.read_text() == text:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    shutil.copy2(path, path.with_name(f"{path.name}.pre-reduced-history-{stamp}.bak"))
    temporary = path.with_name(f".{path.name}.reduced-history.tmp")
    temporary.write_text(text)
    os.chmod(temporary, path.stat().st_mode)
    try:
        os.replace(temporary, path)
    except OSError as error:
        if error.errno != errno.ESTALE:
            raise
        with path.open("w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.unlink(missing_ok=True)


def patch_loop(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        # v1.1: the inline pool counter uses os.path for the reduced ledger.
        # A missing import made it fall back to the raw candidate count; pool
        # merging still filtered correctly, but cache rebuild thresholds did not.
        text = text.replace(
            "import json, sqlite3, sys\npath, bank, reduced_path =",
            "import json, os, sqlite3, sys\npath, bank, reduced_path =",
            1,
        )
        write(path, text)
        return
    marker_lines = (
        '# reduced_history_upstream_v1: reduced Boltz observations are real search\n'
        '# history even though they intentionally never enter the full-score bank.\n'
        'REDUCED_ACCUMULATOR="$ART/.reduced-primary-scores-rxn$REACTION.json"\n'
    )
    if 'REACTION=3\nsibling_scan_enabled()' in text:
        text = replace_once(
            text,
            'REACTION=3\nsibling_scan_enabled()',
            'REACTION=3\n' + marker_lines + 'sibling_scan_enabled()',
            "reduced accumulator path",
        )
    else:
        # The shared multi-reaction loop defines REACTION before ART/HIST, so
        # the accumulator can safely be bound beside those canonical paths.
        text = replace_once(
            text,
            'HIST="$ART/public-history.csv"\n',
            'HIST="$ART/public-history.csv"\n' + marker_lines,
            "reduced accumulator path",
        )
    text = replace_once(
        text,
        '''pool_path, part_path = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 0
bank = sys.argv[4] if len(sys.argv) > 4 else ""
lane = sys.argv[5] if len(sys.argv) > 5 else ""
scored = set()
bank_keys = {}
''',
        '''pool_path, part_path = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 0
bank = sys.argv[4] if len(sys.argv) > 4 else ""
lane = sys.argv[5] if len(sys.argv) > 5 else ""
reduced_path = sys.argv[6] if len(sys.argv) > 6 else ""
scored = set()
reduced = set()
bank_keys = {}
''',
        "pool merge arguments",
    )
    text = replace_once(
        text,
        '''    bank_keys = dict(con.execute("SELECT inchikey, name FROM candidates"))
    con.close()
part = json.load(open(part_path))
''',
        '''    bank_keys = dict(con.execute("SELECT inchikey, name FROM candidates"))
    con.close()
if reduced_path and os.path.exists(reduced_path):
    try:
        reduced_payload = json.load(open(reduced_path))
        reduced_rows = reduced_payload if isinstance(reduced_payload, list) else (
            reduced_payload.get("scores") or reduced_payload.get("candidates") or []
        )
        reduced = {str(row["name"]) for row in reduced_rows if row.get("name") is not None}
    except (OSError, ValueError, TypeError):
        # Fail open here only for pool construction; reduced_rank_acquisition.py
        # remains the fail-closed final guard before any reduced NIM call.
        reduced = set()
part = json.load(open(part_path))
''',
        "pool merge reduced history",
    )
    text = replace_once(
        text,
        '''already = 0
collide = 0
for c in part["candidates"]:
''',
        '''already = 0
already_reduced = 0
collide = 0
for c in part["candidates"]:
''',
        "pool merge counters",
    )
    text = replace_once(
        text,
        '''    if c["name"] in scored:
        already += 1
        continue
''',
        '''    if c["name"] in scored:
        already += 1
        continue
    if c["name"] in reduced:
        already_reduced += 1
        continue
''',
        "pool merge reduced exclusion",
    )
    text = replace_once(
        text,
        '''                  "already_scored_dropped": already, "bank_inchikey_dropped": collide,
                  "part": len(part["candidates"]), "pool": len(pool["candidates"])}))
' "$1" "$2" "${3:-0}" "$BANK" "${4:-}" 2>&1 | tail -1
''',
        '''                  "already_scored_dropped": already,
                  "already_reduced_dropped": already_reduced,
                  "bank_inchikey_dropped": collide,
                  "part": len(part["candidates"]), "pool": len(pool["candidates"])}))
' "$1" "$2" "${3:-0}" "$BANK" "${4:-}" "$REDUCED_ACCUMULATOR" 2>&1 | tail -1
''',
        "pool merge invocation",
    )
    text = replace_once(
        text,
        '''path, bank = sys.argv[1], sys.argv[2]
names = [c["name"] for c in (json.load(open(path)).get("candidates") or [])]
con = sqlite3.connect("file:%s?mode=ro" % bank, uri=True, timeout=120)
scored = {row[0] for row in con.execute("SELECT name FROM score_summaries")}
con.close()
print(sum(1 for n in names if n not in scored))
' "$1" "$BANK" 2>/dev/null || acq_candidate_count "$1"
''',
        '''path, bank, reduced_path = sys.argv[1], sys.argv[2], sys.argv[3]
names = [c["name"] for c in (json.load(open(path)).get("candidates") or [])]
con = sqlite3.connect("file:%s?mode=ro" % bank, uri=True, timeout=120)
scored = {row[0] for row in con.execute("SELECT name FROM score_summaries")}
con.close()
reduced = set()
if os.path.exists(reduced_path):
    try:
        payload = json.load(open(reduced_path))
        rows = payload if isinstance(payload, list) else (payload.get("scores") or payload.get("candidates") or [])
        reduced = {str(row["name"]) for row in rows if row.get("name") is not None}
    except (OSError, ValueError, TypeError):
        reduced = set()
print(sum(1 for n in names if n not in scored and n not in reduced))
' "$1" "$BANK" "$REDUCED_ACCUMULATOR" 2>/dev/null || acq_candidate_count "$1"
''',
        "pool count reduced exclusion",
    )
    text = replace_once(
        text,
        "import json, sqlite3, sys\npath, bank, reduced_path =",
        "import json, os, sqlite3, sys\npath, bank, reduced_path =",
        "pool count os import",
    )
    for label, signature in (
        ("frontier", 'frontier_lane_take() {'),
        ("frontier gap", 'frontier_gap_lane_take() {'),
        ("block bandit", 'block_bandit_take() {'),
        ("competitor motif", 'competitor_motif_take() {'),
    ):
        start = text.index(signature)
        end = text.index("\n}", start)
        section = text[start:end]
        section = replace_once(
            section,
            '  local -a exclude=()\n',
            '  local -a exclude=()\n'
            '  [ -f "$REDUCED_ACCUMULATOR" ] && exclude+=(--pending "$REDUCED_ACCUMULATOR")\n',
            f"{label} reduced pending",
        )
        text = text[:start] + section + text[end:]
    if 'explore_lane_take() {' in text:
        explore_start = text.index('explore_lane_take() {')
        explore_end = text.index("\n}", explore_start)
        explore_section = text[explore_start:explore_end]
        if '"${exclude[@]}"' in explore_section:
            explore_section = replace_once(
                explore_section,
                '  local -a exclude=()\n',
                '  local -a exclude=()\n'
                '  [ -f "$REDUCED_ACCUMULATOR" ] && exclude+=(--pending "$REDUCED_ACCUMULATOR")\n',
                "explore reduced pending array",
            )
            text = text[:explore_start] + explore_section + text[explore_end:]
        else:
            text = replace_once(
                text,
            r'''explore_lane_take() {   # $1: limit, $2: part file, $3: pending acquisition (may be empty)
  "${RUN_PROXY[@]}" tools/explore_lane.py --reaction "$REACTION" --bank "$BANK" take \
    --shortlist "$EXPLORE_SHORTLIST" --limit "$1" --output "$2" \
    --max-per-role0 "$EXPLORE_MAX_PER_ROLE0" \
    ${3:+--pending "$3"} 2>&1 | tail -1
}
''',
            r'''explore_lane_take() {   # $1: limit, $2: part file, $3: pending acquisition (may be empty)
  local -a exclude=()
  [ -f "$REDUCED_ACCUMULATOR" ] && exclude+=(--pending "$REDUCED_ACCUMULATOR")
  [ -n "${3:-}" ] && [ -f "$3" ] && exclude+=(--pending "$3")
  "${RUN_PROXY[@]}" tools/explore_lane.py --reaction "$REACTION" --bank "$BANK" take \
    --shortlist "$EXPLORE_SHORTLIST" --limit "$1" --output "$2" \
    --max-per-role0 "$EXPLORE_MAX_PER_ROLE0" "${exclude[@]}" 2>&1 | tail -1
}
''',
                "explore reduced pending",
            )
    write(path, text)


def patch_pending_reader(path: Path, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        return
    old = '''        if path and Path(path).is_file():
            for row in json.loads(Path(path).read_text()).get("candidates") or []:
                pending_names.add(row["name"])
                if row.get("inchikey"):
                    pending_keys.add(row["inchikey"])
'''
    new = f'''        if path and Path(path).is_file():
            payload = json.loads(Path(path).read_text())
            # {marker}: reduced-score accumulators are lists, acquisitions dicts.
            pending = payload if isinstance(payload, list) else payload.get("candidates") or []
            for row in pending:
                pending_names.add(row["name"])
                if row.get("inchikey"):
                    pending_keys.add(row["inchikey"])
'''
    if old in text:
        text = replace_once(text, old, new, f"{path.name} pending reader")
    else:
        frontier_old = '''        if path and Path(path).is_file():
            for row in json.loads(Path(path).read_text()).get("candidates") or []:
                skip_names.add(row["name"])
                if row.get("inchikey"):
                    skip_keys.add(row["inchikey"])
'''
        frontier_new = f'''        if path and Path(path).is_file():
            payload = json.loads(Path(path).read_text())
            # {marker}: reduced-score accumulators are lists, acquisitions dicts.
            pending = payload if isinstance(payload, list) else payload.get("candidates") or []
            for row in pending:
                skip_names.add(row["name"])
                if row.get("inchikey"):
                    skip_keys.add(row["inchikey"])
'''
        text = replace_once(
            text, frontier_old, frontier_new, f"{path.name} pending reader",
        )
    write(path, text)


def patch_explore(path: Path) -> None:
    text = path.read_text()
    if "reduced_history_pending_v1" not in text:
        legacy = '''    pending_names, pending_keys = set(), set()
    if args.pending and Path(args.pending).is_file():
        pending = json.loads(Path(args.pending).read_text()).get("candidates") or []
        pending_names = {c["name"] for c in pending}
        pending_keys = {c.get("inchikey") for c in pending if c.get("inchikey")}
'''
        repeatable = '''    pending_names, pending_keys = set(), set()
    for pending_path in args.pending or []:
        if pending_path and Path(pending_path).is_file():
            pending = json.loads(Path(pending_path).read_text()).get("candidates") or []
            pending_names |= {c["name"] for c in pending}
            pending_keys |= {c.get("inchikey") for c in pending if c.get("inchikey")}
'''
        replacement = '''    pending_names, pending_keys = set(), set()
    for path in args.pending or []:
        if path and Path(path).is_file():
            payload = json.loads(Path(path).read_text())
            # reduced_history_pending_v1: accumulators are lists; acquisitions dicts.
            pending = payload if isinstance(payload, list) else payload.get("candidates") or []
            pending_names.update(c["name"] for c in pending)
            pending_keys.update(c.get("inchikey") for c in pending if c.get("inchikey"))
'''
        if legacy in text:
            text = replace_once(text, legacy, replacement, "explore pending reader")
            text = replace_once(
                text,
                '    take.add_argument("--pending", help="the round acquisition already queued for scoring")\n',
                '    take.add_argument("--pending", action="append", help="acquisition or reduced-score accumulator to exclude; repeatable")\n',
                "explore repeatable pending argument",
            )
        else:
            text = replace_once(text, repeatable, replacement, "explore pending reader")
    write(path, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--loop-script",
        default="tools/run_continuous_rxn3_event_motif_bandit_boxnative.sh",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    patch_loop(root / args.loop_script)
    explore = root / "tools/explore_lane.py"
    if explore.is_file():
        patch_explore(explore)
    patch_pending_reader(
        root / "tools/build_frontier_crossproduct.py", "reduced_history_pending_v1",
    )
    patch_pending_reader(
        root / "tools/build_block_bandit.py", "reduced_history_pending_v1",
    )
    patch_pending_reader(
        root / "tools/build_competitor_motif.py", "reduced_history_pending_v1",
    )
    print({"event": "reduced_history_upstream_installed", "policy": MARKER})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
