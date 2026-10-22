#!/usr/bin/env python3
"""Sequences the slave still has to replay for one closed epoch.

Every P20809 submission on that epoch's public leaderboard (all miners, us
included) plus our own qualified shortlist, minus any sequence that already
has >= PASSES stored replay passes in master's local SQLite. Coverage is per
sequence, not per epoch: a fold is a property of (sequence, target).

Exit codes: 0 = list written (possibly empty), 3 = leaderboard not published
yet (caller retries later).
"""
from __future__ import annotations

import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies"
DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"
SUPPORTED_LENGTH = 127
REPLAY_LABELS = (("public_multipass_rivals", "multipass_sum10_pass"),)


def leaderboard_sequences(epoch: int) -> list[str] | None:
    req = urllib.request.Request(API.format(epoch=epoch), headers={"User-Agent": "nova-slave-multipass/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        payload = {}
    rows = payload.get("data") or []
    if not rows:
        # No published board (since 25693 the validator's upload is rejected): use the rebuilt-from-log one.
        try:
            from tools.leaderboard_fallback import load_rows
        except ImportError:
            from leaderboard_fallback import load_rows
        rows = load_rows(epoch) or []
    seqs = []
    for submission in rows:
        for nb in submission.get("nanobodies") or []:
            if nb.get("protein_name") == TARGET and nb.get("sequence"):
                seqs.append(str(nb["sequence"]).strip().upper())
    return seqs or None


def qualified_sequences(epoch: int) -> list[str]:
    qdir = Path(f"/root/live/epoch{epoch}/checkpoints/qualified")
    seqs = []
    for f in sorted(qdir.glob("uid*.json")) if qdir.is_dir() else []:
        seqs += [str(s).strip().upper() for s in json.loads(f.read_text()).get("sequences") or []]
    return seqs


def pass_counts(seqs: list[str]) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    counts: dict[str, int] = {}
    label_sql = " OR ".join("(o.source=? AND o.score_kind=?)" for _ in REPLAY_LABELS)
    label_args = [x for pair in REPLAY_LABELS for x in pair]
    for i in range(0, len(seqs), 500):
        chunk = seqs[i:i + 500]
        rows = conn.execute(
            f"SELECT n.sequence, COUNT(*) FROM observations o JOIN nodes n ON n.id=o.node_id "
            f"WHERE n.target=? AND n.sequence IN ({','.join('?' * len(chunk))}) AND ({label_sql}) "
            f"AND o.metrics_json NOT IN ('', '{{}}') GROUP BY n.sequence",
            (TARGET, *chunk, *label_args),
        ).fetchall()
        counts.update({s: c for s, c in rows})
    conn.close()
    return counts


def main() -> int:
    epoch, out_path, passes = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    board = leaderboard_sequences(epoch)
    if board is None:
        print(f"  leaderboard for epoch {epoch} not published yet")
        return 3
    wanted = sorted(set(board) | set(qualified_sequences(epoch)))
    # The fast scorer only supports the modal length (127); a wave with any
    # other length is refused, and a lone odd-length wave hangs. These are
    # ~0.1% of submissions (1 x 128, 2 x 129 since 25600) and cannot be replayed.
    odd = [s for s in wanted if len(s) != SUPPORTED_LENGTH]
    if odd:
        print(f"  skipping {len(odd)} sequence(s) of unsupported length "
              f"{sorted({len(s) for s in odd})}")
    wanted = [s for s in wanted if len(s) == SUPPORTED_LENGTH]
    counts = pass_counts(wanted)
    todo = [s for s in wanted if counts.get(s, 0) < passes]
    out_path.write_text(json.dumps(todo))
    print(f"  epoch {epoch}: {len(wanted)} submitted/qualified, {len(wanted) - len(todo)} already "
          f"have >= {passes} passes, {len(todo)} to score -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
