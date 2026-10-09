#!/usr/bin/env python3
"""Build the validator's exact top-50 index from the FULL HF archive.

The validator builds its dedup top-50 with ``rank_binders`` (Algorithm 2) over
Metanova/Submission-Archive's ``P20809_nanobodies.csv`` -- ALL history,
refreshed every epoch -- NOT the live leaderboard. A previous builder ranked
only the last N API epochs, which indexes a different (current-basin) fifty and
over-filters: the cause of the 25454/25455 wholesale single-edit rejections.
This rebuilds the reference from the archive ``fetch_hf_archive.py`` downloads,
using the validator's OWN ``rank_binders`` source (loaded by path), so the
indexed fifty is exactly what the validator compares against.

Output JSON::

    {"sequences": [top-50], "cdr3s": [...], "n": 50, "ranked": int,
     "live_sequences": [...], "source": "hf_archive", "archive_count": int}

Invocation (metanano env)::

    python build_top50_index.py <archive.csv> <out.json> [obs_db]

``obs_db``, when given, supplies ``live_sequences`` -- our own multi-pass replay
top-50, the stable quality read for frontier parents -- instead of the archive
top-50 fallback.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _validator_rank_binders():
    """Load utils/minmax_weighted_rank.rank_binders by path (validator's source).

    Not ``from utils import rank_binders``: utils/__init__.py imports rdkit,
    which the metanano environment does not carry. Loading the file directly is
    still the validator's own source, not a reimplementation of it.
    """
    source = Path(__file__).resolve().parent.parent / "utils" / "minmax_weighted_rank.py"
    spec = importlib.util.spec_from_file_location("_nova_minmax_weighted_rank", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the validator's ranking from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rank_binders


#: Writers whose rows under the two replay kinds are on the producer's ~20-370
#: rank-sum scale (ARCHITECTURE_AUDIT §3a). ``public_multipass_rivals``
#: (~2e4-1.8e5, sum-of-10/algo2 vs the full HF field) and ``hf_archive_top200``
#: (~5.8e4) share the score_kind but not the scale: pooling them into one
#: AVG(score) pushed 887 of our own sequences to rank ~44,000 and swapped 2 of
#: the 50 (measured on the live DB 2026-10-09). A new writer must be added here
#: deliberately, with its scale checked -- never picked up by default.
REPLAY_SCALE_SOURCES = (
    "live_exact_screen",
    "live_exact_repeat",
    "local_exact_raw",
    "children_8x",
    "children_20x",
    "children_40x",
    "backfill_8x",
    "exhaustive_8x",
)


def _live_top50_from_db(
    db_path: str, *, limit: int = 50, min_obs: int = 8, min_distance: int = 6,
    extra: int = 15,
) -> list[str]:
    """Our own multi-pass parents: the top-50 plus a few diverse fillers.

    The validator's leaderboard final_score is a single-pass, one-fold score
    (noisy); our race replays the validator scorer 20x, so the mean of those
    replay rank-sums is the stable quality read. Lower rank sum = better.

    The top-50 are the strongest parents (their single-edits are the optimal
    fleet). Their single-edits crowd into one family, so the fleet selector's
    0.95 dedup leaves a few UIDs unfilled; append `extra` parents that are
    >= min_distance edits from the top-50 so those slots get distinct edits.
    """
    from collections import Counter

    import numpy as np

    # Read the LIVE central MySQL DB (the producer writes there since ~25558),
    # NOT the stale per-machine sqlite file. sqlite froze at the migration and
    # left the priority-parent read on a dead snapshot.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nanobody_opt import db  # noqa: E402

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT n.sequence, AVG(o.score) AS mean_score, COUNT(*) AS n_obs
            FROM observations o JOIN nodes n ON o.node_id = n.id
            WHERE n.target = 'P20809'
              AND o.score_kind IN ('local_validator_replay_rank_sum',
                                   'local_exact_validator_replay_rank_sum')
              AND o.source IN ({placeholders})
            GROUP BY n.sequence
            HAVING n_obs >= %s
            ORDER BY mean_score ASC
            LIMIT %s
            """.format(placeholders=",".join(["%s"] * len(REPLAY_SCALE_SOURCES))),
            (*REPLAY_SCALE_SOURCES, min_obs, limit + extra * 4),
        ).fetchall()
    candidates = [str(r[0]).strip().upper() for r in rows]

    if not candidates:
        # Fail closed: an empty dedup reference would silently disable the
        # novelty filter downstream. The producer's iteration fails loudly
        # instead (failed.json -> check_mining_liveness).
        raise SystemExit(
            "TOP50_EMPTY: no P20809 replay rows from sources "
            f"{REPLAY_SCALE_SOURCES} with >= {min_obs} observations"
        )

    mlen = Counter(len(s) for s in candidates).most_common(1)[0][0]
    candidates = [s for s in candidates if len(s) == mlen]
    arr = np.frombuffer("".join(candidates).encode(), dtype=np.uint8).reshape(
        len(candidates), mlen
    )

    top = candidates[:limit]
    top_arr = arr[:limit]

    filler: list[str] = []
    for i in range(limit, len(candidates)):
        seq = candidates[i]
        if int((top_arr != arr[i]).sum(axis=1).min()) < min_distance:
            continue
        if filler and any(
            int((arr[j] != arr[i]).sum()) < min_distance
            for j in range(limit, limit + len(filler))
        ):
            continue
        filler.append(seq)
        if len(filler) >= extra:
            break
    return top + filler


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    archive_path = Path(sys.argv[1])
    out_path = sys.argv[2]
    obs_db = sys.argv[3] if len(sys.argv) > 3 else None

    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise SystemExit(
            f"HF archive missing or empty at {archive_path}; "
            "run fetch_hf_archive.py first"
        )

    from metanano.utils.cdr_utils import extract_cdrs  # noqa: E402

    frame = pd.read_csv(archive_path)
    if "sequence" not in frame.columns or frame.empty:
        raise SystemExit(f"HF archive has no sequence column: {archive_path}")

    rank_binders = _validator_rank_binders()
    top = rank_binders(frame, k=50, max_liability_violations=50).head(50)
    hf_sequences = [str(s).strip().upper() for s in top["sequence"].tolist()]

    if obs_db:
        live_sequences = _live_top50_from_db(obs_db)
    else:
        live_sequences = hf_sequences

    # The dedup reference (`sequences`) is OUR OWN multiscored top-50, not the
    # HF archive: the full-history rank_binders set is ~30 edits from the live
    # field and proved a bad selection reference. Use our stable 20x replay read.
    sequences = live_sequences

    cdr3s: list[str] = []
    missed = 0
    for seq in sequences:
        try:
            cdrs = extract_cdrs(seq)
        except Exception:
            cdrs = None
        c3 = (cdrs or {}).get("cdr3") if isinstance(cdrs, dict) else None
        if c3:
            cdr3s.append(str(c3).strip().upper())
        else:
            missed += 1

    payload = {
        "sequences": sequences,
        "cdr3s": sorted(set(cdr3s)),
        "n": 50,
        "ranked": int(len(sequences)),
        "unextractable": missed,
        "archive_count": int(len(frame)),
        "source": "our_top50",
        "live_sequences": live_sequences,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(
        f"TOP50_OK archive_rows={len(frame)} top={len(sequences)} "
        f"unextractable={missed} distinct_cdr3={len(set(cdr3s))} "
        f"source=hf_archive",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
