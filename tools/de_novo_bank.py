"""Convert ``nanobody_opt/recombine.py`` candidates into a native-pass reference bank.

``recombine.py`` emits a proxy-ranked CSV of gate-passing, novel recombinants::

    sequence,max_jaccard_to_top50,proxy_rank_sum,cdr_sweep,exhaustive_cdr,cdr_template

The live sequence phase consumes a reference bank at ``hosts.slave.references``,
whose rows are validated by ``nanobody_opt.pr100_reference_donors.select``. A row
is a donor only: the pipeline grafts its CDR3 into a freshly generated scaffold,
so the bank carries no scores and no target evidence. The schema is::

    {"rows": [{"cells": ["IGHV3-23:16"],
               "sequence": "<AA seq>",
               "sequence_hash": "<sha256 hex>",
               "source": "de_novo_recombine_<timestamp>"}]}

This module is intentionally import-light (stdlib + ``nanobody_opt`` + the pinned
annotation stack) so it can run under the qualified novelty/annotation
interpreter on the master, not under the CUDA scorer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanobody_opt import rules  # noqa: E402
from nanobody_opt.pr100_reference_donors import GENES  # noqa: E402

ALLOWED_GENES = frozenset(GENES)
CDR3_MIN, CDR3_MAX = 5, 23
CELL_RE = re.compile(r"(IGHV3-\d+):(\d+)")


def default_annotate(sequence):
    """The pinned annotation the pipeline itself uses for V gene + CDR3 length."""
    from nanobody_opt.pr100_filter import _annotate_uncached
    return _annotate_uncached(sequence)


def allowed_v_genes(annotation):
    genes = ((annotation or {}).get("germline") or {}).get("V", {}).get("genes") or []
    return sorted({g for g in genes if isinstance(g, str) and g in ALLOWED_GENES})


def convert(sequences, *, source, annotate=None):
    """Return ``(rows, summary)`` for the reference bank.

    ``sequences`` is an iterable of raw amino-acid strings. Deduplication is by
    ``rules.sequence_hash`` and keeps the first occurrence (the recombine CSV is
    already proxy-ranked, so first occurrence preserves the best ranking).
    ``annotate`` defaults to the pinned ``_annotate_uncached`` and may be
    injected for tests. The result is deterministic for a fixed ``source``.
    """
    annotate = annotate or default_annotate
    rows = []
    seen = set()
    summary = {
        "input": 0,
        "kept": 0,
        "dedup_dropped": 0,
        "annotation_failed": 0,
        "cdr3_length_out_of_range": 0,
        "no_allowed_v_gene": 0,
        "v_gene_distribution": Counter(),
        "cdr3_length_distribution": Counter(),
    }
    for sequence in sequences:
        summary["input"] += 1
        if not isinstance(sequence, str) or not sequence.strip():
            summary["annotation_failed"] += 1
            continue
        key = rules.sequence_hash(sequence)
        if key in seen:
            summary["dedup_dropped"] += 1
            continue
        seen.add(key)
        try:
            annotation = annotate(sequence)
        except Exception:
            summary["annotation_failed"] += 1
            continue
        length = (annotation or {}).get("cdr3_length")
        if not isinstance(length, int) or isinstance(length, bool):
            summary["annotation_failed"] += 1
            continue
        if not CDR3_MIN <= length <= CDR3_MAX:
            summary["cdr3_length_out_of_range"] += 1
            continue
        genes = allowed_v_genes(annotation)
        if not genes:
            summary["no_allowed_v_gene"] += 1
            continue
        rows.append({
            "cells": [f"{gene}:{length}" for gene in genes],
            "sequence": sequence.strip().upper(),
            "sequence_hash": key,
            "source": source,
        })
        summary["v_gene_distribution"][genes[0]] += 1
        summary["cdr3_length_distribution"][length] += 1
    summary["kept"] = len(rows)
    return rows, summary


def validate(rows):
    """Reject any row that ``pr100_reference_donors.select`` could not consume.

    Mirrors the identity and cell checks in ``select``'s fallback path plus the
    ``source`` provenance the bank schema requires. Raises ``ValueError`` on the
    first invalid row; returns ``None`` when every row is acceptable.
    """
    for row in rows:
        sequence = row.get("sequence")
        digest = row.get("sequence_hash")
        if (not isinstance(sequence, str) or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or rules.sequence_hash(sequence) != digest):
            raise ValueError("invalid native-pass donor identity")
        cells = row.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("native-pass donor has no cell")
        for value in cells:
            match = CELL_RE.fullmatch(value) if isinstance(value, str) else None
            if not match or match[1] not in GENES or not CDR3_MIN <= int(match[2]) <= CDR3_MAX:
                raise ValueError("invalid native-pass donor cell")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError("native-pass donor has no source")
    return None


def read_sequences(path):
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "sequence" not in reader.fieldnames:
            raise ValueError("recombine CSV is missing the 'sequence' column")
        return [row["sequence"] for row in reader]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def raise_fd_limit(limit=262144):
    """Annotating many sequences spawns one hmmscan per sequence; the box's
    soft ``RLIMIT_NOFILE`` (1024) is below what a full bank sweep needs. Raise
    the soft limit toward the hard limit before the annotation loop."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, hard), hard))
    except (ImportError, ValueError, OSError):
        pass


def main(argv=None):
    raise_fd_limit()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True,
                        help="recombine.py --out CSV (columns: sequence, ...)")
    parser.add_argument("--out", required=True, help="output reference bank JSON")
    parser.add_argument("--summary", default=None,
                        help="optional summary JSON (in -> kept, distributions)")
    parser.add_argument("--source", default=None,
                        help="provenance label (default: de_novo_recombine_<unix>)")
    args = parser.parse_args(argv)

    source = args.source or f"de_novo_recombine_{int(time.time())}"
    rows, summary = convert(read_sequences(args.candidates), source=source)
    validate(rows)
    bank = {"rows": rows}
    atomic_json(args.out, bank)
    if args.summary:
        atomic_json(args.summary, summary)
    print(json.dumps({
        "bank": args.out,
        "source": source,
        "input": summary["input"],
        "kept": summary["kept"],
        "dedup_dropped": summary["dedup_dropped"],
        "annotation_failed": summary["annotation_failed"],
        "cdr3_length_out_of_range": summary["cdr3_length_out_of_range"],
        "no_allowed_v_gene": summary["no_allowed_v_gene"],
        "v_gene_distribution": dict(summary["v_gene_distribution"]),
        "cdr3_length_distribution": dict(summary["cdr3_length_distribution"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
