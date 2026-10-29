"""Build a PR100 native-pass reference bank from a competitor's submissions.

The rival-parent lane is *opt-in* and additive: this tool produces a bank the
operator can point ``hosts.slave.references`` at, it does not modify the live
config. It reuses ``tools/de_novo_bank`` for the annotation/convert/validate
half, so the emitted rows carry the exact schema ``pr100_reference_donors``
consumes::

    {"rows": [{"cells": ["IGHV3-23:16"],
               "sequence": "<AA seq>",
               "sequence_hash": "<sha256 hex>",
               "source": "rival_bank_<unix>"}]}

The input is ``competitor_lane.fetch_rival_sequences``: the rival's recent
leaderboard submissions over the last ``--n-epochs`` epochs, best (lowest
validator ``final_score``) first. Every sequence is then annotated with the
pinned ``pr100_filter._annotate_uncached`` stack, filtered to the 9-gene
IGHV3 set + CDR3 length 5-23, and deduped by ``rules.sequence_hash`` -- the
same conversion path as ``tools/de_novo_bank``.

This module is intentionally import-light (stdlib + ``nanobody_opt``) so it
runs under the qualified annotation interpreter, not under the CUDA scorer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nanobody_opt.competitor_lane import fetch_rival_sequences
from tools import de_novo_bank


def build(epoch, coldkey_prefix, *, n_epochs=15, source=None, annotate=None):
    """Return ``(bank, summary, sequences)`` for the rival-parent bank.

    ``bank`` is the ``{"rows": [...]}`` payload in the exact
    ``pr100_reference_donors`` schema. ``sequences`` is the raw deduped rival
    submission list (best first) that ``fetch_rival_sequences`` returned, for
    provenance. ``annotate`` defaults to the pinned ``_annotate_uncached`` and
    is injectable for tests (see ``tests/test_build_rival_bank.py``).
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    if not isinstance(coldkey_prefix, str) or not coldkey_prefix.strip():
        raise ValueError('rival coldkey prefix must be a nonempty string')
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, int) or n_epochs < 1:
        raise ValueError('n_epochs must be a positive integer')
    sequences = fetch_rival_sequences(epoch, coldkey_prefix, n_epochs=n_epochs)
    source = source or f"rival_bank_{int(time.time())}"
    rows, summary = de_novo_bank.convert(sequences, source=source, annotate=annotate)
    de_novo_bank.validate(rows)
    return {"rows": rows}, summary, sequences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", required=True, type=int,
                        help="current epoch; rival submissions are fetched "
                             "from the previous n_epochs epochs")
    parser.add_argument("--rival-coldkey", required=True,
                        help="coldkey prefix of the competitor whose recent "
                             "submissions become the bank parents")
    parser.add_argument("--out", required=True, type=Path,
                        help="output reference bank JSON")
    parser.add_argument("--n-epochs", type=int, default=15,
                        help="how many prior epochs to fetch (default 15)")
    parser.add_argument("--source", default=None,
                        help="provenance label (default: rival_bank_<unix>)")
    parser.add_argument("--summary", default=None, type=Path,
                        help="optional summary JSON (in -> kept, distributions)")
    args = parser.parse_args(argv)

    bank, summary, sequences = build(args.epoch, args.rival_coldkey,
                                     n_epochs=args.n_epochs, source=args.source)
    de_novo_bank.atomic_json(args.out, bank)
    if args.summary:
        de_novo_bank.atomic_json(args.summary, summary)
    print(json.dumps({
        "bank": str(args.out),
        "source": bank["rows"][0]["source"] if bank["rows"] else (args.source or ""),
        "fetched_rival_sequences": len(sequences),
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
