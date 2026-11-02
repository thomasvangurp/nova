"""PLAN path for the challenge backfill: derive each missing epoch's challenge from
the validator-accepted submissions in the local observation DB.

Source of truth:  accepted submissions (V gene + CDR3 length are validator-enforced).
Verification:     rules.challenge(config, epoch, block_hash, epoch_length) must agree.
Write path:       unchanged -- ledger.record_challenge(...) in apply, then chain verify.

Validator acceptance semantics (neurons/validator/nanobody_validity.py):
  * V gene  (l.164): any(allowed_v in germline['V']['genes'] for seq in uid_seqs)
                     -> 'genes' is a TIE LIST; a sequence votes for every tied gene.
  * CDR3    (l.206): every chain's len(cdr3_seq) == allowed_cdr3_length.
  * nanobody_requirements.num_sequences == 1, so per-sequence == per-UID voting.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from nanobody_opt import rules

CONFIG_NAME = "nanobody_pr100.json"            # single git commit -> one ruleset era
DERIVATION_VERSION = "accepted-submissions-v1"
SOURCE = "validator_accepted_submissions"
VERIFY_DISAGREES = "submission-derived challenge disagrees with block-hash re-derivation"
CELL_RE = re.compile(r"(IGHV3-\d+):(\d+)")     # mirrors tools/de_novo_bank.CELL_RE

ACCEPTED_SQL = """
SELECT n.sequence, o.metrics_json, o.provenance_json
FROM observations AS o JOIN nodes AS n ON n.id = o.node_id
WHERE o.source = 'public_validator_api'
  AND o.score_kind = 'validator_final_rank_sum'
  AND o.epoch = ? AND n.target = ?
"""

Annotation = Optional[tuple[frozenset, int]]   # (tied category genes, cdr3_length)
Annotator = Callable[[str], Annotation]


class AnnotationRuntimeError(RuntimeError):
    """Pinned annotator unavailable on this host. Aborts the plan: no silent fallback."""


# --------------------------------------------------------------------------- annotation
def cells_to_annotation(cells) -> Annotation:
    """Parse stored ["<V_gene>:<cdr3_length>", ...] (one cell per tied gene)."""
    if not isinstance(cells, list) or not cells:
        return None                      # empty cells == "not annotated", never "no gene"
    genes, lengths = set(), set()
    for cell in cells:
        match = CELL_RE.fullmatch(cell) if isinstance(cell, str) else None
        if not match:
            return None
        genes.add(match[1])
        lengths.add(int(match[2]))
    return (frozenset(genes), lengths.pop()) if len(lengths) == 1 else None


def stored_cells(*json_blobs):
    for blob in json_blobs:
        try:
            cells = json.loads(blob or "{}").get("cells")
        except (ValueError, AttributeError):
            continue
        if cells:
            return cells
    return None


def pinned_annotator(cache_dir) -> Annotator:
    """Re-annotation via the repo's validator-parity annotator (lazy init)."""
    state = {}

    def annotate(sequence: str) -> Annotation:
        if not state:
            try:
                from nanobody_opt import pr100_filter
                from tools import de_novo_bank
                pr100_filter.activate_filter()
                pr100_filter.validate_annotation_runtime()   # versions, HMM panel, hmmscan sha
            except Exception as error:
                raise AnnotationRuntimeError(
                    f"pinned PR100 annotation runtime unavailable ({type(error).__name__}: "
                    f"{error}); run `plan` on a host with the pinned runtime") from error
            state.update(filter=pr100_filter, bank=de_novo_bank)
        result = state["filter"].annotate(sequence, cache_dir)   # cache keyed by identity
        length = result.get("cdr3_length")
        if result.get("status") != "ok" or type(length) is not int:
            return None
        return frozenset(state["bank"].allowed_v_genes(result)), length

    annotate.identity = lambda: (state["filter"].annotation_identity() if state else None)
    return annotate


# --------------------------------------------------------------------------- derivation
def resolve_category_key(categories: dict, gene: str):
    keys = sorted(key for key, genes in categories.items() if gene in genes)
    if not keys:
        return None
    return keys[0] if len(keys) == 1 else "ambiguous"


def derive_from_accepted(obs_conn, epoch: int, *, target: str, categories: dict,
                         annotate: Annotator) -> dict:
    """Majority (V gene, CDR3 length) over validator-accepted sequences for `epoch`.

    Every accepted sequence counts in the denominator; unannotatable ones cast no
    vote, so annotation failure can only lower support, never manufacture a majority.
    """
    sequences = {}
    for sequence, metrics_json, provenance_json in obs_conn.execute(
            ACCEPTED_SQL, (int(epoch), target)):
        sequence = sequence.strip().upper()
        sequences.setdefault(rules.sequence_hash(sequence),
                             (sequence, stored_cells(provenance_json, metrics_json)))
    total = len(sequences)
    evidence = {"accepted": total, "from_stored_cells": 0, "reannotated": 0,
                "unannotatable": 0, "no_category_gene": 0, "candidates": []}
    if not total:
        return {"status": "skip", "reason": "no validator-accepted submissions in local DB",
                "evidence": evidence}

    votes = Counter()
    for sequence, cells in sequences.values():
        annotation = cells_to_annotation(cells)
        if annotation is not None:
            evidence["from_stored_cells"] += 1
        else:
            annotation = annotate(sequence)      # AnnotationRuntimeError propagates
            evidence["reannotated" if annotation else "unannotatable"] += 1
        if annotation is None:
            continue
        genes, length = annotation
        if not genes:
            evidence["no_category_gene"] += 1
            continue
        for gene in genes:                       # tied genes: each gets this sequence's vote
            votes[(gene, length)] += 1

    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    evidence["candidates"] = [{"allowed_v": g, "allowed_cdr3_length": n, "support": s}
                              for (g, n), s in ranked[:5]]
    if not ranked:
        return {"status": "skip", "reason": "no accepted submission could be annotated",
                "evidence": evidence}
    (gene, length), support = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == support:
        tied = sorted(f"{g}:{n}" for (g, n), s in ranked if s == support)
        return {"status": "skip", "evidence": evidence,
                "reason": f"split: {len(tied)} pairs tied at support {support}/{total}: {tied}"}
    if 2 * support <= total:
        return {"status": "skip", "evidence": evidence,
                "reason": f"no clear majority: {gene}:{length} has {support}/{total}"}
    category_key = resolve_category_key(categories, gene)
    if category_key is None:
        return {"status": "skip", "evidence": evidence,
                "reason": f"majority V gene {gene} is in no configured V category"}
    evidence.update(support=support, unanimous=support == total)
    return {"status": "ok", "allowed_v": gene, "allowed_cdr3_length": length,
            "category_key": category_key, "evidence": evidence}


# --------------------------------------------------------------------------- plan
def build_plan(obs_conn, epochs, *, epoch_length: int, fetch_block_hash,
               annotate: Annotator, config_path: Path | None = None) -> dict:
    """Plan entries for the given (already ledger-missing) epochs.

    fetch_block_hash(block_number) -> "0x..."  : the existing H6 archive-RPC fetch.
    action="record" (challenge + provenance for record_challenge) or "skip" + reason.
    """
    config_path = Path(config_path or rules.CONFIG_DIR / CONFIG_NAME)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    ruleset_sha256 = rules.fingerprint(config)
    target = str(config["protein_selection"]["nanobody"]["target"])
    categories = config["germline_filtering"]["v_neighborhood_categories"]
    if config["nanobody_requirements"].get("num_sequences") != 1:
        raise ValueError("per-sequence voting requires num_sequences == 1 "
                         "(validator V check is any-of per UID)")

    entries = []
    for epoch in sorted({int(e) for e in epochs}):
        entry = {"epoch": epoch, "action": "skip", "reason": None, "challenge": None,
                 "category_key": None, "derivation_version": DERIVATION_VERSION,
                 "source": SOURCE, "provenance": None}
        entries.append(entry)

        # Constant-era guard: the pinned config for this epoch must be this config.
        if epoch < rules.ACTIVATION_EPOCH:
            entry["reason"] = f"epoch precedes PR100 activation ({rules.ACTIVATION_EPOCH})"
            continue
        if rules.fingerprint(rules.pinned_config(epoch)) != ruleset_sha256:
            entry["reason"] = "config file differs from the pinned PR100 config"
            continue

        # 1. Source: accepted submissions (local, cheap -> before any RPC call).
        derived = derive_from_accepted(obs_conn, epoch, target=target,
                                       categories=categories, annotate=annotate)
        evidence = derived["evidence"]
        if derived["status"] != "ok":
            entry.update(reason=derived["reason"], provenance={"evidence": evidence})
            continue

        # 2. Block hash (H6): needed for the challenge and record_challenge's own check.
        start_block = epoch * int(epoch_length)
        try:
            block_hash = fetch_block_hash(start_block)
        except Exception as error:               # type only: RPC URLs may carry keys
            block_hash, fetch_error = None, type(error).__name__
        else:
            fetch_error = None
        if not (isinstance(block_hash, str) and re.fullmatch(r"0x[0-9a-fA-F]{64}", block_hash)):
            entry.update(provenance={"evidence": evidence},
                         reason=f"epoch-start block hash unavailable from archive RPC"
                                f" (block {start_block}; {fetch_error or 'empty/invalid'})")
            continue
        block_hash = block_hash.lower()

        challenge = {"epoch": epoch, "target": target, "epoch_length": int(epoch_length),
                     "epoch_start_block": start_block, "epoch_start_block_hash": block_hash,
                     "ruleset_sha256": ruleset_sha256,
                     "allowed_v": derived["allowed_v"],
                     "allowed_cdr3_length": derived["allowed_cdr3_length"]}

        # 3. Verification: block-hash re-derivation must reproduce the challenge exactly.
        try:
            expected = rules.challenge(config, epoch, block_hash, epoch_length)
        except ValueError as error:
            entry.update(provenance={"evidence": evidence},
                         reason=f"block-hash re-derivation failed: {error}")
            continue
        if challenge != expected:
            fields = sorted(k for k in expected.keys() | challenge.keys()
                            if expected.get(k) != challenge.get(k))
            entry.update(reason=VERIFY_DISAGREES, provenance={
                "evidence": evidence, "differing_fields": fields,
                "block_hash_derived": {k: expected.get(k) for k in fields}})
            continue

        entry.update(action="record", challenge=challenge,
                     category_key=derived["category_key"], provenance={
                         "derived_from": SOURCE,
                         "verified_by": "rules.challenge(block_hash)",
                         "config_file": CONFIG_NAME,
                         "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                         "ruleset_sha256": ruleset_sha256,
                         "category_key": derived["category_key"],
                         "annotation_identity": getattr(annotate, "identity", lambda: None)(),
                         "evidence": evidence})

    return {"derivation_version": DERIVATION_VERSION, "config_file": CONFIG_NAME,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "ruleset_sha256": ruleset_sha256, "entries": entries,
            "summary": dict(Counter(e["action"] for e in entries))}
