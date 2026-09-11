"""Versioned, policy-gated external VHH evidence storage.

This module is deliberately separate from the scored mutation-network path.
External sequences and structural records never become NOVA ``nodes`` or
``observations``.  They can only be linked to an already-existing node by an
explicit, cutoff-dated sequence match.

The first adapter accepts a locally downloaded, SHA-256-pinned SAbDab2
single-domain summary.  It retains source rows and structural provenance but
does not infer antigen accessions, affinities, or P20809 binding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


EXTERNAL_EVIDENCE_SCHEMA_VERSION = 1
SABDAB2_LICENSE = "CC-BY-4.0"
SABDAB2_HOST = "sabdab.opig.stats.ox.ac.uk"
SABDAB2_SUMMARY_PATH = "/api/download/all-single-domain-summary"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ExternalEvidenceError(ValueError):
    """Raised when an external artifact fails a provenance or quality gate."""


class DatasetPolicyError(ExternalEvidenceError):
    """Raised when a dataset is not permitted for this evidence store."""


@dataclass(frozen=True)
class DatasetRelease:
    """Immutable provenance and policy declaration for one source artifact."""

    dataset_id: str
    name: str
    release: str
    release_date: str
    retrieved_at: str
    source_url: str
    license_id: str
    commercial_use_allowed: bool
    tdm_ai_allowed: bool
    redistribution_allowed: bool
    artifact_sha256: str
    terms_snapshot_uri: str

    @classmethod
    def sabdab2(
        cls,
        *,
        dataset_id: str,
        release: str,
        release_date: str,
        retrieved_at: str,
        source_url: str,
        artifact_sha256: str,
        terms_snapshot_uri: str,
    ) -> "DatasetRelease":
        """Build the fixed policy declaration for a SAbDab2 CC-BY release."""
        return cls(
            dataset_id=dataset_id,
            name="SAbDab2 single-domain summary",
            release=release,
            release_date=release_date,
            retrieved_at=retrieved_at,
            source_url=source_url,
            license_id=SABDAB2_LICENSE,
            commercial_use_allowed=True,
            tdm_ai_allowed=True,
            redistribution_allowed=True,
            artifact_sha256=artifact_sha256.lower(),
            terms_snapshot_uri=terms_snapshot_uri,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_iso_date(value: str, *, field: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ExternalEvidenceError(
            f"{field} must be an ISO date (YYYY-MM-DD): {value!r}"
        ) from error
    if parsed.isoformat() != value:
        raise ExternalEvidenceError(
            f"{field} must be an ISO date (YYYY-MM-DD): {value!r}"
        )
    return parsed


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ExternalEvidenceError(
            f"{field} must be an ISO timestamp with timezone: {value!r}"
        ) from error
    if parsed.tzinfo is None:
        raise ExternalEvidenceError(f"{field} must include a timezone: {value!r}")
    return parsed


def _source_date(value: Any, *, field: str) -> date | None:
    text = str(value or "").strip()
    if not text or text.upper() == "NA":
        return None
    for pattern in ("%Y%m%d", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    raise ExternalEvidenceError(f"invalid {field}: {text!r}")


def _optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if not text or text.upper() == "NA" else text


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}:{digest.hexdigest()}"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_release(release: DatasetRelease, artifact: Path) -> date:
    if not release.dataset_id.startswith("sabdab2:"):
        raise DatasetPolicyError("SAbDab2 dataset_id must start with 'sabdab2:'")
    if release.name != "SAbDab2 single-domain summary":
        raise DatasetPolicyError("only the SAbDab2 summary adapter is supported")
    if release.release.strip().lower() in {"", "current", "latest", "unversioned"}:
        raise DatasetPolicyError("a pinned SAbDab2 release identifier is required")
    release_day = _parse_iso_date(release.release_date, field="release_date")
    retrieved = _parse_timestamp(release.retrieved_at, field="retrieved_at")
    if retrieved.date() < release_day:
        raise ExternalEvidenceError("retrieved_at precedes release_date")
    parsed_url = urlparse(release.source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != SABDAB2_HOST
        or parsed_url.path.rstrip("/") != SABDAB2_SUMMARY_PATH
        or parsed_url.query
    ):
        raise DatasetPolicyError(f"SAbDab2 source_url must be HTTPS on {SABDAB2_HOST}")
    if release.license_id != SABDAB2_LICENSE:
        raise DatasetPolicyError(f"SAbDab2 license must be {SABDAB2_LICENSE}")
    if (
        not release.commercial_use_allowed
        or not release.tdm_ai_allowed
        or not release.redistribution_allowed
    ):
        raise DatasetPolicyError(
            "commercial, TDM/AI, and redistribution permission must be explicit"
        )
    if not release.terms_snapshot_uri.strip():
        raise DatasetPolicyError("terms_snapshot_uri is required")
    expected = release.artifact_sha256.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ExternalEvidenceError("artifact_sha256 must be 64 lowercase hex digits")
    actual = sha256_file(artifact)
    if actual != expected:
        raise ExternalEvidenceError(
            f"artifact SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return release_day


def _connect_graph(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise ExternalEvidenceError(
            "external evidence requires an existing mutation-network SQLite store"
        )
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not {"metadata", "nodes", "observations"}.issubset(tables):
        connection.close()
        raise ExternalEvidenceError(
            "database is not an initialized mutation-network store"
        )
    return connection


def initialize_external_schema(path: Path) -> None:
    """Add the versioned external-evidence extension to a graph database."""
    with _connect_graph(path) as connection:
        version = connection.execute(
            "SELECT value FROM metadata WHERE key='external_evidence_schema_version'"
        ).fetchone()
        if version is not None and str(version[0]) != str(
            EXTERNAL_EVIDENCE_SCHEMA_VERSION
        ):
            raise ExternalEvidenceError(
                f"unsupported external evidence schema version {version[0]}"
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS external_datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                release TEXT NOT NULL,
                release_date TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                license_id TEXT NOT NULL,
                commercial_use_allowed INTEGER NOT NULL CHECK(commercial_use_allowed IN (0, 1)),
                tdm_ai_allowed INTEGER NOT NULL CHECK(tdm_ai_allowed IN (0, 1)),
                redistribution_allowed INTEGER NOT NULL CHECK(redistribution_allowed IN (0, 1)),
                artifact_sha256 TEXT NOT NULL,
                terms_snapshot_uri TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS external_records (
                record_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES external_datasets(dataset_id),
                source_accession TEXT NOT NULL,
                source_url TEXT NOT NULL,
                released_at TEXT NOT NULL,
                source_updated_at TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                ingest_status TEXT NOT NULL CHECK(ingest_status IN ('usable', 'quarantined')),
                quality_flags_json TEXT NOT NULL DEFAULT '[]',
                raw_json TEXT NOT NULL,
                UNIQUE(dataset_id, source_accession)
            );
            CREATE INDEX IF NOT EXISTS external_records_dataset_date
                ON external_records(dataset_id, released_at, source_updated_at);
            CREATE TABLE IF NOT EXISTS external_vhh (
                vhh_id TEXT PRIMARY KEY,
                sequence_sha256 TEXT NOT NULL UNIQUE,
                raw_sequence TEXT NOT NULL,
                variable_sequence TEXT NOT NULL,
                imgt_sequence TEXT,
                length INTEGER NOT NULL,
                modality TEXT NOT NULL,
                source_species TEXT,
                cdr1 TEXT,
                cdr2 TEXT,
                cdr3 TEXT,
                numbering_scheme TEXT,
                quality_flags_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS external_antigens (
                antigen_id TEXT PRIMARY KEY,
                sequence_sha256 TEXT,
                uniprot_accession TEXT,
                isoform TEXT,
                sequence_version TEXT,
                name TEXT,
                organism_taxid TEXT,
                raw_sequence TEXT,
                mature_sequence TEXT,
                quality_flags_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS external_interactions (
                interaction_id TEXT PRIMARY KEY,
                vhh_id TEXT NOT NULL REFERENCES external_vhh(vhh_id),
                antigen_id TEXT NOT NULL REFERENCES external_antigens(antigen_id),
                record_id TEXT NOT NULL REFERENCES external_records(record_id),
                evidence_kind TEXT NOT NULL,
                raw_value TEXT,
                value REAL,
                unit TEXT,
                assay TEXT,
                experimental_or_predicted TEXT NOT NULL,
                confidence TEXT,
                conditions_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS external_complexes (
                complex_id TEXT PRIMARY KEY,
                interaction_id TEXT REFERENCES external_interactions(interaction_id),
                record_id TEXT NOT NULL REFERENCES external_records(record_id),
                pdb_id TEXT NOT NULL,
                assembly_id TEXT,
                model_id TEXT,
                vhh_chain_ids TEXT,
                antigen_chain_ids TEXT,
                method TEXT,
                resolution REAL,
                structure_sha256 TEXT,
                structure_uri TEXT,
                release_date TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS external_record_links (
                record_id TEXT NOT NULL REFERENCES external_records(record_id),
                entity_kind TEXT NOT NULL
                  CHECK(entity_kind IN ('vhh', 'antigen', 'interaction', 'complex')),
                entity_id TEXT NOT NULL,
                PRIMARY KEY(record_id, entity_kind, entity_id)
            );
            CREATE TABLE IF NOT EXISTS external_nova_links (
                dataset_id TEXT NOT NULL REFERENCES external_datasets(dataset_id),
                record_id TEXT NOT NULL REFERENCES external_records(record_id),
                external_vhh_id TEXT NOT NULL REFERENCES external_vhh(vhh_id),
                nova_node_id INTEGER NOT NULL REFERENCES nodes(id),
                alignment_method TEXT NOT NULL,
                sequence_identity REAL NOT NULL,
                query_coverage REAL NOT NULL,
                target_coverage REAL NOT NULL,
                cdr1_identity REAL,
                cdr2_identity REAL,
                cdr3_identity REAL,
                is_exact INTEGER NOT NULL CHECK(is_exact IN (0, 1)),
                cutoff_snapshot TEXT NOT NULL,
                PRIMARY KEY(
                    record_id, nova_node_id, alignment_method, cutoff_snapshot
                )
            );
            CREATE VIEW IF NOT EXISTS eligible_external_nova_links AS
            SELECT link.*
            FROM external_nova_links AS link
            JOIN external_datasets AS dataset
              ON dataset.dataset_id=link.dataset_id
            JOIN external_records AS record
              ON record.record_id=link.record_id
            WHERE dataset.license_id='CC-BY-4.0'
              AND dataset.commercial_use_allowed=1
              AND dataset.tdm_ai_allowed=1
              AND dataset.redistribution_allowed=1
              AND record.ingest_status='usable'
              AND date(dataset.release_date) <= date(link.cutoff_snapshot)
              AND date(record.released_at) <= date(link.cutoff_snapshot)
              AND date(record.source_updated_at) <= date(link.cutoff_snapshot);
            CREATE TABLE IF NOT EXISTS proposal_external_features (
                epoch INTEGER NOT NULL,
                node_id INTEGER NOT NULL REFERENCES nodes(id),
                feature_version TEXT NOT NULL,
                evidence_cutoff TEXT NOT NULL,
                dataset_manifest_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                PRIMARY KEY(epoch, node_id, feature_version)
            );
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO metadata(key, value)
            VALUES('external_evidence_schema_version', ?)
            """,
            (str(EXTERNAL_EVIDENCE_SCHEMA_VERSION),),
        )


_REQUIRED_SABDAB2_COLUMNS = {
    "INSTANCE",
    "PDB",
    "SABDAB_ID",
    "Hchain",
    "VH",
    "CDR-H1",
    "CDR-H2",
    "CDR-H3",
    "model",
    "antigen_chain",
    "antigen_type",
    "antigen_name",
    "date",
    "SABDABdepo_date",
    "SABDABupdate_date",
    "heavy_species",
    "antigen_species",
    "agtaxid",
    "method",
    "resolution",
    "type",
}


def _read_sabdab2_rows(
    path: Path, *, release_day: date
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_SABDAB2_COLUMNS - fields)
        if missing:
            raise ExternalEvidenceError(
                f"SAbDab2 summary is missing required columns: {missing}"
            )
        rows = list(reader)

    prepared: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    summary = {
        "rows_read": len(rows),
        "non_sdh_rows_skipped": 0,
        "sdh_records": 0,
        "usable_vhh_records": 0,
        "quarantined_vhh_records": 0,
        "bound_complex_records": 0,
    }
    for raw in rows:
        if str(raw.get("type", "")).strip().upper() != "SD-H":
            summary["non_sdh_rows_skipped"] += 1
            continue
        instance = str(raw.get("INSTANCE", "")).strip()
        if not instance:
            raise ExternalEvidenceError("SAbDab2 SD-H row has no INSTANCE")
        if instance in seen_instances:
            raise ExternalEvidenceError(f"duplicate SAbDab2 INSTANCE: {instance}")
        seen_instances.add(instance)
        summary["sdh_records"] += 1

        released_at = _source_date(
            raw.get("SABDABdepo_date") or raw.get("date"),
            field=f"{instance} released_at",
        )
        updated_at = _source_date(
            raw.get("SABDABupdate_date")
            or raw.get("SABDABdepo_date")
            or raw.get("date"),
            field=f"{instance} source_updated_at",
        )
        if released_at is None or updated_at is None:
            raise ExternalEvidenceError(
                f"SAbDab2 row {instance} has no availability date"
            )
        if released_at > release_day or updated_at > release_day:
            raise ExternalEvidenceError(
                f"SAbDab2 row {instance} postdates declared release {release_day}"
            )

        sequence = "".join(str(raw.get("VH", "")).split()).upper()
        quality_flags: list[str] = []
        unknown = sorted(set(sequence) - AMINO_ACIDS)
        if unknown:
            quality_flags.append(f"noncanonical_residues:{''.join(unknown)}")
        if not 80 <= len(sequence) <= 180:
            quality_flags.append(f"variable_length:{len(sequence)}")
        for field, flag in (
            ("PDB", "missing_pdb"),
            ("SABDAB_ID", "missing_sabdab_id"),
            ("Hchain", "missing_vhh_chain"),
        ):
            if _optional(raw.get(field)) is None:
                quality_flags.append(flag)
        usable = not quality_flags
        if usable:
            summary["usable_vhh_records"] += 1
        else:
            summary["quarantined_vhh_records"] += 1
        if _optional(raw.get("antigen_chain")) is not None:
            summary["bound_complex_records"] += 1
        prepared.append(
            {
                "raw": raw,
                "instance": instance,
                "released_at": released_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "sequence": sequence,
                "quality_flags": quality_flags,
                "usable": usable,
            }
        )
    return prepared, summary


def _insert_immutable_dataset(
    connection: sqlite3.Connection, release: DatasetRelease
) -> None:
    columns = tuple(asdict(release))
    values = tuple(
        int(value) if isinstance(value, bool) else value
        for value in asdict(release).values()
    )
    existing = connection.execute(
        f"SELECT {', '.join(columns)} FROM external_datasets WHERE dataset_id=?",
        (release.dataset_id,),
    ).fetchone()
    if existing is not None:
        if tuple(existing) != values:
            raise ExternalEvidenceError(
                f"dataset_id {release.dataset_id!r} already names a different artifact"
            )
        return
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO external_datasets({', '.join(columns)}) VALUES({placeholders})",
        values,
    )


def _float_or_none(value: Any) -> float | None:
    text = _optional(value)
    if text is None:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def ingest_sabdab2_summary(
    database: Path,
    summary_csv: Path,
    release: DatasetRelease,
) -> dict[str, Any]:
    """Atomically ingest a pinned SAbDab2 SD-H summary into external tables."""
    release_day = _validate_release(release, summary_csv)
    prepared, summary = _read_sabdab2_rows(summary_csv, release_day=release_day)
    initialize_external_schema(database)

    with _connect_graph(database) as connection:
        _insert_immutable_dataset(connection, release)
        for item in prepared:
            raw = item["raw"]
            instance = item["instance"]
            record_id = _stable_id("record", release.dataset_id, instance)
            raw_json = _json(raw)
            record_sha256 = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO external_records(
                    record_id, dataset_id, source_accession, source_url,
                    released_at, source_updated_at, record_sha256,
                    ingest_status, quality_flags_json, raw_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, source_accession) DO UPDATE SET
                    record_id=excluded.record_id,
                    source_url=excluded.source_url,
                    released_at=excluded.released_at,
                    source_updated_at=excluded.source_updated_at,
                    record_sha256=excluded.record_sha256,
                    ingest_status=excluded.ingest_status,
                    quality_flags_json=excluded.quality_flags_json,
                    raw_json=excluded.raw_json
                """,
                (
                    record_id,
                    release.dataset_id,
                    instance,
                    release.source_url,
                    item["released_at"],
                    item["updated_at"],
                    record_sha256,
                    "usable" if item["usable"] else "quarantined",
                    _json(item["quality_flags"]),
                    raw_json,
                ),
            )
            if not item["usable"]:
                continue

            sequence = item["sequence"]
            sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            vhh_id = f"vhh:{sequence_sha256}"
            connection.execute(
                """
                INSERT OR IGNORE INTO external_vhh(
                    vhh_id, sequence_sha256, raw_sequence, variable_sequence,
                    imgt_sequence, length, modality, source_species,
                    cdr1, cdr2, cdr3, numbering_scheme, quality_flags_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IMGT', '[]')
                """,
                (
                    vhh_id,
                    sequence_sha256,
                    str(raw.get("VH", "")),
                    sequence,
                    sequence,
                    len(sequence),
                    "SD-H",
                    _optional(raw.get("heavy_species")),
                    _optional(raw.get("CDR-H1")),
                    _optional(raw.get("CDR-H2")),
                    _optional(raw.get("CDR-H3")),
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO external_record_links VALUES(?, 'vhh', ?)",
                (record_id, vhh_id),
            )

            antigen_chain = _optional(raw.get("antigen_chain"))
            antigen_id = None
            interaction_id = None
            if antigen_chain is not None:
                antigen_id = _stable_id(
                    "antigen",
                    raw.get("PDB"),
                    raw.get("model"),
                    antigen_chain,
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO external_antigens(
                        antigen_id, name, organism_taxid, quality_flags_json
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        antigen_id,
                        _optional(raw.get("antigen_name")),
                        _optional(raw.get("agtaxid")),
                        _json(["summary_only", "uniprot_unmapped"]),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO external_record_links VALUES(?, 'antigen', ?)",
                    (record_id, antigen_id),
                )
                interaction_id = _stable_id(
                    "interaction",
                    vhh_id,
                    antigen_id,
                    raw.get("PDB"),
                    raw.get("model"),
                    raw.get("Hchain"),
                    antigen_chain,
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO external_interactions(
                        interaction_id, vhh_id, antigen_id, record_id,
                        evidence_kind, experimental_or_predicted,
                        conditions_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        vhh_id,
                        antigen_id,
                        record_id,
                        "experimentally_determined_complex",
                        "experimental_structure",
                        _json(
                            {
                                "antigen_chain": antigen_chain,
                                "antigen_name": _optional(raw.get("antigen_name")),
                                "antigen_species": _optional(
                                    raw.get("antigen_species")
                                ),
                                "antigen_type": _optional(raw.get("antigen_type")),
                                "model": _optional(raw.get("model")),
                            }
                        ),
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO external_record_links
                    VALUES(?, 'interaction', ?)
                    """,
                    (record_id, interaction_id),
                )

            complex_id = _stable_id(
                "complex",
                raw.get("PDB"),
                raw.get("model"),
                raw.get("Hchain"),
                antigen_chain or "",
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO external_complexes(
                    complex_id, interaction_id, record_id, pdb_id, assembly_id,
                    model_id, vhh_chain_ids, antigen_chain_ids, method,
                    resolution, release_date
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    complex_id,
                    interaction_id,
                    record_id,
                    str(raw.get("PDB", "")).strip(),
                    None,
                    _optional(raw.get("model")),
                    _optional(raw.get("Hchain")),
                    antigen_chain,
                    _optional(raw.get("method")),
                    _float_or_none(raw.get("resolution")),
                    item["released_at"],
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO external_record_links VALUES(?, 'complex', ?)",
                (record_id, complex_id),
            )

    return {
        "dataset_id": release.dataset_id,
        "artifact_sha256": release.artifact_sha256,
        **summary,
    }


def eligible_sabdab2_vhh(
    database: Path,
    *,
    as_of: str,
) -> list[dict[str, Any]]:
    """Return record-level VHH evidence satisfying policy and temporal gates."""
    cutoff = _parse_iso_date(as_of, field="as_of").isoformat()
    initialize_external_schema(database)
    with _connect_graph(database) as connection:
        rows = connection.execute(
            """
            SELECT
                d.dataset_id, d.release, d.release_date, d.artifact_sha256,
                d.license_id, r.record_id, r.source_accession,
                r.released_at, r.source_updated_at, v.*
            FROM external_datasets AS d
            JOIN external_records AS r ON r.dataset_id=d.dataset_id
            JOIN external_record_links AS l
              ON l.record_id=r.record_id AND l.entity_kind='vhh'
            JOIN external_vhh AS v ON v.vhh_id=l.entity_id
            WHERE d.name='SAbDab2 single-domain summary'
              AND d.license_id=?
              AND d.commercial_use_allowed=1
              AND d.tdm_ai_allowed=1
              AND d.redistribution_allowed=1
              AND r.ingest_status='usable'
              AND date(d.release_date) <= date(?)
              AND date(r.released_at) <= date(?)
              AND date(r.source_updated_at) <= date(?)
            ORDER BY r.source_accession, v.vhh_id
            """,
            (SABDAB2_LICENSE, cutoff, cutoff, cutoff),
        ).fetchall()
    return [dict(row) for row in rows]


def link_exact_nova_sequences(
    database: Path,
    *,
    target: str,
    as_of: str,
) -> int:
    """Link eligible external VHHs to existing NOVA nodes by exact sequence."""
    target = target.strip()
    if not target:
        raise ExternalEvidenceError("target is required for an exact NOVA link")
    cutoff = _parse_iso_date(as_of, field="as_of").isoformat()
    initialize_external_schema(database)
    with _connect_graph(database) as connection:
        before = connection.total_changes
        connection.execute(
            """
            INSERT OR IGNORE INTO external_nova_links(
                dataset_id, record_id, external_vhh_id, nova_node_id,
                alignment_method,
                sequence_identity, query_coverage, target_coverage,
                cdr1_identity, cdr2_identity, cdr3_identity,
                is_exact, cutoff_snapshot
            )
            SELECT DISTINCT
                d.dataset_id, r.record_id, v.vhh_id, n.id,
                'exact_sha256', 1.0, 1.0, 1.0,
                CASE WHEN v.cdr1 IS NULL THEN NULL ELSE 1.0 END,
                CASE WHEN v.cdr2 IS NULL THEN NULL ELSE 1.0 END,
                CASE WHEN v.cdr3 IS NULL THEN NULL ELSE 1.0 END,
                1, ?
            FROM external_datasets AS d
            JOIN external_records AS r ON r.dataset_id=d.dataset_id
            JOIN external_record_links AS l
              ON l.record_id=r.record_id AND l.entity_kind='vhh'
            JOIN external_vhh AS v ON v.vhh_id=l.entity_id
            JOIN nodes AS n ON n.sequence=v.variable_sequence AND n.target=?
            WHERE d.name='SAbDab2 single-domain summary'
              AND d.license_id=?
              AND d.commercial_use_allowed=1
              AND d.tdm_ai_allowed=1
              AND d.redistribution_allowed=1
              AND r.ingest_status='usable'
              AND date(d.release_date) <= date(?)
              AND date(r.released_at) <= date(?)
              AND date(r.source_updated_at) <= date(?)
            """,
            (cutoff, target, SABDAB2_LICENSE, cutoff, cutoff, cutoff),
        )
        return connection.total_changes - before


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser(
        "ingest-sabdab2",
        help="ingest a locally downloaded, SHA-256-pinned SAbDab2 summary",
    )
    ingest.add_argument("--database", required=True, type=Path)
    ingest.add_argument("--summary-csv", required=True, type=Path)
    ingest.add_argument("--dataset-id", required=True)
    ingest.add_argument("--release", required=True)
    ingest.add_argument("--release-date", required=True)
    ingest.add_argument("--retrieved-at", required=True)
    ingest.add_argument("--source-url", required=True)
    ingest.add_argument("--expected-sha256", required=True)
    ingest.add_argument("--terms-snapshot-uri", required=True)

    link = subparsers.add_parser(
        "link-exact",
        help="link eligible SAbDab2 VHHs to existing graph nodes by exact sequence",
    )
    link.add_argument("--database", required=True, type=Path)
    link.add_argument("--target", required=True)
    link.add_argument("--as-of", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "ingest-sabdab2":
        release = DatasetRelease.sabdab2(
            dataset_id=args.dataset_id,
            release=args.release,
            release_date=args.release_date,
            retrieved_at=args.retrieved_at,
            source_url=args.source_url,
            artifact_sha256=args.expected_sha256,
            terms_snapshot_uri=args.terms_snapshot_uri,
        )
        result = ingest_sabdab2_summary(args.database, args.summary_csv, release)
    else:
        result = {
            "exact_links_written": link_exact_nova_sequences(
                args.database,
                target=args.target,
                as_of=args.as_of,
            )
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
