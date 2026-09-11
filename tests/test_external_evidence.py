import csv
import sqlite3

import pytest

from nanobody_opt.external_evidence import (
    DatasetPolicyError,
    DatasetRelease,
    ExternalEvidenceError,
    eligible_sabdab2_vhh,
    ingest_sabdab2_summary,
    link_exact_nova_sequences,
    sha256_file,
)
from nanobody_opt.mutation_network import update_observation_store


FIELDS = [
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
]
SEQUENCE = "A" * 100


def _row(**overrides):
    row = {
        "INSTANCE": "pdb_00001abc-A",
        "PDB": "pdb_00001abc",
        "SABDAB_ID": "sabdab2_example",
        "Hchain": "A",
        "VH": SEQUENCE,
        "CDR-H1": "AAAA",
        "CDR-H2": "AAAAA",
        "CDR-H3": "AAAAAA",
        "model": "0",
        "antigen_chain": "B",
        "antigen_type": "PROTEIN",
        "antigen_name": "Example protein",
        "date": "2026/04/01",
        "SABDABdepo_date": "20260501",
        "SABDABupdate_date": "20260502",
        "heavy_species": "lama glama",
        "antigen_species": "homo sapiens",
        "agtaxid": "9606",
        "method": "XRAY",
        "resolution": "1.8",
        "type": "SD-H",
    }
    row.update(overrides)
    return row


def _summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _release(path, **overrides):
    values = {
        "dataset_id": "sabdab2:ui-2.1.0:2026-06-01",
        "release": "ui-2.1.0-api-2.0.10",
        "release_date": "2026-06-01",
        "retrieved_at": "2026-06-02T01:02:03Z",
        "source_url": (
            "https://sabdab.opig.stats.ox.ac.uk/api/download/all-single-domain-summary"
        ),
        "artifact_sha256": sha256_file(path),
        "terms_snapshot_uri": "https://sabdab.opig.stats.ox.ac.uk/about",
    }
    values.update(overrides)
    return DatasetRelease.sabdab2(**values)


def _graph(path):
    update_observation_store(
        path,
        [
            {
                "observation_id": "epoch1:sequence",
                "sequence": SEQUENCE,
                "score": 70,
                "epoch": 1,
                "target": "P20809",
            }
        ],
    )


def test_ingest_is_idempotent_and_does_not_create_scored_nodes(tmp_path):
    database = tmp_path / "graph.sqlite"
    _graph(database)
    summary_path = tmp_path / "sabdab.csv"
    _summary(
        summary_path,
        [
            _row(),
            _row(
                INSTANCE="pdb_00002def-C",
                PDB="pdb_00002def",
                VH="C" * 100,
                antigen_chain="NA",
                antigen_type="NA",
                antigen_name="NA",
            ),
            _row(INSTANCE="pdb_00003ghi-S", type="VNAR"),
            _row(INSTANCE="pdb_00004jkl-X", VH="X" + "A" * 99),
        ],
    )
    release = _release(summary_path)

    result = ingest_sabdab2_summary(database, summary_path, release)
    again = ingest_sabdab2_summary(database, summary_path, release)

    assert result == again
    assert result["rows_read"] == 4
    assert result["sdh_records"] == 3
    assert result["usable_vhh_records"] == 2
    assert result["quarantined_vhh_records"] == 1
    assert result["non_sdh_rows_skipped"] == 1
    assert result["bound_complex_records"] == 2
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM external_records").fetchone()[0]
            == 3
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM external_vhh").fetchone()[0] == 2
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM external_interactions").fetchone()[
                0
            ]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM external_complexes").fetchone()[0]
            == 2
        )
        quarantined = connection.execute(
            "SELECT ingest_status, quality_flags_json FROM external_records "
            "WHERE source_accession='pdb_00004jkl-X'"
        ).fetchone()
        assert quarantined[0] == "quarantined"
        assert "noncanonical_residues:X" in quarantined[1]
        interaction = connection.execute(
            "SELECT raw_value, value, unit, assay FROM external_interactions"
        ).fetchone()
        assert interaction == (None, None, None, None)
        model = connection.execute(
            "SELECT assembly_id, model_id FROM external_complexes "
            "WHERE antigen_chain_ids='B'"
        ).fetchone()
        assert model == (None, "0")
        antigen = connection.execute(
            "SELECT uniprot_accession, raw_sequence FROM external_antigens"
        ).fetchone()
        assert antigen == (None, None)


def test_new_release_preserves_records_without_duplicate_entities(tmp_path):
    database = tmp_path / "graph.sqlite"
    _graph(database)
    summary_path = tmp_path / "sabdab.csv"
    _summary(summary_path, [_row()])
    ingest_sabdab2_summary(database, summary_path, _release(summary_path))
    ingest_sabdab2_summary(
        database,
        summary_path,
        _release(
            summary_path,
            dataset_id="sabdab2:ui-2.1.0:2026-06-02",
            release="ui-2.1.0-api-2.0.10-r2",
            release_date="2026-06-02",
            retrieved_at="2026-06-03T01:02:03Z",
        ),
    )

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM external_datasets").fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM external_records").fetchone()[0]
            == 2
        )
        for table in (
            "external_vhh",
            "external_antigens",
            "external_interactions",
            "external_complexes",
        ):
            assert (
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
            )
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM external_record_links
            WHERE entity_kind='complex'
            """
            ).fetchone()[0]
            == 2
        )


def test_artifact_and_license_policy_are_fail_closed(tmp_path):
    database = tmp_path / "graph.sqlite"
    _graph(database)
    summary_path = tmp_path / "sabdab.csv"
    _summary(summary_path, [_row()])

    bad_hash = _release(summary_path, artifact_sha256="0" * 64)
    with pytest.raises(ExternalEvidenceError, match="SHA-256 mismatch"):
        ingest_sabdab2_summary(database, summary_path, bad_hash)

    values = _release(summary_path).__dict__ | {
        "license_id": "CC-BY-NC-4.0",
        "commercial_use_allowed": False,
    }
    with pytest.raises(DatasetPolicyError, match="license must be CC-BY-4.0"):
        ingest_sabdab2_summary(database, summary_path, DatasetRelease(**values))
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "external_datasets" not in tables


def test_declared_release_cannot_precede_source_update(tmp_path):
    database = tmp_path / "graph.sqlite"
    _graph(database)
    summary_path = tmp_path / "sabdab.csv"
    _summary(summary_path, [_row(SABDABupdate_date="20260602")])

    with pytest.raises(ExternalEvidenceError, match="postdates declared release"):
        ingest_sabdab2_summary(database, summary_path, _release(summary_path))

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "external_datasets" not in tables


def test_temporal_gate_controls_evidence_and_exact_nova_links(tmp_path):
    database = tmp_path / "graph.sqlite"
    _graph(database)
    summary_path = tmp_path / "sabdab.csv"
    _summary(summary_path, [_row()])
    ingest_sabdab2_summary(database, summary_path, _release(summary_path))

    assert eligible_sabdab2_vhh(database, as_of="2026-05-31") == []
    assert link_exact_nova_sequences(database, target="P20809", as_of="2026-05-31") == 0

    eligible = eligible_sabdab2_vhh(database, as_of="2026-06-01")
    assert len(eligible) == 1
    assert eligible[0]["variable_sequence"] == SEQUENCE
    assert eligible[0]["license_id"] == "CC-BY-4.0"
    assert link_exact_nova_sequences(database, target="P20809", as_of="2026-06-01") == 1
    assert link_exact_nova_sequences(database, target="P20809", as_of="2026-06-01") == 0
    with sqlite3.connect(database) as connection:
        link = connection.execute(
            """
            SELECT alignment_method, sequence_identity, is_exact, cutoff_snapshot
            FROM external_nova_links
            """
        ).fetchone()
        assert link == ("exact_sha256", 1.0, 1, "2026-06-01")
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM eligible_external_nova_links"
            ).fetchone()[0]
            == 1
        )
        connection.execute("UPDATE external_datasets SET tdm_ai_allowed=0")
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM eligible_external_nova_links"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        )
