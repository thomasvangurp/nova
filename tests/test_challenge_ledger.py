"""The ledger must reproduce the host's fingerprint and challenge exactly."""
from nanobody_opt import challenge_ledger as L
from nanobody_opt import rules


def _config():
    return rules.pinned_config(rules.ACTIVATION_EPOCH)


def test_fingerprint_matches_rules():
    config = _config()
    assert L.fingerprint(config) == rules.fingerprint(config)


def test_derive_matches_rules():
    config = _config()
    block_hash = "0x" + "ab" * 32
    for epoch in (26000, 26001, 26002):
        challenge, category_key = L.derive_challenge(config, epoch, block_hash, 361)
        host = rules.challenge(config, epoch, block_hash, 361)
        assert challenge == host
        # category_key is the V bucket the host picked but does not record
        assert category_key in config["germline_filtering"]["v_neighborhood_categories"]


def test_record_and_verify_round_trip(tmp_path):
    config = _config()
    block_hash = "0x" + "cd" * 32
    epoch, epoch_length = 26000, 361
    challenge, _ = L.derive_challenge(config, epoch, block_hash, epoch_length)
    db = str(tmp_path / "ledger.sqlite")
    conn = L.connect(db)
    config_bytes = L.canonical_json(config)
    r1 = L.record_challenge(config_bytes, challenge, L.DERIVATION_VERSION,
                            "live", {"node": "test"}, conn=conn)
    assert r1["inserted"] is True
    # idempotent
    r2 = L.record_challenge(config_bytes, challenge, L.DERIVATION_VERSION,
                            "live", {"node": "test"}, conn=conn)
    assert r2["inserted"] is False
    L.verify_challenge(epoch, challenge, conn=conn)


def test_record_challenge_best_effort_writes_live_row(tmp_path):
    from nanobody_opt import pr100_epoch_continuation as ec
    config = dict(rules.pinned_config(rules.ACTIVATION_EPOCH))
    config["state_root"] = str(tmp_path)
    epoch, block_hash = 26000, "0x" + "cd" * 32
    challenge = rules.challenge(config, epoch, block_hash, 361)
    clock = {"block": epoch * 361, "epoch": epoch}
    ec._record_challenge_best_effort(config, epoch, clock, challenge, dry_run=False)
    conn = L.connect(str(tmp_path / "challenge-ledger.sqlite"))
    row = conn.execute("SELECT epoch, source, derivation_version FROM epoch_challenges").fetchone()
    conn.close()
    assert row is not None, "no ledger row was written"
    assert row[0] == epoch and row[1] == "live" and row[2] == L.DERIVATION_VERSION
