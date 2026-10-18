"""Recover completed-epoch snapshots from archival chain data and public payloads.

No wallet or authenticated API is used. Library output is suppressed; only
public commitment references, protein sequences, and safe summaries are stored.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt.commit_submission import write_receipt

#: The validator reads a commitment only while start_block < commit_block <
#: end - BLACKOUT_BLOCKS. Canonical definition lives in nanobody_opt.fleet
#: (BLACKOUT_BLOCKS = 16); reconciliation must not count a commit the
#: validator never read.
BLACKOUT_BLOCKS = 16


def reconciled_documents(epoch: int, target: str, recovered: list[dict],
                         previous: dict, legacy: dict) -> tuple[dict, dict]:
    """Retain checkpoints only from exact, previously immutable receipts."""
    known = {(x.get("uid"), x.get("hotkey"), x.get("sequence")): x
             for x in legacy.get("submissions", [])}
    entries = []
    for proof in sorted(recovered, key=lambda x: x["uid"]):
        old = known.get((proof["uid"], proof["hotkey"], proof["sequence"]), {})
        digest = hashlib.sha256(proof["sequence"].encode()).hexdigest()
        if proof.get("sequence_sha256") != digest:
            raise ValueError("recovered sequence digest does not match")
        entry = {**old, **proof}
        if not old:
            entry["prediction_at_commit"] = None
            entry["prediction_provenance"] = "unrecoverable_from_mutable_snapshot"
        entries.append(entry)
    uids = [x["uid"] for x in entries]
    reported = sorted(set(legacy.get("legacy_reported_uids", legacy.get("uids", []))))
    ledger = {"schema_version": 2, "epoch": epoch, "uids": uids,
              "submissions": entries, "legacy_reported_uids": reported,
              "unverified_legacy_uids": sorted(set(reported) - set(uids)),
              "reconciliation_source": "archive_chain_and_decrypted_payload"}
    ledger["validator_readable_uids"] = [x["uid"] for x in entries
                                        if x.get("validator_readable_at_epoch_end", True)]
    snapshot = {"schema_version": previous.get("schema_version", 1),
                "artifact_kind": "nanobody-fleet", "target": target, "epoch": epoch,
                "status": "committed", "publication_stage": "committed",
                "publication_source": "archive_chain_and_decrypted_payload",
                "submission_count": len(entries), "submissions": entries,
                "validator_readable_submission_count": len(ledger["validator_readable_uids"]),
                "unverified_legacy_uids": ledger["unverified_legacy_uids"]}
    return ledger, snapshot


def preserve_and_write(path: Path, payload: dict, backups: Path, label: str) -> None:
    if path.exists():
        raw = path.read_bytes()
        if json.loads(raw) == payload:
            return
        backups.mkdir(parents=True, exist_ok=True)
        destination = backups / f"{label}_{hashlib.sha256(raw).hexdigest()[:16]}.json"
        if not destination.exists():
            destination.write_bytes(raw)
    write_receipt(path, payload)


def recover(subtensor, timelock, *, epoch: int, epoch_length: int, coldkey: str,
            github_prefix: str, extra_receipts: list[dict] = ()) -> list[dict]:
    from bittensor.core.chain_data.utils import decode_metadata

    start = epoch * epoch_length
    end = start + epoch_length
    if subtensor.get_current_block() < end:
        raise ValueError("cannot reconcile an unfinished epoch")
    graph = subtensor.metagraph(68, block=end - 1, lite=True)
    pending, records = {}, []
    eligible = {uid: str(graph.hotkeys[uid]) for uid, owner in enumerate(graph.coldkeys)
                if owner == coldkey}
    subjects = [(uid, hotkey, [end - 1], None) for uid, hotkey in eligible.items()]
    for proof in extra_receipts:
        uid, hotkey = proof.get("uid"), proof.get("hotkey")
        if eligible.get(uid) == hotkey:
            continue
        observed = proof.get("commit_block") if proof.get("commitment_content") else proof.get("observed_block")
        if isinstance(uid, int) and isinstance(hotkey, str) and isinstance(observed, int):
            scheduled = proof.get("commit_block", observed)
            floor = max(start + 1, observed - 64,
                        min(observed - 16, scheduled if isinstance(scheduled, int) else observed))
            probes = [observed] if proof.get("commitment_content") else range(observed, floor - 1, -1)
            subjects.append((uid, hotkey, [b for b in probes if start < b < end],
                             proof.get("sequence")))
    expected_sequences = {}
    for uid, hotkey, probes, expected in subjects:
        metadata = None
        for probe in probes:
            found = subtensor.get_commitment_metadata(68, hotkey, block=probe)
            if isinstance(found, dict) and start < int(found.get("block", 0)) < end - BLACKOUT_BLOCKS:
                metadata = found
                break
        if not isinstance(metadata, dict):
            continue
        block = int(metadata["block"])
        if not start < block < end - BLACKOUT_BLOCKS:
            continue
        readable = eligible.get(uid) == hotkey
        if not readable:
            at_inclusion = subtensor.metagraph(68, block=block, lite=True)
            if str(at_inclusion.hotkeys[uid]) != hotkey or at_inclusion.coldkeys[uid] != coldkey:
                continue
        content = decode_metadata(metadata)
        if not isinstance(content, str) or not re.fullmatch(
                re.escape(github_prefix) + r"/[0-9a-f]{20}\.txt", content):
            raise ValueError("unexpected public submission reference")
        with urlopen("https://raw.githubusercontent.com/" + content, timeout=30) as response:
            body = response.read(100000)
        if hashlib.sha256(body).hexdigest()[:20] != content.rsplit("/", 1)[1][:-4]:
            raise ValueError("ciphertext content hash does not match")
        encrypted = ast.literal_eval(body.decode())
        if not (isinstance(encrypted, tuple) and len(encrypted) == 2
                and isinstance(encrypted[0], int) and isinstance(encrypted[1], bytes)):
            raise ValueError("unexpected ciphertext format")
        pending[uid] = encrypted
        if expected:
            expected_sequences[uid] = expected
        records.append({"uid": uid, "hotkey": hotkey, "commit_block": block,
                        "commitment_content": content,
                        "validator_readable_at_epoch_end": readable})
    plaintext = timelock.decrypt_dict(pending)
    for row in records:
        message = plaintext.get(row["uid"])
        if not isinstance(message, str) or not re.fullmatch(r"~\|[ACDEFGHIKLMNPQRSTVWY]{90,150}", message):
            raise ValueError("cannot recover exact valid submission payload")
        sequence = message[2:]
        if row["uid"] in expected_sequences and sequence != expected_sequences[row["uid"]]:
            raise ValueError("historical receipt does not match decrypted payload")
        row.update(sequence=sequence, sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
    if not records:
        raise ValueError("no valid commitments recovered")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--epoch-length", type=int, default=361)
    parser.add_argument("--coldkey", required=True)
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument("--github-prefix", default="thomasvangurp/nova-sn68-submissions/main/data/results")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.runtime_root
    ledger_path = root / "live" / f"epoch{args.epoch}" / "committed_uids.json"
    history = root / "artifacts/nanobody_history" / f"epoch{args.epoch}.json"
    backups = root / "artifacts/commitment_reconciliation" / f"epoch{args.epoch}"
    try:
        previous = json.loads(history.read_text()) if history.exists() else {}
        legacy = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        receipt_lookup = {(r.get("uid"), r.get("hotkey")): r
                          for r in legacy.get("submissions", [])}
        if backups.exists():
            for path in backups.glob("legacy_ledger_*.json"):
                old = json.loads(path.read_text())
                for row in old.get("submissions", []):
                    receipt_lookup.setdefault((row.get("uid"), row.get("hotkey")), row)
        legacy = {**legacy, "submissions": list(receipt_lookup.values())}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import bittensor as bt
            sys.path.insert(0, str(root / "utils"))
            from btdr import QuicknetBittensorDrandTimelock
            subtensor = bt.Subtensor(network="wss://archive.chain.opentensor.ai:443", retry_forever=False)
            records = recover(subtensor, QuicknetBittensorDrandTimelock(), epoch=args.epoch,
                              epoch_length=args.epoch_length, coldkey=args.coldkey,
                              github_prefix=args.github_prefix, extra_receipts=legacy["submissions"])
        ledger, snapshot = reconciled_documents(args.epoch, args.target, records, previous, legacy)
        if args.apply:
            preserve_and_write(ledger_path, ledger, backups, "legacy_ledger")
            preserve_and_write(history, snapshot, backups, "legacy_history")
        write_receipt(backups / "recovered_receipts.json", ledger)
        print(json.dumps({"epoch": args.epoch, "applied": args.apply,
                          "verified_uids": ledger["uids"],
                          "validator_readable_uids": ledger["validator_readable_uids"],
                          "unverified_legacy_uids": ledger["unverified_legacy_uids"]}))
        return 0
    except Exception as error:
        print(json.dumps({"epoch": args.epoch, "exception": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
