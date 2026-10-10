#!/usr/bin/env python3
"""Volume + kill -9 soak for nanobody_opt.replicate_local (ARCHITECTURE_REDESIGN
new finding 8). SANDBOX ONLY: needs a disposable MySQL with db/schema.sql
applied; it TRUNCATEs every table there and refuses a database whose name does
not contain "test".

What it does, concurrently:
  * WRITER -- the real local write path, mutation_network.update_observation_store
    with NOVA_LOCAL_SQLITE=1, writes ``--tranches`` tranches of 10,000 rows
    (the live race's tranche size) built from real payload templates, ~30%
    with a parent (lineage edge), ~5% with a structure (artifact). After each
    tranche it REWRITES 2,000 random earlier observations (new score / count),
    so rows change while their old version is in flight, and records one exact
    run per tranche the way record_exact_run does (DELETE + INSERT).
  * OTHER BOX -- inserts nodes straight into the remote nodes table, so the
    remote AUTO_INCREMENT moves independently of the local one (the remote is
    also pre-seeded with ``--seed-other`` foreign nodes plus 500 of the soak's
    own sequences under different ids): local and remote ids collide.
  * REPLICATOR -- ``python -m nanobody_opt.replicate_local --loop-interval 1
    --trace`` as a subprocess, SIGKILLed at random --kill-min..--kill-max s intervals and
    restarted, for the whole write phase.
Then the replicator runs unkilled until the backlog is empty, and the remote
is compared with the local store BY IDENTITY: every observation, artifact,
lineage edge (both endpoints by (target, sequence)) and exact run must match
exactly, no remote node identity may be duplicated, and the foreign nodes
must be untouched.

    python tools/replication_soak.py --remote-creds creds.env --workdir /root/scratch/repl/soak
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import MySQLdb  # noqa: F401
except ImportError:  # pragma: no cover
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb  # noqa: F401

AA = "ACDEFGHIKLMNPQRSTVWY"
TARGET = "P20809"
TABLES = ["nodes", "observations", "observation_artifacts", "lineage_edges",
          "exact_runs", "exact_run_observations"]


def read_creds(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def remote_connect(creds: dict[str, str]):
    return MySQLdb.connect(
        host=creds["SN68_MYSQL_HOST"], port=int(creds["SN68_MYSQL_PORT"]),
        user=creds["SN68_MYSQL_USER"], password=creds["SN68_MYSQL_PASSWORD"],
        database=creds["SN68_MYSQL_DB"], charset="utf8mb4", autocommit=False,
    )


def random_sequence(rng: random.Random, length: int = 127) -> str:
    return "".join(rng.choice(AA) for _ in range(length))


def mutate(rng: random.Random, sequence: str, edits: int = 3) -> str:
    chars = list(sequence)
    for _ in range(edits):
        index = rng.randrange(len(chars))
        chars[index] = rng.choice(AA.replace(chars[index], ""))
    return "".join(chars)


def vector(sequence: str) -> bytes:
    return bytes(AA.index(residue) for residue in sequence)


RUNNERS: list["ReplicatorRunner"] = []


class ReplicatorRunner:
    def __init__(self, local: Path, creds_path: Path, log_path: Path) -> None:
        self.local, self.creds_path, self.log_path = local, creds_path, log_path
        self.proc: subprocess.Popen | None = None
        self.kills = 0
        self.kill_times: list[float] = []
        RUNNERS.append(self)

    fault = 0.0

    def start(self) -> None:
        log = open(self.log_path, "a")
        env = dict(os.environ)
        env["REPLICATE_LOCAL_FAULT_KILL_AFTER_COMMIT"] = str(self.fault)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "nanobody_opt.replicate_local", "--local", str(self.local),
             "--remote-creds", str(self.creds_path), "--loop-interval", "1", "--trace"],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
        )

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
            self.kills += 1
            self.kill_times.append(time.time())
            with open(self.log_path, "a") as log:
                log.write(f"SOAK_SIGKILL {time.time():.3f}\n")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote-creds", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--templates", type=Path,
                        help="JSON {observations:[...]} of real payload rows (optional)")
    parser.add_argument("--tranches", type=int, default=3)
    parser.add_argument("--tranche-rows", type=int, default=10_000)
    parser.add_argument("--seed-other", type=int, default=50_000)
    parser.add_argument("--no-kill", action="store_true")
    parser.add_argument("--fault-after-commit", type=float, default=0.0,
                        help="probability the replicator SIGKILLs itself right after a "
                             "remote commit, before the local mark (write phase only)")
    parser.add_argument("--kill-min", type=float, default=3.0)
    parser.add_argument("--kill-max", type=float, default=15.0,
                        help="SIGKILL interval bounds (s); connecting over the tunnel alone takes ~3 s")
    parser.add_argument("--kill-seconds", type=float, default=180.0,
                        help="keep SIGKILLing the replicator at least this long")
    parser.add_argument("--tranche-gap", type=float, default=20.0,
                        help="seconds between tranches (the live race writes one per few minutes)")
    parser.add_argument("--seed", type=int, default=68)
    args = parser.parse_args()

    os.environ["NOVA_LOCAL_SQLITE"] = "1"
    from nanobody_opt import mutation_network

    rng = random.Random(args.seed)
    creds = read_creds(args.remote_creds)
    args.workdir.mkdir(parents=True, exist_ok=True)
    local_path = args.workdir / "soak_local.sqlite"
    for suffix in ("", "-wal", "-shm", ".replication_status.json"):
        Path(f"{local_path}{suffix}").unlink(missing_ok=True)
    log_path = args.workdir / "replicator.log"
    log_path.write_text("")

    remote = remote_connect(creds)
    cur = remote.cursor()
    cur.execute("SELECT DATABASE()")
    database = cur.fetchone()[0]
    if "test" not in database:
        raise SystemExit(f"refusing to truncate non-test database {database!r}")
    for table in TABLES:
        cur.execute(f"TRUNCATE TABLE {table}")
    remote.commit()

    templates = [{"metrics_json": json.dumps({"score": 1.0, "iptm": 0.8, "ptm": 0.7}),
                  "pose_json": None, "provenance_json": json.dumps({"k": "v" * 300}),
                  "source": "live_exact_screen", "score_kind": "live_exact_rank_sum"}]
    if args.templates and args.templates.is_file():
        templates = json.loads(args.templates.read_text())["observations"]

    # ---- seed the remote like another box already wrote there -------------
    soak_sequences = [random_sequence(rng) for _ in range(args.tranches * args.tranche_rows)]
    foreign = [random_sequence(rng) for _ in range(args.seed_other)]
    t0 = time.time()
    cur.executemany(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(%s,%s,%s,%s)",
        [(TARGET, s, len(s), vector(s)) for s in foreign],
    )
    # 500 of OUR sequences already exist remotely, under ids >> the local ones.
    preexisting = rng.sample(soak_sequences, 500)
    cur.executemany(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(%s,%s,%s,%s)",
        [(TARGET, s, len(s), vector(s)) for s in preexisting],
    )
    remote.commit()
    cur.execute("SELECT COUNT(*), COALESCE(SUM(CRC32(CONCAT(id, ':', sequence))), 0) FROM nodes")
    foreign_count, foreign_sum = cur.fetchone()
    print(f"SOAK_SEEDED remote_nodes={foreign_count} in {time.time() - t0:.1f}s", flush=True)

    # ---- concurrent "other box" node writer ----------------------------------
    stop = threading.Event()
    other_inserted = [0]
    other_ids: list[int] = []

    def other_box() -> None:
        conn = remote_connect(creds)
        c = conn.cursor()
        local_rng = random.Random(args.seed + 1)
        while not stop.is_set():
            batch = [random_sequence(local_rng) for _ in range(100)]
            c.executemany(
                "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(%s,%s,%s,%s)",
                [(TARGET, s, len(s), vector(s)) for s in batch],
            )
            conn.commit()
            other_inserted[0] += len(batch)
            stop.wait(0.5)
        conn.close()

    other_thread = threading.Thread(target=other_box, daemon=True)
    other_thread.start()

    # ---- replicator under random SIGKILL -------------------------------------
    # The writer creates the store (db.connect_local applies local_schema.sql).
    with mutation_network._connect(local_path):
        pass
    runner = ReplicatorRunner(local_path, args.remote_creds, log_path)
    runner.fault = args.fault_after_commit
    runner.start()

    def killer() -> None:
        killer_rng = random.Random(args.seed + 2)
        while not stop.is_set():
            if stop.wait(killer_rng.uniform(args.kill_min, args.kill_max)):
                return
            runner.kill()
            runner.start()

    killer_thread = threading.Thread(target=killer, daemon=True)
    if not args.no_kill:
        killer_thread.start()

    # ---- the writer: real update_observation_store ---------------------------
    kill_phase_start = time.time()
    written: list[str] = []
    write_seconds = []
    poison_id = None
    for tranche in range(args.tranches):
        rows = []
        base = tranche * args.tranche_rows
        for i in range(args.tranche_rows):
            template = rng.choice(templates)
            sequence = soak_sequences[base + i]
            row = {
                "observation_id": f"soak-{tranche}-{i}",
                "sequence": sequence, "target": TARGET, "epoch": 90_000 + tranche,
                "source": template["source"], "score_kind": template["score_kind"],
                "score": rng.uniform(0, 300), "observation_count": 1,
                "metrics": json.loads(template["metrics_json"]),
                "pose_stability": json.loads(template["pose_json"]) if template["pose_json"] else None,
                "soak_provenance": template["provenance_json"][:400],
            }
            if rng.random() < 0.30:
                parent = (soak_sequences[rng.randrange(base + i)] if base + i and rng.random() < 0.5
                          else mutate(rng, sequence))
                row["parent_sequence"] = parent
            if rng.random() < 0.05:
                row["structure"] = {"path": f"/root/live/s/{tranche}/{i}.cif", "sample": i}
            rows.append(row)
        if tranche == 0:
            rows[1234]["source"] = "x" * 100  # varchar(64): must quarantine, not wedge
            poison_id = rows[1234]["observation_id"]
        t0 = time.time()
        mutation_network.update_observation_store(local_path, rows)
        write_seconds.append(round(time.time() - t0, 2))
        written.extend(row["observation_id"] for row in rows)
        # In-flight rewrites of earlier rows (new version must win remotely).
        rewrite_ids = set(rng.sample(written, min(2000, len(written)))) - {poison_id}
        lookup = {row["observation_id"]: row for row in rows}
        con = sqlite3.connect(local_path, timeout=60)
        con.execute("BEGIN IMMEDIATE")
        con.executemany(
            "UPDATE observations SET score=?, observation_count=observation_count+1 "
            "WHERE observation_id=?",
            [(rng.uniform(0, 300), oid) for oid in rewrite_ids],
        )
        con.commit()
        # One exact run per tranche, written like record_exact_run.
        members = list(dict.fromkeys(
            [r["observation_id"] for r in rows[:500]] + sorted(rewrite_ids)[:100]))
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            """INSERT INTO exact_runs(target, run_id, epoch, source_path, file_sha256,
                   scorer_protocol, expected_rows, ingested_rows, complete, ingested_at)
               VALUES(?,?,?,?,?,?,?,?,1,?)
               ON CONFLICT(target, run_id) DO UPDATE SET ingested_at=excluded.ingested_at""",
            (TARGET, f"soak-run-{tranche}", 90_000 + tranche, "/x", "sha", "p",
             len(members), len(members), time.time()),
        )
        con.execute("DELETE FROM exact_run_observations WHERE target=? AND run_id=?",
                    (TARGET, f"soak-run-{tranche}"))
        con.executemany(
            "INSERT INTO exact_run_observations(target, run_id, observation_id) VALUES(?,?,?)",
            [(TARGET, f"soak-run-{tranche}", m) for m in members],
        )
        con.commit()
        con.close()
        print(f"SOAK_TRANCHE {tranche} rows={len(rows)} write_s={write_seconds[-1]} "
              f"rewrites={len(rewrite_ids)} kills_so_far={runner.kills}", flush=True)
        time.sleep(args.tranche_gap)

    # Rewrite membership of run 0 after the fact (DELETE+INSERT must propagate).
    con = sqlite3.connect(local_path, timeout=60)
    con.execute("BEGIN IMMEDIATE")
    con.execute("DELETE FROM exact_run_observations WHERE target=? AND run_id='soak-run-0'", (TARGET,))
    con.executemany("INSERT INTO exact_run_observations(target, run_id, observation_id) VALUES(?,?,?)",
                    [(TARGET, "soak-run-0", f"soak-0-{i}") for i in range(10)])
    con.execute("UPDATE exact_runs SET expected_rows=10, ingested_rows=10 WHERE run_id='soak-run-0'")
    con.commit()
    con.close()

    if not args.no_kill:
        time.sleep(max(0.0, args.kill_seconds - (time.time() - kill_phase_start)))
    stop.set()
    killer_thread.join(timeout=10)
    # Drain phase: no faults, no kills.
    runner.fault = 0.0
    kills_before = runner.kills
    runner.kill()  # end of the kill phase, not a test kill
    runner.kills = kills_before
    other_thread.join(timeout=10)
    runner.start()
    drain_start = time.time()
    while True:
        con = sqlite3.connect(local_path, timeout=60)
        dirty = sum(con.execute(f"SELECT COUNT(*) FROM {t} WHERE replicated_at IS NULL").fetchone()[0]
                    for t in ("observations", "observation_artifacts", "lineage_edges",
                              "exact_runs", "exact_run_observations"))
        unmapped = con.execute("SELECT COUNT(*) FROM nodes n WHERE NOT EXISTS (SELECT 1 FROM "
                               "replication_node_map m WHERE m.local_id=n.id)").fetchone()[0]
        con.close()
        if dirty == 0 and unmapped == 0:
            break
        if time.time() - drain_start > 1800:
            print(f"SOAK_FAIL drain timeout dirty={dirty} unmapped={unmapped}")
            return 1
        time.sleep(2)
    drain_seconds = round(time.time() - drain_start, 1)
    runner.stop()

    # ---- verification ---------------------------------------------------------
    failures = []
    local = sqlite3.connect(local_path)
    remote.commit()  # fresh snapshot
    obs_sql = """SELECT o.observation_id, n.target, n.sequence, o.epoch, o.source, o.score_kind,
                        o.score, o.observation_count, o.metrics_json, o.pose_json, o.provenance_json
                 FROM observations o JOIN nodes n ON n.id=o.node_id"""
    local_obs = {r[0]: tuple(r[1:]) for r in local.execute(obs_sql)}
    cur.execute(obs_sql)
    remote_obs = {r[0]: tuple(r[1:]) for r in cur.fetchall()}
    expected_obs = {k: v for k, v in local_obs.items() if k != poison_id}
    if remote_obs != expected_obs:
        missing = set(expected_obs) - set(remote_obs)
        extra = set(remote_obs) - set(expected_obs)
        differ = [k for k in set(expected_obs) & set(remote_obs) if expected_obs[k] != remote_obs[k]]
        failures.append(f"observations mismatch missing={len(missing)} extra={len(extra)} "
                        f"differ={len(differ)} e.g. {(sorted(missing) + sorted(extra) + differ)[:3]}")
    poison_state = local.execute("SELECT replicated_at FROM observations WHERE observation_id=?",
                                 (poison_id,)).fetchone()[0]
    if poison_state != -1:
        failures.append(f"poison row replicated_at={poison_state}, expected -1")

    lin_sql = """SELECT p.target, p.sequence, c.sequence, e.epoch, e.source, e.proposal_type,
                        e.mutations_json, e.metadata_json
                 FROM lineage_edges e JOIN nodes p ON p.id=e.parent_node_id
                 JOIN nodes c ON c.id=e.child_node_id"""
    local_lin = set(local.execute(lin_sql))
    cur.execute(lin_sql)
    remote_lin = set(tuple(r) for r in cur.fetchall())
    if local_lin != remote_lin:
        failures.append(f"lineage mismatch local={len(local_lin)} remote={len(remote_lin)} "
                        f"sym_diff={len(local_lin ^ remote_lin)}")
    art_sql = "SELECT observation_id, confidence_path, writer_path, structure_json, sample_selection_json FROM observation_artifacts"
    local_art = set(local.execute(art_sql))
    cur.execute(art_sql)
    remote_art = set(tuple(r) for r in cur.fetchall())
    if local_art != remote_art:
        failures.append(f"artifacts mismatch local={len(local_art)} remote={len(remote_art)}")
    run_sql = "SELECT target, run_id, epoch, expected_rows, ingested_rows, complete, ingested_at FROM exact_runs"
    member_sql = "SELECT target, run_id, observation_id FROM exact_run_observations"
    cur.execute(run_sql)
    remote_runs = set(tuple(r) for r in cur.fetchall())
    cur.execute(member_sql)
    remote_members = set(tuple(r) for r in cur.fetchall())
    if set(local.execute(run_sql)) != remote_runs:
        failures.append("exact_runs mismatch")
    if set(local.execute(member_sql)) != remote_members:
        failures.append(f"exact_run_observations mismatch local={local.execute('SELECT COUNT(*) FROM exact_run_observations').fetchone()[0]} remote={len(remote_members)}")
    # Barrier: every remote complete run's members exist remotely.
    cur.execute("""SELECT COUNT(*) FROM exact_run_observations e
                   JOIN exact_runs r ON r.target=e.target AND r.run_id=e.run_id AND r.complete=1
                   LEFT JOIN observations o ON o.observation_id=e.observation_id
                   WHERE o.observation_id IS NULL""")
    if cur.fetchone()[0]:
        failures.append("complete exact run with members missing on remote")
    # Node identity: no duplicates, every map entry names the same identity.
    cur.execute("SELECT COUNT(*) - COUNT(DISTINCT target, sequence) FROM nodes")
    if cur.fetchone()[0]:
        failures.append("duplicate remote node identities")
    mapping = local.execute("""SELECT m.local_id, m.remote_id, n.target, n.sequence
                               FROM replication_node_map m JOIN nodes n ON n.id=m.local_id""").fetchall()
    remote_nodes = {}
    ids = [r[1] for r in mapping]
    for start in range(0, len(ids), 5000):
        chunk = ids[start:start + 5000]
        cur.execute("SELECT id, target, sequence FROM nodes WHERE id IN (" + ",".join(["%s"] * len(chunk)) + ")", chunk)
        remote_nodes.update({r[0]: (r[1], r[2]) for r in cur.fetchall()})
    bad_map = sum(1 for local_id, remote_id, t, s in mapping if remote_nodes.get(remote_id) != (t, s))
    same_id = sum(1 for local_id, remote_id, *_ in mapping if local_id == remote_id)
    if bad_map:
        failures.append(f"{bad_map} map entries point at a different remote identity")
    local_nodes = local.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    if len(mapping) != local_nodes:
        failures.append(f"mapped {len(mapping)} of {local_nodes} local nodes")
    # Foreign rows untouched: the seeded node set is unchanged.
    cur.execute("SELECT COUNT(*), COALESCE(SUM(CRC32(CONCAT(id, ':', sequence))), 0) FROM nodes "
                "WHERE id <= %s", (foreign_count,))
    if tuple(cur.fetchone()) != (foreign_count, foreign_sum):
        failures.append("pre-existing remote nodes changed")
    pre_ok = 0
    for seq in preexisting[:50]:
        cur.execute("SELECT id FROM nodes WHERE target=%s AND sequence=%s", (TARGET, seq))
        found = cur.fetchall()
        pre_ok += len(found) == 1 and found[0][0] <= foreign_count
    if pre_ok != 50:
        failures.append("pre-existing shared sequences were duplicated or re-id'd")

    # ---- crash-window evidence from the trace -----------------------------------
    events = [line.split() for line in log_path.read_text().splitlines()
              if line.startswith(("REPLICATE_TRACE", "SOAK_SIGKILL"))]
    in_window = 0
    pending = False
    for event in events:
        if event[0] == "SOAK_SIGKILL":
            in_window += pending
            pending = False
        elif event[2] == "remote_committed":
            pending = True
        elif event[2] == "marked":
            pending = False
    self_kills = sum(1 for line in log_path.read_text().splitlines()
                     if line.startswith("REPLICATE_FAULT_SELF_KILL"))
    superseded = sum(1 for line in log_path.read_text().splitlines() if '"superseded"' in line)
    report = {
        "tranches": args.tranches, "rows_per_tranche": args.tranche_rows,
        "local_observations": len(local_obs), "remote_observations": len(remote_obs),
        "lineage_edges": len(local_lin), "artifacts": len(local_art),
        "local_nodes": local_nodes, "remote_nodes": len(remote_nodes),
        "map_entries_with_same_raw_id": same_id,
        "other_box_nodes_inserted_concurrently": other_inserted[0],
        "sigkills": runner.kills,
        "sigkills_between_remote_commit_and_local_mark": in_window,
        "self_sigkills_after_remote_commit_before_local_mark": self_kills,
        "passes_reporting_superseded_rows": superseded,
        "writer_seconds_per_tranche": write_seconds,
        "final_drain_seconds": drain_seconds,
        "failures": failures,
    }
    print("SOAK_REPORT " + json.dumps(report, indent=1), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        for runner_proc in RUNNERS:
            runner_proc.stop()
