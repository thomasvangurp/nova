-- Local SQLite primary store for master's own write path (ARCHITECTURE_REDESIGN
-- new finding 8). Schema mirrors the production MySQL schema (db/schema.sql in
-- the clean-worktree export, captured via SHOW CREATE TABLE 2026-10-10) for the
-- tables master's own race writes to. Cross-box-shared tables/rows (anything
-- written by the dashboard box, e.g. public_multipass_rivals/multipass_sum10_pass)
-- are NOT mirrored here -- those reads continue to go to Hetzner MySQL directly,
-- see nanobody_opt/db.py's connect() (unchanged) vs connect_local() (new).
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=OFF;

CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target TEXT NOT NULL,
  sequence TEXT NOT NULL,
  length INTEGER NOT NULL,
  residue_vector BLOB NOT NULL,
  UNIQUE (target, sequence)
);

CREATE TABLE IF NOT EXISTS observations (
  observation_id TEXT PRIMARY KEY,
  node_id INTEGER NOT NULL,
  epoch INTEGER,
  source TEXT NOT NULL,
  score_kind TEXT NOT NULL,
  score REAL NOT NULL,
  observation_count INTEGER NOT NULL DEFAULT 1,
  metrics_json TEXT NOT NULL,
  pose_json TEXT,
  provenance_json TEXT NOT NULL,
  replicated_at REAL,
  repl_token INTEGER NOT NULL DEFAULT (random())
);
CREATE INDEX IF NOT EXISTS idx_observations_node ON observations(node_id);
CREATE INDEX IF NOT EXISTS idx_observations_node_epoch ON observations(node_id, epoch);
CREATE INDEX IF NOT EXISTS idx_observations_node_oc ON observations(node_id, observation_count);
CREATE INDEX IF NOT EXISTS idx_obs_source_kind ON observations(source, score_kind);
-- Replication cursor: unreplicated rows are found with
-- "WHERE replicated_at IS NULL", not a timestamp scan, so this index is the
-- one that matters for the replicator's hot path.
CREATE INDEX IF NOT EXISTS idx_observations_unreplicated ON observations(replicated_at) WHERE replicated_at IS NULL;

CREATE TABLE IF NOT EXISTS observation_artifacts (
  observation_id TEXT PRIMARY KEY,
  confidence_path TEXT,
  writer_path TEXT,
  structure_json TEXT NOT NULL,
  sample_selection_json TEXT,
  replicated_at REAL,
  repl_token INTEGER NOT NULL DEFAULT (random())
);
CREATE INDEX IF NOT EXISTS idx_artifacts_unreplicated ON observation_artifacts(replicated_at) WHERE replicated_at IS NULL;

CREATE TABLE IF NOT EXISTS lineage_edges (
  parent_node_id INTEGER NOT NULL,
  child_node_id INTEGER NOT NULL,
  epoch INTEGER NOT NULL DEFAULT -1,
  source TEXT NOT NULL,
  proposal_type TEXT NOT NULL DEFAULT 'unknown',
  mutations_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  replicated_at REAL,
  repl_token INTEGER NOT NULL DEFAULT (random()),
  PRIMARY KEY (parent_node_id, child_node_id, epoch, source)
);
CREATE INDEX IF NOT EXISTS idx_lineage_edges_child ON lineage_edges(child_node_id, epoch);
CREATE INDEX IF NOT EXISTS idx_lineage_unreplicated ON lineage_edges(replicated_at) WHERE replicated_at IS NULL;

CREATE TABLE IF NOT EXISTS exact_runs (
  target TEXT NOT NULL,
  run_id TEXT NOT NULL,
  epoch INTEGER,
  source_path TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  scorer_protocol TEXT,
  expected_rows INTEGER NOT NULL,
  ingested_rows INTEGER NOT NULL,
  complete INTEGER NOT NULL,
  ingested_at REAL NOT NULL,
  replicated_at REAL,
  repl_token INTEGER NOT NULL DEFAULT (random()),
  PRIMARY KEY (target, run_id)
);
CREATE INDEX IF NOT EXISTS idx_exact_runs_unreplicated ON exact_runs(replicated_at) WHERE replicated_at IS NULL;

CREATE TABLE IF NOT EXISTS exact_run_observations (
  target TEXT NOT NULL,
  run_id TEXT NOT NULL,
  observation_id TEXT NOT NULL,
  replicated_at REAL,
  repl_token INTEGER NOT NULL DEFAULT (random()),
  PRIMARY KEY (target, run_id, observation_id)
);
CREATE INDEX IF NOT EXISTS idx_exact_run_obs_observation ON exact_run_observations(observation_id);
CREATE INDEX IF NOT EXISTS idx_exact_run_obs_unreplicated ON exact_run_observations(replicated_at) WHERE replicated_at IS NULL;

CREATE TABLE IF NOT EXISTS mutation_effects (
  target TEXT NOT NULL,
  position INTEGER NOT NULL,
  from_residue TEXT NOT NULL,
  to_residue TEXT NOT NULL,
  region TEXT NOT NULL,
  contact_class TEXT NOT NULL,
  support INTEGER NOT NULL,
  median_score_delta REAL NOT NULL,
  uncertainty REAL NOT NULL,
  graph_version TEXT NOT NULL,
  replicated_at REAL,
  PRIMARY KEY (target, position, from_residue, to_residue, region, contact_class, graph_version)
);

CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  replicated_at REAL
);

CREATE TABLE IF NOT EXISTS edges (
  source_node_id INTEGER NOT NULL,
  target_node_id INTEGER NOT NULL,
  distance INTEGER NOT NULL,
  mutations_json TEXT NOT NULL,
  context_json TEXT NOT NULL,
  replicated_at REAL,
  PRIMARY KEY (source_node_id, target_node_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);

CREATE TABLE IF NOT EXISTS proposal_scores (
  epoch INTEGER NOT NULL,
  node_id INTEGER NOT NULL,
  graph_version TEXT NOT NULL,
  prospective_score REAL NOT NULL,
  prospective_rank INTEGER NOT NULL,
  features_json TEXT NOT NULL,
  replicated_at REAL,
  PRIMARY KEY (epoch, node_id, graph_version)
);


-- ---------------------------------------------------------------------------
-- Replication bookkeeping (nanobody_opt/replicate_local.py; ARCHITECTURE_
-- REDESIGN new finding 8). The writer never has to touch any of this.
--
-- replicated_at  NULL  = dirty, not yet confirmed on Hetzner;
--                > 0   = unix time the replicator confirmed the remote commit;
--                -1    = quarantined: Hetzner refused this exact row version
--                        (data error); see replication_quarantine. Any later
--                        change to the row makes it dirty again.
-- repl_token     random 64-bit version stamp. A fresh one is drawn on every
--                INSERT (column default, which also covers INSERT OR REPLACE
--                and DELETE+INSERT) and on every UPDATE that changes a data
--                column (triggers below). The replicator marks a row
--                replicated only "WHERE rowid=? AND repl_token=<token it
--                sent>", so a row the producer rewrites while its old version
--                is in flight to Hetzner stays dirty and is re-sent; the
--                replicator can never mark a version it did not send.
--
-- The triggers fire only when a data column actually changes (an idempotent
-- re-upsert of identical values is not re-sent). They update only
-- replicated_at/repl_token, which are not in their own UPDATE OF lists, so
-- they cannot recurse, and the replicator's own mark UPDATE never fires them.
-- ---------------------------------------------------------------------------

CREATE TRIGGER IF NOT EXISTS observations_repl_dirty
AFTER UPDATE OF node_id, epoch, source, score_kind, score, observation_count, metrics_json, pose_json, provenance_json, observation_id ON observations
WHEN OLD.node_id IS NOT NEW.node_id OR OLD.epoch IS NOT NEW.epoch OR OLD.source IS NOT NEW.source OR OLD.score_kind IS NOT NEW.score_kind OR OLD.score IS NOT NEW.score OR OLD.observation_count IS NOT NEW.observation_count OR OLD.metrics_json IS NOT NEW.metrics_json OR OLD.pose_json IS NOT NEW.pose_json OR OLD.provenance_json IS NOT NEW.provenance_json OR OLD.observation_id IS NOT NEW.observation_id
BEGIN
  UPDATE observations SET replicated_at=NULL, repl_token=random() WHERE rowid=NEW.rowid;
END;

CREATE TRIGGER IF NOT EXISTS observation_artifacts_repl_dirty
AFTER UPDATE OF observation_id, confidence_path, writer_path, structure_json, sample_selection_json ON observation_artifacts
WHEN OLD.observation_id IS NOT NEW.observation_id OR OLD.confidence_path IS NOT NEW.confidence_path OR OLD.writer_path IS NOT NEW.writer_path OR OLD.structure_json IS NOT NEW.structure_json OR OLD.sample_selection_json IS NOT NEW.sample_selection_json
BEGIN
  UPDATE observation_artifacts SET replicated_at=NULL, repl_token=random() WHERE rowid=NEW.rowid;
END;

CREATE TRIGGER IF NOT EXISTS lineage_edges_repl_dirty
AFTER UPDATE OF parent_node_id, child_node_id, epoch, source, proposal_type, mutations_json, metadata_json ON lineage_edges
WHEN OLD.parent_node_id IS NOT NEW.parent_node_id OR OLD.child_node_id IS NOT NEW.child_node_id OR OLD.epoch IS NOT NEW.epoch OR OLD.source IS NOT NEW.source OR OLD.proposal_type IS NOT NEW.proposal_type OR OLD.mutations_json IS NOT NEW.mutations_json OR OLD.metadata_json IS NOT NEW.metadata_json
BEGIN
  UPDATE lineage_edges SET replicated_at=NULL, repl_token=random() WHERE rowid=NEW.rowid;
END;

CREATE TRIGGER IF NOT EXISTS exact_runs_repl_dirty
AFTER UPDATE OF target, run_id, epoch, source_path, file_sha256, scorer_protocol, expected_rows, ingested_rows, complete, ingested_at ON exact_runs
WHEN OLD.target IS NOT NEW.target OR OLD.run_id IS NOT NEW.run_id OR OLD.epoch IS NOT NEW.epoch OR OLD.source_path IS NOT NEW.source_path OR OLD.file_sha256 IS NOT NEW.file_sha256 OR OLD.scorer_protocol IS NOT NEW.scorer_protocol OR OLD.expected_rows IS NOT NEW.expected_rows OR OLD.ingested_rows IS NOT NEW.ingested_rows OR OLD.complete IS NOT NEW.complete OR OLD.ingested_at IS NOT NEW.ingested_at
BEGIN
  UPDATE exact_runs SET replicated_at=NULL, repl_token=random() WHERE rowid=NEW.rowid;
END;

CREATE TRIGGER IF NOT EXISTS exact_run_observations_repl_dirty
AFTER UPDATE OF target, run_id, observation_id ON exact_run_observations
WHEN OLD.target IS NOT NEW.target OR OLD.run_id IS NOT NEW.run_id OR OLD.observation_id IS NOT NEW.observation_id
BEGIN
  UPDATE exact_run_observations SET replicated_at=NULL, repl_token=random() WHERE rowid=NEW.rowid;
END;


-- nodes are replicated by IDENTITY, never by id. nodes.id is a surrogate
-- AUTOINCREMENT key assigned independently here and on Hetzner (and Hetzner's
-- counter is also advanced by every other box), so a raw local id can name a
-- different (target, sequence) remotely. The replicator resolves each local
-- node to Hetzner's id for the same (target, sequence) (inserting it there if
-- missing) and rewrites every node reference (observations.node_id,
-- lineage_edges.parent_node_id/child_node_id) through this map before sending.
-- A row here means "this local node exists on Hetzner with this id". Nodes are
-- immutable and never deleted, so an entry never goes stale.
CREATE TABLE IF NOT EXISTS replication_node_map (
  local_id INTEGER PRIMARY KEY,
  remote_id INTEGER NOT NULL
);

-- Rows Hetzner refused with a data error (too long for a varchar, bad value
-- for an int column, dangling node id, ...), isolated by bisection so one bad
-- row can never wedge replication of the rows behind it.
CREATE TABLE IF NOT EXISTS replication_quarantine (
  table_name TEXT NOT NULL,
  row_key TEXT NOT NULL,
  repl_token INTEGER,
  error TEXT NOT NULL,
  quarantined_at REAL NOT NULL,
  PRIMARY KEY (table_name, row_key)
);

-- NOT replicated to Hetzner: mutation_effects, metadata, edges,
-- proposal_scores. edges/mutation_effects/metadata(graph_version:*) are a
-- derived snapshot that rebuild_nearest_edges recomputes from nodes +
-- observations (both replicated), and pushing a snapshot built from this box's
-- local view would overwrite the graph_version pointer other boxes read.
-- proposal_scores has 0 rows and no reader or writer at HEAD. Their
-- replicated_at columns are unused. They are still covered by the off-box file
-- snapshot (tools/pull_local_sqlite_backup.sh), which copies the whole file.
