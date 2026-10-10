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
  replicated_at REAL
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
  replicated_at REAL
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
  PRIMARY KEY (target, run_id)
);
CREATE INDEX IF NOT EXISTS idx_exact_runs_unreplicated ON exact_runs(replicated_at) WHERE replicated_at IS NULL;

CREATE TABLE IF NOT EXISTS exact_run_observations (
  target TEXT NOT NULL,
  run_id TEXT NOT NULL,
  observation_id TEXT NOT NULL,
  replicated_at REAL,
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
