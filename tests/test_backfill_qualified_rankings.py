"""Stdlib-only checks for retrospective strict artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nanobody_opt.publication_gates import gate_defaults
from tools import backfill_qualified_rankings as backfill


def payload():
    return {
        "target": "P20809",
        "qualification": {"pose_stability": {
            "minimum_structure_observations": 1,
            "minimum_median_contact_jaccard": .35,
            "maximum_cdr3_rmsd": 5,
            "minimum_epitope_occupancy": 0,
        }},
        "candidates": [{
            "observation_count": 1, "competitive_rate": .1,
            "observation_competitive_rate": .1, "score_p90": 200,
            "p90_category_rank_sums": {"confidence": 150, "physical_interaction": 40, "developability": 10},
            "pose_stability": {
                "structure_observation_count": 1,
                "contact_jaccard_median": .5,
                "cdr3_rmsd_max_angstrom": 2,
                "epitope_occupancy_min": 0,
            },
        }],
    }


class ValidationTests(unittest.TestCase):
    def test_real_gate_passing_profile_is_accepted(self):
        backfill.validate(payload(), "P20809", gate_defaults())

    def test_missing_structural_evidence_is_not_qualified(self):
        data = payload()
        data["candidates"][0]["pose_stability"]["structure_observation_count"] = 0
        with self.assertRaises(ValueError):
            backfill.validate(data, "P20809", gate_defaults())

    def test_nonfinite_or_failing_pose_is_rejected(self):
        for field, value in (("contact_jaccard_median", .34),
                             ("cdr3_rmsd_max_angstrom", 5.01),
                             ("cdr3_rmsd_max_angstrom", float("nan")),
                             ("epitope_occupancy_min", float("inf"))):
            with self.subTest(field=field, value=value):
                data = payload()
                data["candidates"][0]["pose_stability"][field] = value
                with self.assertRaises(ValueError):
                    backfill.validate(data, "P20809", gate_defaults())

    def test_scalar_quality_and_evidence_are_not_bypassed(self):
        for field, value in (("observation_count", 0), ("score_p90", 401),
                             ("competitive_rate", float("nan"))):
            with self.subTest(field=field):
                data = payload()
                data["candidates"][0][field] = value
                with self.assertRaises(ValueError):
                    backfill.validate(data, "P20809", gate_defaults())
        data = payload()
        data["candidates"][0]["p90_category_rank_sums"]["developability"] = 31
        with self.assertRaises(ValueError):
            backfill.validate(data, "P20809", gate_defaults())

    def test_different_gate_policy_is_rejected(self):
        data = payload()
        data["qualification"]["pose_stability"]["maximum_cdr3_rmsd"] = 50
        with self.assertRaises(ValueError):
            backfill.validate(data, "P20809", gate_defaults())

    def test_empty_or_wrong_target_is_rejected(self):
        data = payload()
        with self.assertRaises(ValueError):
            backfill.validate(data, "wrong", gate_defaults())
        data["candidates"] = []
        with self.assertRaises(ValueError):
            backfill.validate(data, "P20809", gate_defaults())

    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "live" / "epoch25170"
            directory.mkdir(parents=True)
            destination = directory / "qualified_rankings.json"
            destination.write_text("existing evidence\n")
            with patch.object(backfill, "emit"), patch.object(backfill.subprocess, "Popen") as process:
                self.assertTrue(backfill.run_epoch(root, 25170, 30, 60))
                process.assert_not_called()
            self.assertEqual(destination.read_text(), "existing evidence\n")

    def test_completed_backfill_preserves_policy_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "live" / "epoch25170"
            directory.mkdir(parents=True)
            exact = directory / "exact_all.json"
            exact.write_text("{}")
            (directory / "provisional_rankings.json").write_text(json.dumps({
                "target": "P20809", "exact_files": [str(exact)],
                "selection_order": "win-probability", "consensus_ranking_policy": "off",
                "selection_metric_policy": "plain",
            }))
            for name in ("config.yaml", "archive.csv", "excluded_sequences.txt"):
                (directory / name).write_text("retained input")
            source = root / "nanobody_opt"
            source.mkdir()
            for name in ("rank_exact.py", "pose_stability.py", "publication_gates.py", "win_probability.py"):
                (source / name).write_text("--consensus-ranking-policy --selection-metric-policy")

            class Completed:
                returncode = 0

                def communicate(self, timeout):
                    return b"UNTRUSTED_CHILD_OUTPUT", b"UNTRUSTED_CHILD_ERROR"

                def poll(self):
                    return 0

            def launch(command, **kwargs):
                self.assertEqual(command[command.index("--before-epoch") + 1], "25170")
                self.assertEqual(command[command.index("--consensus-ranking-policy") + 1], "off")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertFalse(any("commit" in part or "wallet" in part for part in command))
                Path(command[command.index("--output") + 1]).write_text(json.dumps(payload()))
                return Completed()

            with patch.object(backfill.subprocess, "Popen", side_effect=launch), patch.object(backfill, "emit") as events:
                self.assertTrue(backfill.run_epoch(root, 25170, 30, 60))
                self.assertNotIn("UNTRUSTED_CHILD", str(events.call_args_list))
            result = json.loads((directory / "qualified_rankings.json").read_text())
            self.assertTrue(result["reconstruction"]["retrospective"])
            self.assertFalse(result["reconstruction"]["chain_submission_performed"])
            self.assertEqual(list(directory.glob(".qualified_backfill_*")), [])

    def test_changed_inputs_fail_closed_and_clean_the_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "live" / "epoch25170"
            directory.mkdir(parents=True)
            exact = directory / "exact_all.json"
            exact.write_text("{}")
            (directory / "provisional_rankings.json").write_text(json.dumps({
                "target": "P20809", "exact_files": [str(exact)],
            }))
            for name in ("config.yaml", "archive.csv", "excluded_sequences.txt"):
                (directory / name).write_text("retained input")
            source = root / "nanobody_opt"
            source.mkdir()
            for name in ("rank_exact.py", "pose_stability.py", "publication_gates.py", "win_probability.py"):
                (source / name).write_text("stub")

            class Mutating:
                returncode = 0

                def communicate(self, timeout):
                    exact.write_text("changed during ranking")
                    return b"", b""

                def poll(self):
                    return 0

            with patch.object(backfill.subprocess, "Popen", return_value=Mutating()), patch.object(backfill, "emit"):
                with self.assertRaises(ValueError):
                    backfill.run_epoch(root, 25170, 30, 60)
            self.assertFalse((directory / "qualified_rankings.json").exists())
            self.assertEqual(list(directory.glob(".qualified_backfill_*")), [])

    def test_future_epoch_inputs_are_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "live" / "epoch25170"
            directory.mkdir(parents=True)
            future = root / "live" / "epoch25171" / "exact_all.json"
            future.parent.mkdir(parents=True)
            future.write_text("{}")
            (directory / "provisional_rankings.json").write_text(json.dumps({"exact_files": [str(future)]}))
            with patch.object(backfill.subprocess, "Popen") as process:
                with self.assertRaises(ValueError):
                    backfill.run_epoch(root, 25170, 30, 60)
                process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
