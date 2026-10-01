"""Network-free regression tests for strict NIM smoke-result validation."""
import copy
import unittest
from unittest.mock import Mock

import requests

import nim19_benchmark as benchmark


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.case = benchmark.smoke_cases()[1]
        self.payload = {
            "confidence_scores": [0.9],
            "structures": [{"format": "mmcif", "structure": "data_example\n_atom_site.id\n1"}],
            "affinities": {"HEME": {"affinity_pred_value": [1.2], "affinity_probability_binary": [0.8]}},
            "metrics": {"model_inference_time_seconds": 4.0, "unknown_field": "untrusted"},
        }

    def run_case(self, payload=None, status=200):
        session = Mock()
        response = session.post.return_value
        response.status_code = status
        response.json.return_value = self.payload if payload is None else payload
        return benchmark.run_request(session, "http://127.0.0.1:8000", self.case, 0)

    def test_valid_structure_and_affinity(self):
        record = self.run_case()
        self.assertTrue(record["passed"])
        self.assertEqual(record["metrics"], {"model_inference_time_seconds": 4.0})
        self.assertNotIn("structure", record)

    def test_empty_affinity_is_not_http_success(self):
        data = copy.deepcopy(self.payload)
        data["affinities"] = {}
        self.assertFalse(self.run_case(data)["passed"])

    def test_nonfinite_or_bad_probability_rejected(self):
        for value in [None, float("nan"), float("inf"), True, -0.1, 1.1]:
            data = copy.deepcopy(self.payload)
            data["affinities"]["HEME"]["affinity_probability_binary"] = [value]
            self.assertFalse(self.run_case(data)["passed"])

    def test_empty_or_non_mmcif_structure_rejected(self):
        for structure in ["", "not a structure", 42]:
            data = copy.deepcopy(self.payload)
            data["structures"][0]["structure"] = structure
            self.assertFalse(self.run_case(data)["passed"])

    def test_missing_or_nonfinite_confidence_rejected(self):
        for confidence in [[], [float("nan")], [True]]:
            data = copy.deepcopy(self.payload)
            data["confidence_scores"] = confidence
            self.assertFalse(self.run_case(data)["passed"])

    def test_http_error_body_never_read(self):
        record = self.run_case(status=422)
        self.assertEqual(record["failure"], "http_error")
        self.assertNotIn("text", record)

    def test_timeout_classification(self):
        session = Mock()
        session.post.side_effect = requests.exceptions.Timeout("not exported")
        record = benchmark.run_request(session, "http://127.0.0.1:8000", self.case, 0)
        self.assertFalse(record["passed"])
        self.assertEqual(record["failure"], "timeout")

    def test_summary_excludes_failed_and_first_latency_from_warm(self):
        records = [
            {"case": "test", "phase": "first_request", "passed": True, "wall_seconds": 100},
            {"case": "test", "phase": "warm", "passed": True, "wall_seconds": 10},
            {"case": "test", "phase": "warm", "passed": True, "wall_seconds": 20},
            {"case": "test", "phase": "warm", "passed": False, "wall_seconds": 1000},
        ]
        result = benchmark.summary(records)["test"]
        self.assertEqual(result["warm_median_seconds"], 15)
        self.assertEqual(result["warm_predictions_per_hour"], 240)


if __name__ == "__main__":
    unittest.main()
