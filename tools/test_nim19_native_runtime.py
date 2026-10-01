"""Network-free tests for extracted-image environment isolation."""
import json
import unittest
from unittest.mock import Mock, patch

import nim19_native_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_public_image_settings_restored_but_unrelated_environment_not_forwarded(self):
        config = {"Env": [
            "PATH=/usr/local/bin:/usr/bin:/bin", "BASH_ENV=/etc/bash.bashrc",
            "NIMTOOLS_VERSION=1.12.0", "NIM_LIB_PATH=/opt/nim/lib",
            "LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/cuda/lib64",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1",
        ]}
        with patch.object(runtime.Path, "read_text", return_value=json.dumps(config)), patch.dict(runtime.os.environ, {"RENTAL_PRIVATE_MARKER": "excluded"}):
            env = runtime.environment()
        self.assertNotIn("RENTAL_PRIVATE_MARKER", env)
        self.assertEqual(env["BASH_ENV"], "/etc/bash.bashrc")
        self.assertEqual(env["NIMTOOLS_VERSION"], "1.12.0")
        self.assertEqual(env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "1")
        self.assertNotIn("/compat", env["LD_LIBRARY_PATH"])
        self.assertEqual(env["NIM_DEFAULT_RANDOM_SEED"], "68")
        self.assertEqual(env["NIM_DISABLE_MODEL_DOWNLOAD"], "1")

    def test_probe_requires_exactly_one_cuda_device(self):
        for visible, devices, expected in [(True, 1, 0), (False, 0, 1), (True, 2, 1)]:
            result = Mock(returncode=0, stdout=b"__NIM_PROBE__"+json.dumps({"gpu_visible": visible, "gpus": devices, "passed": visible}).encode(), stderr=b"")
            with patch.object(runtime, "prepare"), patch.object(runtime, "environment", return_value={}), patch.object(runtime.subprocess, "run", return_value=result), patch.object(runtime, "emit"):
                self.assertEqual(runtime.probe(), expected)

    def test_process_identity_uses_numeric_start_ticks(self):
        fields=['S']+['0']*18+['123456']+['0']*3
        with patch.object(runtime.Path,'read_text',return_value='42 (python) '+' '.join(fields)):
            self.assertEqual(runtime.process_start_ticks(42),123456)
        with patch.object(runtime.Path,'read_text',side_effect=FileNotFoundError):
            self.assertIsNone(runtime.process_start_ticks(42))


if __name__ == "__main__":
    unittest.main()
