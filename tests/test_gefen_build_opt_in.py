"""CPU-only tests for the optional local Gefen compiler preflight patch."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("gefen"), "optional Gefen required")
class GefenBuildOptInTests(unittest.TestCase):
    def test_default_rejects_minor_mismatch_and_opt_in_preserves_major_rejection(self):
        import gefen.kernels._build_notice as build
        with patch.object(build, "_nvcc_path", return_value="/usr/local/cuda/bin/nvcc"), \
             patch.object(build, "_run_command", return_value="release 13.1, V13.1.115"), \
             patch.object(build, "_torch_cuda_version", return_value=(13, 0)), \
             patch.object(build, "_cuda_diagnostic", return_value="mock diagnostic"), \
             patch.object(build, "_cuda_install_instructions", return_value="mock instructions"), \
             patch.object(build, "_find_cuda_runtime_header", return_value=Path("/mock/cuda_runtime.h")):
            with patch.dict("os.environ", {"MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH": ""}):
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    build._check_cuda_toolkit()
            with patch.dict("os.environ", {"MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH": "1"}):
                self.assertEqual(build._check_cuda_toolkit()[0], "/usr/local/cuda/bin/nvcc")
                with patch.object(build, "_torch_cuda_version", return_value=(12, 8)):
                    with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                        build._check_cuda_toolkit()
