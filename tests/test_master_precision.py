"""Independent CPU checks of the master-weight precision rationale."""

import unittest

import torch

from replication.probe_master_precision import exhaustive_sr_values, evidence


class MasterPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_small_repeated_updates_survive_fp32_but_stall_in_bf16(self):
        result = evidence()
        self.assertEqual(result["initial_bf16_value"], result["bf16_final"])
        self.assertAlmostEqual(result["initial_bf16_value"] - result["fp32_final"], 32 * 1.8e-5, delta=2e-8)
        self.assertGreater(result["bf16_spacing_here"] / 2, result["update"])

    def test_unchanged_bf16_values_do_not_lose_precision_on_repeated_casts(self):
        torch.manual_seed(42)
        initial = torch.randn(10000).bfloat16()
        value = initial
        for _ in range(32):
            value = value.float().bfloat16()
        torch.testing.assert_close(value, initial, rtol=0, atol=0)

    def test_exhaustive_rounding_offsets_preserve_expectation(self):
        for value in (.013 - 1.8e-5, -.013 + 1.8e-5, 1. - 1e-4, -1. + 1e-4):
            with self.subTest(value=value):
                target, samples = exhaustive_sr_values(value)
                self.assertEqual(samples.numel(), 65536)
                self.assertEqual(samples.unique().numel(), 2)
                self.assertEqual(float(samples.double().mean()), float(target))
