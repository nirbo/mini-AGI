import unittest

import torch

from replication.probe_nvfp4 import dequantize, check_kernel


class Nvfp4OracleTests(unittest.TestCase):
    def test_e2m1_nibble_order_sign_and_global_scale(self):
        packed = torch.full((128, 64), 0xD2, dtype=torch.uint8)
        actual = dequantize(packed, torch.ones(128, 8), torch.tensor(2.))
        self.assertTrue(torch.equal(actual[:, ::2], torch.full((128, 64), 2.)))
        self.assertTrue(torch.equal(actual[:, 1::2], torch.full((128, 64), -6.)))

    def test_scale_layout_known_boundary_coordinates(self):
        packed = torch.full((128, 64), 0x22, dtype=torch.uint8)
        scales = torch.arange(1, 1025, dtype=torch.float32).reshape(128, 8)
        actual = dequantize(packed, scales, torch.tensor(2.))
        for row, col, expected in ((0, 0, 2), (0, 16, 4), (31, 0, 994),
                                   (32, 0, 10), (127, 0, 1018), (0, 64, 1026)):
            self.assertEqual(float(actual[row, col]), expected)

    def test_kernel_acceptance_rejects_nonfinite_and_wrong_values(self):
        check_kernel({"finite": True, "relative_l2": .002})
        for metrics in ({"finite": False, "relative_l2": 0}, {"finite": True, "relative_l2": .1}):
            with self.assertRaises(AssertionError):
                check_kernel(metrics)

    def test_scale_layout_crosses_128_row_tile_boundary(self):
        packed = torch.full((256, 64), 0x22, dtype=torch.uint8)
        scales = torch.arange(1, 2049, dtype=torch.float32).reshape(256, 8)
        actual = dequantize(packed, scales, torch.tensor(2.))
        for row, col, expected in ((128, 0, 2050), (255, 64, 4090)):
            self.assertEqual(float(actual[row, col]), expected)


if __name__ == "__main__":
    unittest.main()
