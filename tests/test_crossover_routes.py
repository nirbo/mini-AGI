"""Independent invariants for the balanced performance workload metadata."""

import unittest

import torch

from replication.sweep_nvfp4_crossover import routes


class CrossoverRouteTests(unittest.TestCase):
    def test_balanced_counts_unique_experts_and_token_major_order(self):
        for e in (1, 3, 6, 8, 16, 32):
            for m in (1, 2, 128):
                for k in range(1, e + 1):
                    if e * m % k:
                        continue
                    ids, tokens = routes(e, m, k, "cpu")
                    with self.subTest(experts=e, rows=m, topk=k):
                        self.assertEqual(ids.numel(), e * m)
                        self.assertTrue(torch.equal(torch.bincount(ids, minlength=e), torch.full((e,), m)))
                        self.assertTrue(bool((tokens[1:] >= tokens[:-1]).all()))
                        for token in range(e * m // k):
                            selected = ids[tokens == token]
                            self.assertEqual(selected.numel(), k)
                            self.assertEqual(selected.unique().numel(), k)

    def test_invalid_dimensions_fail(self):
        for e, m, k in ((0, 128, 1), (2, 0, 1), (2, 128, 3), (3, 1, 2)):
            with self.assertRaises(ValueError):
                routes(e, m, k, "cpu")
