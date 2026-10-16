"""Cached compute storage must preserve cast-per-use training semantics."""

import itertools
import os
import unittest
from unittest.mock import patch

import torch

import test_causality as fixtures
from minagi.precision import amp, compute_dtype, set_compute_dtype


class WeightCacheTests(unittest.TestCase):
    model = fixtures.CausalityTests.model
    inputs = fixtures.CausalityTests.inputs
    assert_prefix = fixtures.CausalityTests.assert_prefix

    def setUp(self):
        torch.set_num_threads(1)
        original = compute_dtype()
        self.addCleanup(set_compute_dtype, original)
        set_compute_dtype("bf16")
        env = patch.dict(os.environ, {"MINAGI_CPU_BF16": "1"})
        env.start()
        self.addCleanup(env.stop)

    def assert_casts(self, pool, data):
        for master, cast in zip((pool.w1, pool.w3, pool.w2), data):
            self.assertTrue(torch.equal(master.detach().to(torch.bfloat16), cast))

    def test_per_use_gradients_accumulate_in_fp32_not_bf16(self):
        pool = self.model().pool
        uses = [pool.training_weights(torch.bfloat16)[0] for _ in range(3)]
        self.assertEqual(len({t.data_ptr() for t in uses}), 1)
        self.assertEqual(len({t.grad_fn for t in uses}), 3)
        gradients = [torch.full_like(t, n) for t, n in zip(uses, (256, 1, 2))]
        torch.autograd.backward(uses, gradients)
        self.assertEqual(pool.w1.grad.dtype, torch.float32)
        self.assertTrue(torch.equal(pool.w1.grad, torch.full_like(pool.w1, 259)))
        # A naive shared differentiable cast cannot represent this sum in BF16.
        master = torch.zeros_like(pool.w1, requires_grad=True)
        shared = master.to(torch.bfloat16)
        torch.autograd.backward([shared] * 3, gradients)
        self.assertFalse(torch.equal(master.grad, pool.w1.grad))
        self.assertTrue(all(not t.requires_grad and t.grad_fn is None
                            for t in pool._infer[1]))

    def test_old_generation_retains_values_and_second_derivative(self):
        pool = self.model().pool
        with torch.no_grad():
            pool.w1.fill_(1)
        old = pool.training_weights(torch.bfloat16)[0]
        with torch.no_grad():
            pool.w1.fill_(2)
        new = pool.training_weights(torch.bfloat16)[0]
        self.assertNotEqual(old.data_ptr(), new.data_ptr())
        self.assertTrue(torch.equal(old, torch.ones_like(old)))
        (old.square().sum() + new.square().sum()).backward()
        self.assertTrue(torch.equal(pool.w1.grad, torch.full_like(pool.w1, 6)))
        pool.w1.grad = None
        w = pool.training_weights(torch.bfloat16)[0]
        first, = torch.autograd.grad(w.square().sum(), pool.w1, create_graph=True)
        second, = torch.autograd.grad(first.sum(), pool.w1)
        self.assertTrue(torch.equal(second, torch.full_like(pool.w1, 2)))

    def test_invalidation_updates_replacements_storage_and_state_load(self):
        pool = self.model().pool
        previous = pool.inference_weights(torch.bfloat16)

        def check_change():
            nonlocal previous
            actual = pool.inference_weights(torch.bfloat16)
            self.assertTrue(all(a.data_ptr() != b.data_ptr()
                                for a, b in zip(previous, actual)))
            self.assert_casts(pool, actual)
            previous = actual

        # A normal optimizer without pool hooks still advances tensor versions.
        opt = torch.optim.SGD(pool.parameters(), lr=1)
        pool.w1.grad = torch.ones_like(pool.w1)
        opt.step()
        check_change()
        state = {k: t.clone() for k, t in pool.state_dict().items()}
        state["w3"].add_(1)
        pool.load_state_dict(state)
        check_change()
        # Replace only w3, preserving its version and shape; w1's pointer alone
        # would miss this. Also cover unversioned replacement of w2's storage.
        pool.w3 = torch.nn.Parameter(pool.w3.detach().clone() + 2)
        check_change()
        pool.w2.data = pool.w2.data.clone() + 3
        check_change()
        pool.w1.data.add_(1)
        pool.invalidate()  # external unversioned edits must explicitly signal
        check_change()
        pool.double()
        self.assertIsNone(pool._infer)
        check_change()
        pool.load_state_dict(state, assign=True)
        check_change()

    def test_paging_invalidates_unversioned_loads_and_optimizer_hook(self):
        pool = self.model(resident=2, experts=4).pool
        pool.begin_text()
        pool.admit(torch.tensor([[[0, 1]]]))
        previous = pool.inference_weights(torch.bfloat16)
        versions = tuple(t._version for t in (pool.w1, pool.w3, pool.w2))
        pool.begin_text()
        pool.admit(torch.tensor([[[2, 3]]]))
        self.assertEqual(versions, tuple(t._version for t in (pool.w1, pool.w3, pool.w2)))
        self.assertIsNone(pool._infer)
        self.assert_casts(pool, pool.inference_weights(torch.bfloat16))
        self.assertTrue(all(a.data_ptr() != b.data_ptr()
                            for a, b in zip(previous, pool._infer[1])))
        opt = torch.optim.AdamW(pool.parameters(), lr=1e-3)
        pool.attach_optimiser(opt)
        for param in pool.parameters():
            param.grad = torch.ones_like(param)
        opt.step()
        self.assertIsNone(pool._infer)
        self.assert_casts(pool, pool.training_weights(torch.bfloat16))

    def test_inference_mode_cache_can_be_reused_for_training_and_fp32_is_noop(self):
        pool = self.model().pool
        with torch.inference_mode():
            data = pool.inference_weights(torch.bfloat16)
        pool.begin_forward(explore=True)
        weights = pool.training_weights(torch.bfloat16)
        self.assertEqual([t.data_ptr() for t in weights], [t.data_ptr() for t in data])
        sum(t.square().sum() for t in weights).backward()
        self.assertTrue(all(t.grad is not None for t in (pool.w1, pool.w3, pool.w2)))
        pool.invalidate()
        masters = (pool.w1, pool.w3, pool.w2)
        self.assertTrue(all(a is b for a, b in zip(masters, pool.training_weights(torch.float32))))
        self.assertIsNone(pool._infer)
        pool.cache_training_weights = False
        self.assertTrue(all(a is b for a, b in zip(masters, pool.training_weights(torch.bfloat16))))

    def compare_training(self, device):
        for dispatch, checkpoint in itertools.product(("exact", "padded"), (False, True)):
            with self.subTest(device=device, dispatch=dispatch, checkpoint=checkpoint), \
                    patch.dict(os.environ, {"MINAGI_DISPATCH": dispatch}):
                results = []
                for cached in (False, True):
                    model = self.model(device=device, steps=4, recur=2, coda=1, freeze=False).train()
                    model.pool.cache_training_weights = cached
                    for site in model.modules():
                        if hasattr(site, "grad_checkpoint"):
                            site.grad_checkpoint = checkpoint
                    text = torch.cat(self.inputs(device), dim=1)
                    # Multiple frames grow the working set before a shared
                    # backward; separate bridges must survive checkpoint replay.
                    caches, loss, logits = model.empty_caches(), 0, []
                    for start in (0, 1, 12):
                        stop = {0: 1, 1: 12, 12: 24}[start]
                        with amp(device):
                            y, term = model(text[:, start:stop], torch.zeros_like(text[:, start:stop]),
                                            caches=caches, pos_offset=start)
                        loss = loss + term * (stop - start) / 24
                        logits.append(y.detach())
                    loss.backward()
                    results.append((loss.detach(), torch.cat(logits, 1), {
                        name: param.grad.detach().clone() for name, param in model.named_parameters()
                        if param.grad is not None}))
                for a, b in zip(results[0][:2], results[1][:2]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(results[0][2].keys(), results[1][2].keys())
                for name, expected in results[0][2].items():
                    torch.testing.assert_close(results[1][2][name], expected, rtol=0, atol=0,
                                               msg=lambda msg: f"{name}: {msg}")

    def test_cpu_bf16_training_matches_uncached(self):
        self.compare_training("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_bf16_training_matches_uncached(self):
        self.compare_training("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_bf16_cached_training_remains_causal(self):
        for dispatch in ("exact", "padded"):
            with patch.dict(os.environ, {"MINAGI_DISPATCH": dispatch}), amp("cuda"):
                self.assert_prefix(self.model(device="cuda", capacity=1.5), training=True)


if __name__ == "__main__":
    unittest.main()
