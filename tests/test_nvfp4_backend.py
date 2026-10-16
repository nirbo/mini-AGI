"""Native FP4 qualification. Run CUDA tests under replication.gpu_monitor."""

import importlib.util
import itertools
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import test_causality as fixtures
from minagi import nvfp4
from minagi.expert_compute import compute_experts
from minagi.pool import PooledMLP, capture_routes
from minagi.precision import amp, set_compute_dtype, compute_dtype
from replication.probe_nvfp4 import dequantize, error_metrics, check_kernel


AVAILABLE = torch.cuda.is_available() and importlib.util.find_spec("quartet2") is not None


class RowScaleTests(unittest.TestCase):
    def test_zero_and_power_of_two_scaling_and_no_gradient_edge(self):
        x = torch.tensor([[0., 0.], [.1, -.3], [3., -4.], [1e-12, -1e-10]], requires_grad=True)
        scale = nvfp4.row_scale(x)
        self.assertFalse(scale.requires_grad)
        torch.testing.assert_close(scale[:3], torch.tensor([[1.], [.25], [4.]]), rtol=0, atol=0)
        self.assertTrue(bool(((x / scale).abs() < 2).all()))
        torch.testing.assert_close((x / scale) * scale, x, rtol=0, atol=0)
        torch.testing.assert_close(nvfp4.row_scale(torch.cat((x, x * 1e10)))[:4], scale, rtol=0, atol=0)
        extremes = torch.tensor([[torch.finfo(torch.bfloat16).max], [2.**-133]], dtype=torch.bfloat16)
        normalized = extremes.float() / nvfp4.row_scale(extremes)
        self.assertTrue(bool(torch.isfinite(normalized).all() & (normalized.abs() < 2).all()))

    def test_empty_and_zero_k_native_groups_are_rejected_before_launch(self):
        from collections import namedtuple
        Q = namedtuple("Q", "fp4 micro_scales tensor_scale")
        for m, n, k in ((0, 128, 128), (128, 128, 0), (128, 0, 128)):
            a = Q(torch.empty(m, k // 2, dtype=torch.uint8), None, None)
            b = Q(torch.empty(n, k // 2, dtype=torch.uint8), None, None)
            with self.assertRaisesRegex(ValueError, "positive multiples"):
                nvfp4.grouped_mm([a], [b])


@unittest.skipUnless(AVAILABLE, "CUDA and optional Quartet-II are required")
class NativeFP4Tests(unittest.TestCase):
    inputs = fixtures.CausalityTests.inputs
    assert_prefix = fixtures.CausalityTests.assert_prefix

    def model(self, **kwargs):
        model = fixtures.CausalityTests.model(self, hidden=128, width=256, device="cuda", **kwargs)
        for site in model.modules():
            if isinstance(site, PooledMLP):
                site.expert_backend = "nvfp4"
        return model

    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM120 required")
        torch.manual_seed(42)
        torch.set_num_threads(1)
        original = compute_dtype()
        self.addCleanup(set_compute_dtype, original)
        set_compute_dtype("bf16")

    def test_unequal_expert_scales_and_all_three_native_gemms(self):
        x = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(2, 256, 128, device="cuda", requires_grad=True)
        with torch.no_grad():
            w[0].mul_(.01)
            w[1].mul_(10)
        original = nvfp4.grouped_mm
        records = []

        def record(left, right):
            out = original(left, right)
            ref = torch.cat([dequantize(*a) @ dequantize(*b).T for a, b in zip(left, right)])
            check_kernel(error_metrics(out, ref))
            records.append((len(left), left[0].fp4.shape[1] * 2,
                            torch.stack([a.tensor_scale * b.tensor_scale for a, b in zip(left, right)])))
            return out

        with patch.object(nvfp4, "grouped_mm", record):
            y = nvfp4.linear(x, w, (128, 256))
            y.backward(torch.randn_like(y))
        self.assertEqual(len(records), 4)  # forward, dgrad, two wgrad K buckets
        self.assertEqual([r[1] for r in records], [128, 256, 128, 256])
        self.assertNotEqual(float(records[0][2][0]), float(records[0][2][1]))
        for t in (y, x.grad, w.grad):
            self.assertTrue(bool(torch.isfinite(t).all()))
        self.assertEqual(w.grad.dtype, torch.float32)

    def test_real_checkpoint_widths_against_decoded_operands(self):
        original = nvfp4.grouped_mm
        for k, n in ((512, 2048), (2048, 512)):
            with self.subTest(k=k, n=n):
                records = []
                def record(left, right):
                    out = original(left, right)
                    ref = torch.cat([dequantize(*a) @ dequantize(*b).T for a, b in zip(left, right)])
                    metrics = error_metrics(out, ref)
                    check_kernel(metrics)
                    records.append(metrics)
                    return out
                x = torch.randn(384, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                w = (torch.randn(2, n, k, device="cuda") / k**.5).requires_grad_()
                with patch.object(nvfp4, "grouped_mm", record):
                    y = nvfp4.linear(x, w, (128, 256))
                    y.backward(torch.randn_like(y))
                self.assertEqual(len(records), 4)

    def test_token_local_scales_survive_outliers_other_experts_and_padding(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1, 256, 128, device="cuda") / 128**.5
        with torch.no_grad():
            y = nvfp4.linear(x, w, (128,))
            suffix = torch.randn_like(x) * 1e6
            full = nvfp4.linear(torch.cat((x, suffix)), w, (256,))
            bank = nvfp4.linear(torch.cat((x, suffix)), torch.cat((w, w * 1e4)), (128, 128))
        torch.testing.assert_close(full[:128], y, rtol=0, atol=0)
        torch.testing.assert_close(bank[:128], y, rtol=0, atol=0)

    def test_zero_input_weight_and_gradient_stay_finite(self):
        for zero in ("input", "weight", "gradient"):
            with self.subTest(zero=zero):
                x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(1, 128, 128, device="cuda")
                if zero == "input":
                    x.zero_()
                if zero == "weight":
                    w.zero_()
                x.requires_grad_()
                w.requires_grad_()
                y = nvfp4.linear(x, w, (128,))
                y.backward(torch.zeros_like(y) if zero == "gradient" else torch.ones_like(y))
                for t in (y, x.grad, w.grad):
                    self.assertTrue(bool(torch.isfinite(t).all()))
                if zero in ("input", "weight"):
                    self.assertEqual(int(y.count_nonzero()), 0)
                if zero == "gradient":
                    self.assertEqual(int(x.grad.count_nonzero() + w.grad.count_nonzero()), 0)

    def test_adapter_checkpoint_zero_score_unused_experts_and_repeated_uses(self):
        x = torch.randn(7, 128, device="cuda", requires_grad=True)
        weights = tuple((torch.randn(*shape, device="cuda") / shape[-1]**.5).requires_grad_()
                        for shape in ((4, 256, 128), (4, 256, 128), (4, 128, 256)))
        tokens = torch.tensor([0, 0, 2, 4, 4], device="cuda")
        experts = torch.tensor([2, 0, 2, 0, 1], device="cuda")
        scores = torch.tensor([0., -.3, .2, .4, .1], device="cuda", requires_grad=True)
        g = torch.randn_like(x)
        results = []
        for recompute in (False, True):
            torch.manual_seed(314)
            with amp("cuda"):
                y = nvfp4.experts(x, [weights], experts, tokens, scores, recompute=recompute)
                y2 = nvfp4.experts(x * .5, [weights], experts, tokens, scores, recompute=recompute)
            grads = torch.autograd.grad(y + y2, (x, *weights, scores), g)
            results.append((y, grads))
            self.assertEqual(int(y[[1, 3, 5, 6]].count_nonzero()), 0)
            self.assertNotEqual(float(grads[-1][0]), 0)
            for grad in grads:
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertEqual(grad.dtype, torch.float32)
            for grad in grads[1:4]:
                self.assertEqual(int(grad[3].count_nonzero()), 0)
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_bf16_residual_with_fp32_route_scores(self):
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weights = tuple((torch.randn(*s, device="cuda") / s[-1]**.5).requires_grad_()
                        for s in ((1, 256, 128), (1, 256, 128), (1, 128, 256)))
        scores = torch.tensor([0., -.3, .2, .4], device="cuda", requires_grad=True)
        with amp("cuda"):
            y = compute_experts(x, [weights], torch.zeros(4, device="cuda", dtype=torch.long),
                                torch.arange(4, device="cuda"), scores, backend="nvfp4")
        y.backward(torch.randn_like(y))
        self.assertEqual(y.dtype, x.dtype)
        self.assertEqual(int(y[0].count_nonzero()), 0)
        self.assertNotEqual(float(scores.grad[0]), 0)
        for t in (x, *weights, scores):
            self.assertTrue(bool(torch.isfinite(t.grad).all()))

    def test_recurrent_model_causality_training_eval_capacity_and_batch(self):
        for training, capacity, batch in itertools.product((False, True), (0, 1.5), (1, 2)):
            with self.subTest(training=training, capacity=capacity, batch=batch):
                model = self.model(capacity=capacity, resident=8, steps=3, temperature=.7)
                # Keep head GEMM row shape fixed to isolate expert causality
                # from unrelated BF16 cuBLAS shape-dependent rounding.
                def rowwise_head(x):
                    rows = [F.linear(row[None], model.head.weight) for row in x.flatten(0, 1)]
                    return torch.cat(rows).view(*x.shape[:2], -1)
                model.head.forward = rowwise_head
                with amp("cuda"):
                    self.assert_prefix(model, training=training, batch=batch)
                    if batch == 2:
                        prefix, suffix = self.inputs("cuda", batch=2)
                        text = torch.cat((prefix, suffix), 1)
                        with torch.set_grad_enabled(training):
                            torch.manual_seed(123)
                            expected, _ = model(text, torch.zeros_like(text) if training else None)
                            changed = text.clone()
                            changed[1] = 0
                            torch.manual_seed(123)
                            actual, _ = model(changed, torch.zeros_like(text) if training else None)
                        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=2e-7)

    def test_recurrent_model_cache_checkpoint_outputs_and_gradients(self):
        results = []
        for cached, recompute in itertools.product((False, True), (False, True)):
            model = self.model(steps=3, recur=2, coda=1, freeze=False).train()
            model.pool.cache_training_weights = cached
            for site in model.modules():
                if isinstance(site, PooledMLP):
                    site.grad_checkpoint = recompute
            prefix, suffix = self.inputs("cuda")
            text = torch.cat((prefix, suffix), 1)
            torch.manual_seed(2718)
            with amp("cuda"), capture_routes() as routes:
                y, loss = model(text, torch.zeros_like(text))
            loss.backward()
            grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads.values()))
            results.append((y.detach(), grads, routes))
        for y, grads, routes in results[1:]:
            torch.testing.assert_close(y, results[0][0], rtol=0, atol=0)
            for name, expected in results[0][1].items():
                torch.testing.assert_close(grads[name], expected, rtol=1e-5, atol=1e-9, msg=name)
            for actual, expected in zip(routes, results[0][2]):
                self.assertTrue(torch.equal(actual[0], expected[0]))

    def test_paging_optimizer_moments_stay_with_expert_identity(self):
        model = self.model(resident=2, experts=4).train()
        p = model.pool
        opt = torch.optim.AdamW(p.parameters(), lr=.001)
        p.attach_optimiser(opt)
        names = ("w1", "w3", "w2")
        for e in range(4):
            entry = p.tiers.fetch(e)
            for j, name in enumerate(names):
                entry[name + "_m"] = torch.full_like(entry[name], .01 * (e + 1) + .001 * j)
                entry[name + "_v"] = torch.full_like(entry[name], .0001 * (e + 1))
        for name in names:
            getattr(p, name).grad = torch.zeros_like(getattr(p, name))
        opt.step()
        for selected in ((0, 1), (2, 3), (0, 1)):
            p.begin_text()
            p.admit(torch.tensor([[selected]], device="cuda"))
            incoming = {name: [(p.tiers.fetch(e)[name + "_m"].to("cuda").clone(),
                                p.tiers.fetch(e)[name + "_v"].to("cuda").clone()) for e in p.slots]
                        for name in names}
            x = torch.randn(4, 128, device="cuda", requires_grad=True)
            with amp("cuda"):
                y = compute_experts(x, [p.training_weights(torch.bfloat16)],
                                    torch.tensor([0, 1, 0, 1], device="cuda"),
                                    torch.arange(4, device="cuda"), torch.ones(4, device="cuda"),
                                    backend="nvfp4", recompute=True)
            opt.zero_grad(set_to_none=True)
            y.square().sum().backward()
            beta1, beta2 = opt.param_groups[0]["betas"]
            expected = {}
            for name in names:
                grad = getattr(p, name).grad
                expected[name] = [(torch.lerp(m, grad[s], 1 - beta1),
                                   v.mul(beta2).addcmul(grad[s], grad[s], value=1 - beta2))
                                  for s, (m, v) in enumerate(incoming[name])]
            opt.step()
            self.assertIsNone(p._infer)
            self.assertEqual(set(p.slots), set(selected))
            for name in names:
                w = getattr(p, name)
                self.assertTrue(bool(torch.isfinite(w).all()))
                for slot, (m, v) in enumerate(expected[name]):
                    torch.testing.assert_close(opt.state[w]["exp_avg"][slot], m, rtol=1e-6, atol=1e-9)
                    torch.testing.assert_close(opt.state[w]["exp_avg_sq"][slot], v, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
