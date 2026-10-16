"""Backend contract, accepted-route derivatives and model integration.

Run CUDA tests through replication.gpu_monitor. Sonic tests are optional when
the package or a supported GPU is absent; the reference needs only PyTorch.
"""

import importlib.util
import itertools
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import test_causality as fixtures
from minagi.expert_compute import backend_name, compute_experts
from minagi.pool import PooledMLP, capture_routes
from minagi.precision import amp, compute_dtype, set_compute_dtype


def scalar_oracle(x, levels, experts, tokens, scores):
    """One independent assignment at a time, including deeper residual blocks."""
    output = torch.zeros_like(x)
    for e, t, score in zip(experts.tolist(), tokens.tolist(), scores):
        y = x[t:t + 1]
        for w1, w3, w2 in levels:
            block = F.linear(F.silu(F.linear(y, w1[e])) * F.linear(y, w3[e]), w2[e])
            y = block if len(levels) == 1 else y + block
        output = output.index_add(0, tokens.new_tensor([t]), y * score)
    return output


def relative_error(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


class ExpertComputeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_reference_outputs_and_all_derivatives_match_independent_oracle(self):
        for depth, padded, recompute in itertools.product((1, 2), (False, True), (False, True)):
            with self.subTest(depth=depth, padded=padded, recompute=recompute):
                torch.manual_seed(42)
                x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
                levels = [tuple(torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                                for shape in ((4, 8, 4), (4, 8, 4), (4, 4, 8))) for _ in range(depth)]
                scores = torch.tensor([0., -.3, .2, .4, .1], dtype=torch.float64, requires_grad=True)
                experts = torch.tensor([2, 0, 2, 0, 1])
                tokens = torch.tensor([0, 0, 2, 4, 4])  # missing tokens, unused expert
                args = x, levels, experts, tokens, scores
                params = [x, *(w for lv in levels for w in lv), scores]
                y = compute_experts(*args, padded=padded, recompute=recompute)
                expected = scalar_oracle(*args)
                g = torch.randn_like(y)
                actual_grads = torch.autograd.grad(y, params, g, retain_graph=True)
                expected_grads = torch.autograd.grad(expected, params, g)
                torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
                for actual, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
                self.assertNotEqual(float(actual_grads[-1][0]), 0)  # a zero gate still learns
                for grad in actual_grads[1:-1]:
                    self.assertEqual(int(grad[3].count_nonzero()), 0)

    def test_backend_selection_is_explicit_and_invalid_values_fail(self):
        with patch.dict(os.environ, {"MINAGI_EXPERT_BACKEND": "sonic"}):
            self.assertEqual(backend_name(), "sonic")
            self.assertEqual(backend_name("reference"), "reference")
        with self.assertRaisesRegex(ValueError, "unknown expert backend"):
            backend_name("typo")
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            compute_experts(torch.ones(1, 16), [(torch.ones(1, 16, 16),) * 3],
                            torch.tensor([0]), torch.tensor([0]), torch.ones(1), backend="sonic")

    def test_empty_routes_return_zeros(self):
        x = torch.randn(3, 4)
        empty = torch.empty(0, dtype=torch.long)
        for backend in ("reference", "sonic"):
            actual = compute_experts(x, [], empty, empty, torch.empty(0), backend=backend)
            self.assertTrue(torch.equal(actual, torch.zeros_like(x)))


SONIC_AVAILABLE = torch.cuda.is_available() and importlib.util.find_spec("sonicmoe") is not None


@unittest.skipUnless(SONIC_AVAILABLE, "CUDA and optional SonicMoE are required")
class SonicIntegrationTests(unittest.TestCase):
    model = fixtures.CausalityTests.model
    inputs = fixtures.CausalityTests.inputs
    assert_prefix = fixtures.CausalityTests.assert_prefix

    def setUp(self):
        torch.set_num_threads(1)
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("GPU architecture unsupported by QuACK")
        original = compute_dtype()
        self.addCleanup(set_compute_dtype, original)
        set_compute_dtype("bf16")
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", original_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def set_backend(self, model, backend, checkpoint=True):
        for site in model.modules():
            if isinstance(site, PooledMLP):
                site.expert_backend = backend
                site.grad_checkpoint = checkpoint

    def test_accepted_routes_outputs_and_gradients_including_zero_scores(self):
        torch.manual_seed(42)
        x = torch.randn(32, 64, device="cuda", requires_grad=True)
        weights = tuple((torch.randn(*shape, device="cuda") / shape[-1]**.5).requires_grad_()
                        for shape in ((8, 128, 64), (8, 128, 64), (8, 64, 128)))
        pairs = [(t, e) for t in range(32) if t not in (3, 5)
                 for e in (t % 4, (t + 1) % 4)]
        tokens, experts = torch.tensor(pairs, device="cuda").T
        scores = torch.randn(len(pairs), device="cuda") * .2
        scores[0] = 0
        scores.requires_grad_()
        g = torch.randn_like(x)
        params = [x, *weights, scores]
        results = []
        for backend, recompute in (("reference", False), ("sonic", False), ("sonic", True)):
            with amp("cuda"):
                y = compute_experts(x, [weights], experts, tokens, scores,
                                    backend=backend, recompute=recompute)
            grads = torch.autograd.grad(y, params, g)
            results.append((y.detach(), grads))
            for grad in grads:
                self.assertEqual(grad.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(grad).all()))
            for grad in grads[1:4]:
                self.assertEqual(int(grad[4:].count_nonzero()), 0)
            self.assertNotEqual(float(grads[-1][0]), 0)
            self.assertEqual(int(y[[3, 5]].count_nonzero()), 0)
        for y, grads in results[1:]:
            self.assertLess(relative_error(y, results[0][0]), .025)
            for actual, expected in zip(grads, results[0][1]):
                self.assertLess(relative_error(actual, expected), .025)

    def test_model_forward_gradients_cache_and_recomputation(self):
        results = []
        for backend, cached, checkpoint in (("reference", True, False),
                                            ("sonic", False, False),
                                            ("sonic", True, False),
                                            ("sonic", True, True)):
            model = self.model(device="cuda", steps=3, recur=2, coda=1, freeze=False).train()
            self.set_backend(model, backend, checkpoint)
            model.pool.cache_training_weights = cached
            text = torch.cat(self.inputs("cuda"), 1)
            caches, losses, outputs = model.empty_caches(), [], []
            # New admitted experts change cache generations before backward.
            with capture_routes() as routes:
                for start, stop in ((0, 1), (1, 12), (12, 24)):
                    with amp("cuda"):
                        y, loss = model(text[:, start:stop], torch.zeros_like(text[:, start:stop]),
                                        caches=caches, pos_offset=start)
                    outputs.append(y.detach())
                    losses.append(loss * (stop - start) / 24)
            loss = sum(losses)
            loss.backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads.values()))
            results.append((torch.cat(outputs, 1), loss.detach(), grads, routes))
        for y, loss, grads, routes in results[1:]:
            self.assertLess(relative_error(y, results[0][0]), .02)
            torch.testing.assert_close(loss, results[0][1], atol=5e-4, rtol=0)
            self.assertEqual(grads.keys(), results[0][2].keys())
            # Fusion can change a near-tied top-k decision after recurrence.
            # Across backends these are then different computation graphs:
            # compare total gradient discrepancy, not relative error of each
            # tiny, cancellation-sensitive parameter. The fixed-route layer
            # test above checks every individual derivative against reference.
            delta = sum((grads[n] - g).float().square().sum() for n, g in results[0][2].items())
            norm = sum(g.float().square().sum() for g in results[0][2].values())
            self.assertLess(float((delta / norm).sqrt()), .05)
        # Within Sonic, cast caching and checkpointing preserve the computation.
        for y, loss, grads, routes in results[2:]:
            torch.testing.assert_close(y, results[1][0], atol=0, rtol=0)
            for actual, expected in zip(routes, results[1][3]):
                self.assertTrue(torch.equal(actual[0], expected[0]))
            for name, expected in results[1][2].items():
                torch.testing.assert_close(grads[name], expected, atol=1e-9, rtol=1e-5, msg=name)

    def test_causal_prefix_in_training_and_inference(self):
        for training, capacity, batch in itertools.product((False, True), (0, 1.5), (1, 2)):
            model = self.model(device="cuda", capacity=capacity, resident=8,
                               steps=3, temperature=.7)
            self.set_backend(model, "sonic")
            # Isolate expert causality from cuBLAS choosing a different output
            # head GEMM for M=5 versus M=24. Diagnostic hooks demonstrated
            # identical pre-head states with a one-ULP BF16 head difference.
            # Project each row at a fixed shape, retaining BF16 autocast.
            def rowwise_head(x):
                rows = [F.linear(row[None], model.head.weight) for row in x.flatten(0, 1)]
                return torch.cat(rows).view(*x.shape[:2], -1)
            model.head.forward = rowwise_head
            with amp("cuda"), self.subTest(training=training, capacity=capacity, batch=batch):
                self.assert_prefix(model, training=training, batch=batch)

    def test_paging_changes_weight_generations_and_keeps_optimizer_ownership(self):
        model = self.model(device="cuda", resident=2, experts=4).train()
        self.set_backend(model, "sonic")
        p = model.pool
        opt = torch.optim.AdamW(p.parameters(), lr=.001)
        p.attach_optimiser(opt)
        names = ("w1", "w3", "w2")
        # Give every expert distinguishable moments before it enters a slot.
        for e in range(4):
            entry = p.tiers.fetch(e)
            for j, name in enumerate(names):
                entry[name + "_m"] = torch.full_like(entry[name], .01 * (e + 1) + .001 * j)
                entry[name + "_v"] = torch.full_like(entry[name], .0001 * (e + 1))
        # Establish Adam's slab state while all physical slots are empty.
        for name in names:
            getattr(p, name).grad = torch.zeros_like(getattr(p, name))
        opt.step()
        for selected in ((0, 1), (2, 3), (0, 1)):
            p.begin_text()
            p.admit(torch.tensor([[selected]], device="cuda"))
            self.assertEqual(set(p.slots), set(selected))
            incoming = {name: [(p.tiers.fetch(e)[name + "_m"].to("cuda").clone(),
                                p.tiers.fetch(e)[name + "_v"].to("cuda").clone()) for e in p.slots]
                        for name in names}
            x = torch.randn(4, 32, device="cuda", requires_grad=True)
            tokens = torch.arange(4, device="cuda").repeat_interleave(2)
            experts = torch.tensor([0, 1] * 4, device="cuda")
            scores = torch.full((8,), .5, device="cuda")
            with amp("cuda"):
                weights = p.training_weights(torch.bfloat16)
                y = compute_experts(x, [weights], experts, tokens, scores, backend="sonic", recompute=True)
                with torch.no_grad():
                    reference = compute_experts(x, [weights], experts, tokens, scores)
                self.assertLess(relative_error(y, reference), .025)
            opt.zero_grad(set_to_none=True)
            y.square().sum().backward()
            beta1, beta2 = opt.param_groups[0]["betas"]
            expected = {}
            for name in names:
                grad = getattr(p, name).grad
                expected[name] = [
                    (torch.lerp(m, grad[s], 1 - beta1),
                     v.mul(beta2).addcmul(grad[s], grad[s], value=1 - beta2))
                    for s, (m, v) in enumerate(incoming[name])]
            opt.step()
            self.assertIsNone(p._infer)
            for name in names:
                w = getattr(p, name)
                self.assertTrue(bool(torch.isfinite(w).all()))
                for slot, (m, v) in enumerate(expected[name]):
                    torch.testing.assert_close(opt.state[w]["exp_avg"][slot], m, rtol=1e-6, atol=1e-9)
                    torch.testing.assert_close(opt.state[w]["exp_avg_sq"][slot], v, rtol=1e-6, atol=1e-9)
        self.assertIs(p._opt, opt)
        self.assertEqual(set(p.slots), {0, 1})


if __name__ == "__main__":
    unittest.main()
