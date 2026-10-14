"""Causality regressions. Run: .venv/bin/python -m unittest discover -s tests -v.

All weights and paging files are temporary; no corpus or checkpoint is needed.
"""

import tempfile
import unittest

import torch
import torch.nn.functional as F

from minagi.paged import PagedPool
from minagi.pool import PooledMLP, capture_routes
from minagi.recur import RecurCoder, RecurConfig


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, *, resident=4, experts=16, capacity=0, freeze=True,
              temperature=0, device="cpu", steps=3, seed=42):
        torch.manual_seed(seed)
        cfg = RecurConfig(
            vocab_size=265, d_model=32, n_head=4, d_ff=64, block=128,
            n_prelude=2, n_recur=1, n_coda=0, max_steps=steps,
            min_steps=1, train_steps_mean=0, use_pool=True,
            pool_experts=experts, pool_max=experts, pool_d_ff=64,
            pool_top_k=2, pool_capacity_factor=capacity, halt_freeze=freeze,
        )
        model = RecurCoder(cfg)
        weights = [
            {k: getattr(e, k).weight.detach().clone() for k in ("w1", "w3", "w2")}
            for e in model.pool.experts
        ]
        tmp = tempfile.TemporaryDirectory(prefix="minagi-causality-")
        self.addCleanup(tmp.cleanup)
        pool = PagedPool(tmp.name, 32, 64, experts, resident=resident,
                         ram_capacity=experts, device=device, read_only=True).to(device)
        for i, entry in enumerate(weights):
            pool.tiers.put(i, entry, dirty=False)
        for site in model.modules():
            if isinstance(site, PooledMLP):
                site._pool[0] = pool
        model.pool = pool
        pool.attach_sites(model)
        pool.select_temperature = temperature
        return model.to(device).eval()

    def inputs(self, device="cpu", batch=1):
        g = torch.Generator().manual_seed(314)
        prefix = torch.randint(0, 265, (batch, 5), generator=g)
        suffix = torch.randint(0, 265, (batch, 19), generator=g)
        return prefix.to(device), suffix.to(device)

    def assert_prefix(self, model, *, training=False, batch=1):
        device = model.tok_emb.weight.device
        prefix, suffix = self.inputs(device, batch)
        model.train(training)

        def run(text):
            torch.manual_seed(123)
            targets = torch.zeros_like(text) if training else None
            with torch.set_grad_enabled(training), capture_routes() as routes:
                logits, _ = model(text, targets)
            return logits.detach(), routes

        expected, routes = run(prefix)
        for tail in (suffix, suffix.flip(1), torch.zeros_like(suffix)):
            actual, longer_routes = run(torch.cat((prefix, tail), dim=1))
            torch.testing.assert_close(actual[:, :prefix.shape[1]], expected,
                                       rtol=1e-5, atol=2e-7)
            # Before halting compresses active positions, the first row has
            # a route for every position. Compare expert IDs, not VRAM slots.
            a = longer_routes[0][0].view(batch, -1, longer_routes[0][0].shape[-1])
            b = routes[0][0].view(batch, -1, routes[0][0].shape[-1])
            self.assertTrue(torch.equal(a[:, :prefix.shape[1]], b))

    def test_future_text_does_not_change_prefix(self):
        self.assert_prefix(self.model())

    def test_capacity_limit_is_causal(self):
        self.assert_prefix(self.model(capacity=1.5))

    def test_training_predictions_are_causal(self):
        self.assert_prefix(self.model(capacity=1.5), training=True)

    def test_each_sequence_in_a_batch_is_causal(self):
        self.assert_prefix(self.model(resident=8), batch=2)

    def test_stochastic_routing_is_causal_with_fixed_seed(self):
        self.assert_prefix(self.model(temperature=0.7))

    def test_chunked_and_tokenwise_predictions_agree(self):
        for capacity in (0, 1.5):
            for freeze in (False, True):
                for temperature in (0, 0.7):
                    with self.subTest(capacity=capacity, freeze=freeze,
                                      temperature=temperature):
                        model = self.model(capacity=capacity, freeze=freeze,
                                           temperature=temperature)
                        prefix, suffix = self.inputs()
                        text = torch.cat((prefix, suffix), dim=1)

                        def read(width):
                            torch.manual_seed(123)
                            caches, logits = model.empty_caches(), []
                            with torch.no_grad():
                                for start in range(0, text.shape[1], width):
                                    y, _ = model(text[:, start:start + width],
                                                 caches=caches, pos_offset=start)
                                    logits.append(y)
                            return torch.cat(logits, dim=1)

                        expected = read(1)
                        for width in (3, 7, text.shape[1]):
                            torch.testing.assert_close(read(width), expected,
                                                       rtol=1e-5, atol=2e-7)

    def test_prefix_gradient_does_not_depend_on_future_text(self):
        model = self.model(capacity=1.5).train()
        prefix, suffix = self.inputs()

        def gradients(text):
            model.zero_grad(set_to_none=True)
            logits, _ = model(text, torch.zeros_like(text))
            F.cross_entropy(logits[:, :prefix.shape[1]].reshape(-1, 265),
                            torch.zeros(prefix.numel(), dtype=torch.long)).backward()
            result = {}
            for name, param in model.named_parameters():
                if name not in ("pool.w1", "pool.w3", "pool.w2"):
                    result[name] = (torch.zeros_like(param) if param.grad is None
                                    else param.grad.detach().clone())
            for name in ("w1", "w3", "w2"):
                param = getattr(model.pool, name)
                grad = torch.zeros(model.pool.n_experts(), *param.shape[1:])
                if param.grad is not None:
                    for slot, expert in enumerate(model.pool.slots):
                        if expert >= 0:
                            grad[expert] = param.grad[slot].detach()
                result[name] = grad
            return result

        expected = gradients(prefix)
        actual = gradients(torch.cat((prefix, suffix), dim=1))
        for name in expected:
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual[name], expected[name],
                                           rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
