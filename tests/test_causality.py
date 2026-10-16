"""Causality regressions. Run: .venv/bin/python -m unittest discover -s tests -v.

All weights and paging files are temporary; no corpus or checkpoint is needed.
"""

import math
import itertools
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from minagi.paged import PagedPool
from minagi.pool import PooledMLP, capture_routes
from minagi.recur import RecurCoder, RecurConfig, load_recur


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original_tf32 = torch.backends.cuda.matmul.allow_tf32
        # Strict FP32 comparisons must not use TF32's reduced mantissa.
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.original_tf32

    def model(self, *, resident=4, experts=16, capacity=0, freeze=True,
              temperature=0, device="cpu", steps=3, seed=42, recur=1, coda=0,
              hidden=32, width=64):
        torch.manual_seed(seed)
        cfg = RecurConfig(
            vocab_size=265, d_model=hidden, n_head=4, d_ff=width, block=128,
            n_prelude=2, n_recur=recur, n_coda=coda, max_steps=steps,
            min_steps=1, train_steps_mean=0, use_pool=True,
            pool_experts=experts, pool_max=experts, pool_d_ff=width,
            pool_top_k=2, pool_capacity_factor=capacity, halt_freeze=freeze,
        )
        model = RecurCoder(cfg)
        weights = [
            {k: getattr(e, k).weight.detach().clone() for k in ("w1", "w3", "w2")}
            for e in model.pool.experts
        ]
        tmp = tempfile.TemporaryDirectory(prefix="minagi-causality-")
        self.addCleanup(tmp.cleanup)
        pool = PagedPool(tmp.name, hidden, width, experts, resident=resident,
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

    def test_capacity_matches_independent_sequential_oracle(self):
        model = self.model(resident=8)
        site = model.recur[0].mlp
        g = torch.Generator().manual_seed(2026)
        B, T, n, k = 2, 103, 8, 3
        requests = torch.stack([torch.randperm(n, generator=g)[:k]
                                for _ in range(B * T)]).view(B, T, k)
        active = torch.rand(B, T, generator=g) > 0.3
        for factor, budget_n in itertools.product((0.3, 1.5, 4), (n, n // B)):
            accepted = torch.zeros(B, n, dtype=torch.long)
            expected = torch.zeros(B, T, k, dtype=torch.bool)
            for b in range(B):
                for t in range(T):
                    if not active[b, t]:
                        continue
                    limit = max(1, math.ceil(factor * k / budget_n * (t + 1)))
                    for j, expert in enumerate(requests[b, t]):
                        if accepted[b, expert] < limit:
                            expected[b, t, j] = True
                            accepted[b, expert] += 1
            site.capacity_factor = factor
            for width in (1, 7, 29, T):
                actual = torch.zeros_like(expected)
                for start in range(0, T, width):
                    mask = active[:, start:start + width]
                    chunk = requests[:, start:start + width]
                    positions = mask.flatten().nonzero(as_tuple=False).squeeze(1)
                    site.begin_forward(B, chunk.shape[1], start, start == 0)
                    if positions.numel():
                        keep = site._capacity_keep(chunk.flatten(0, 1)[positions],
                                                   positions, n, k, budget_n=budget_n)
                        actual[:, start:start + width][mask] = keep
                with self.subTest(factor=factor, width=width, budget_n=budget_n):
                    self.assertTrue(torch.equal(actual, expected))

    def test_training_predictions_are_causal(self):
        self.assert_prefix(self.model(capacity=1.5), training=True)

    def test_each_sequence_in_a_batch_is_causal(self):
        for capacity in (0, 1.5):
            for temperature in (0, 0.7):
                self.assert_prefix(self.model(resident=8, capacity=capacity,
                                               temperature=temperature), batch=2)
                self.assert_prefix(self.model(resident=8, capacity=capacity,
                                               temperature=temperature),
                                   batch=2, training=True)

    def test_stochastic_routing_is_causal_with_fixed_seed(self):
        self.assert_prefix(self.model(temperature=0.7))

    def test_random_pages_do_not_depend_on_chunk_boundaries(self):
        model = self.model(temperature=0.7)
        pool = model.pool
        positions = [(b, t) for b in range(2) for t in range(257)]
        torch.manual_seed(123)
        pool.begin_text(batch_size=2)
        expected = pool.routing_noise(positions, site=1, depth=3, width=16)
        torch.manual_seed(123)
        pool.begin_text(batch_size=2)
        actual = torch.cat([pool.routing_noise([position], site=1, depth=3, width=16)
                            for position in positions])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(pool._noise_pages), 2)

    def test_multiple_recurrent_sites_and_small_resident_pools(self):
        for resident in (1, 4):
            self.assert_prefix(self.model(resident=resident, freeze=False,
                                          recur=2, coda=1, temperature=0.7))

    def test_configured_24_step_ceiling_is_causal(self):
        self.assert_prefix(self.model(steps=24, capacity=1.5))
        self.assert_prefix(self.model(steps=24, capacity=1.5), training=True)

    def test_fully_resident_pool_capacity_is_causal(self):
        cfg = self.model(capacity=1.5).cfg
        self.assert_prefix(RecurCoder(cfg).eval())

    def test_capacity_ledger_handles_changed_expert_count(self):
        site = self.model().recur[0].mlp
        site.capacity_factor = 1.5
        site.begin_forward(1, 2, 0, True)
        site._capacity_keep(torch.tensor([[0, 1], [1, 2]]), torch.arange(2), 4, 2)
        site.begin_forward(1, 2, 2, False)
        for n in (5, 3):
            keep = site._capacity_keep(torch.tensor([[0, 1], [1, 2]]),
                                       torch.arange(2), n, 2)
            self.assertEqual(keep.shape, (2, 2))
            self.assertEqual(site._capacity_state[0][0].shape, (1, n))

    def test_other_batch_sequences_do_not_change_predictions(self):
        for temperature in (0, 0.7):
            model = self.model(resident=8, temperature=temperature, capacity=1.5)
            prefix, suffix = self.inputs(batch=2)
            text = torch.cat((prefix, suffix), dim=1)
            with torch.no_grad():
                torch.manual_seed(123)
                expected, _ = model(text)
                changed = text.clone()
                changed[1] = 0
                torch.manual_seed(123)
                actual, _ = model(changed)
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=2e-7)

    def test_mixed_adaptive_halting_is_causal(self):
        model = self.model(steps=5, capacity=1.5, temperature=0.7)
        with torch.no_grad():
            model.halt.weight.normal_(0, 2)
            model.halt.bias.zero_()
        self.assert_prefix(model)
        prefix, suffix = self.inputs()
        text = torch.cat((prefix, suffix), dim=1)
        with torch.no_grad():
            torch.manual_seed(123)
            expected, extra = model(text, collect=True)
            self.assertGreater(extra["steps"].unique().numel(), 1)
            self.assertLess(float(extra["steps"].min()), model.cfg.max_steps)
            for width in (1, 3, 7):
                torch.manual_seed(123)
                caches, chunks = model.empty_caches(), []
                for start in range(0, text.shape[1], width):
                    y, _ = model(text[:, start:start + width], caches=caches,
                                 pos_offset=start)
                    chunks.append(y)
                torch.testing.assert_close(torch.cat(chunks, dim=1), expected,
                                           rtol=1e-5, atol=2e-7)
        self.assert_prefix(model, training=True)

    def test_padded_dispatch_and_checkpointing_are_causal(self):
        with patch.dict(os.environ, {"MINAGI_DISPATCH": "padded"}):
            model = self.model(capacity=1.5)
            self.assert_prefix(model, training=True)
            self.assert_prefix(model)

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

    def test_cached_training_gradients_match_full_window(self):
        # FP64 parameters isolate graph/weight correctness from different
        # FP32 GEMM and loss-reduction orders across chunk shapes. The model's
        # explicit FP32 halting/loss computations are still exercised.
        model = self.model(capacity=1.5, steps=5).double().train()
        with torch.no_grad():
            model.halt.weight.normal_(0, 1)
            model.halt.bias.zero_()
        prefix, suffix = self.inputs()
        text = torch.cat((prefix, suffix), dim=1)

        def run(width):
            model.zero_grad(set_to_none=True)
            caches, loss = model.empty_caches(), 0
            for start in range(0, text.shape[1], width):
                chunk = text[:, start:start + width]
                _, value = model(chunk, torch.zeros_like(chunk), caches=caches,
                                 pos_offset=start)
                loss = loss + value * chunk.shape[1] / text.shape[1]
            loss.backward()
            grad = {}
            for name, param in model.named_parameters():
                if name.startswith("pool.w"):
                    value = torch.zeros(model.pool.n_experts(), *param.shape[1:])
                    if param.grad is not None:
                        for slot, expert in enumerate(model.pool.slots):
                            if expert >= 0:
                                value[expert] = param.grad[slot].detach()
                else:
                    value = (torch.zeros_like(param) if param.grad is None
                             else param.grad.detach().clone())
                grad[name] = value
            return loss.detach(), grad

        expected_loss, expected = run(text.shape[1])
        for width in (1, 3, 7):
            loss, actual = run(width)
            torch.testing.assert_close(loss, expected_loss)
            for name in expected:
                with self.subTest(width=width, parameter=name):
                    torch.testing.assert_close(actual[name], expected[name],
                                               rtol=3e-5, atol=3e-7)

    def test_new_text_resets_admission_and_cached_text_keeps_slots(self):
        model = self.model()
        pool = model.pool
        pool.begin_text()
        requests = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
        mask = pool.admit(requests)
        for t, expected in enumerate(({1, 2}, {1, 2, 3, 4}, {1, 2, 3, 4})):
            got = {e for slot, e in enumerate(pool.slots) if mask[0, t, slot]}
            self.assertEqual(got, expected)
        slots = list(pool.slots)
        pool.begin_forward(pos_offset=3)
        pool.admit(torch.tensor([[[7, 8], [9, 10]]]))
        self.assertEqual(pool.slots, slots)
        pool.begin_text()
        pool.admit(torch.tensor([[[7, 8]]]))
        self.assertEqual(pool._admitted, {7, 8})

    def test_fixed_pool_replication_control_prepares_new_masks(self):
        model = self.model()
        prefix, suffix = self.inputs()
        with torch.no_grad():
            model(torch.cat((prefix, suffix), dim=1))
            frozen = set(model.pool._admitted)
            model.pool.freeze_admission = True
            self.assert_prefix(model)
            self.assertEqual(model.pool._admitted, frozen)

    def test_existing_checkpoint_shapes_load_without_migration(self):
        model = self.model()
        # An existing .pt checkpoint contains exactly these weight keys and
        # config fields. Save the original fully resident representation.
        cfg = model.cfg
        torch.manual_seed(42)
        source = RecurCoder(cfg)
        tmp = tempfile.TemporaryDirectory(prefix="minagi-checkpoint-")
        self.addCleanup(tmp.cleanup)
        filename = os.path.join(tmp.name, "old.pt")
        torch.save({"cfg": vars(cfg), "model": source.state_dict()}, filename)
        loaded, _ = load_recur(filename, torch.device("cpu"))
        self.assertEqual(set(loaded.state_dict()), set(source.state_dict()))
        for key in source.state_dict():
            self.assertTrue(torch.equal(loaded.state_dict()[key], source.state_dict()[key]))

    def test_existing_directory_checkpoint_loads_into_causal_paging(self):
        from dataclasses import asdict
        from minagi import store
        from train import build_paged
        expected_model = self.model()
        torch.manual_seed(42)
        source = RecurCoder(expected_model.cfg)
        tmp = tempfile.TemporaryDirectory(prefix="minagi-checkpoint-")
        self.addCleanup(tmp.cleanup)
        cfg = asdict(source.cfg)
        cfg["pool_resident"] = 4
        store.save(source, tmp.name, cfg=cfg)
        loaded, _, _, _ = build_paged(tmp.name, torch.device("cpu"),
                                      resident=4, read_only=True)
        loaded.eval()
        text = torch.cat(self.inputs(), dim=1)
        with torch.no_grad():
            expected, _ = expected_model(text)
            actual, _ = loaded(text)
        torch.testing.assert_close(actual, expected)
        self.assert_prefix(loaded)

    def test_paged_optimizer_restores_incoming_experts_moments(self):
        model = self.model(resident=2, experts=4)
        pool = model.pool
        for e in (2, 3):
            entry = pool.tiers.fetch(e)
            for name in ("w1", "w3", "w2"):
                entry[name + "_m"] = torch.full_like(entry[name], e * 1e-4)
                entry[name + "_v"] = torch.full_like(entry[name], e * 1e-8)
        pool.begin_text()
        pool.admit(torch.tensor([[[0, 1]]]))
        params = [pool.w1, pool.w3, pool.w2]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        pool.attach_optimiser(optimizer)
        for param in params:
            param.grad = torch.zeros_like(param)
        optimizer.step()       # establish the optimizer's moment tensors
        pool.begin_text()
        pool.admit(torch.tensor([[[2, 3]]]))
        pool._own_moments()
        for slot, e in enumerate(pool.slots):
            entry = pool.tiers.fetch(e)
            for name, param in zip(("w1", "w3", "w2"), params):
                self.assertTrue(torch.equal(param[slot], entry[name]))
                self.assertTrue(torch.equal(optimizer.state[param]["exp_avg"][slot],
                                            entry[name + "_m"]))
                self.assertTrue(torch.equal(optimizer.state[param]["exp_avg_sq"][slot],
                                            entry[name + "_v"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_predictions_and_gradients_are_causal(self):
        for capacity in (0, 1.5):
            for temperature in (0, 0.7):
                with self.subTest(capacity=capacity, temperature=temperature):
                    model = self.model(device="cuda", capacity=capacity,
                                       temperature=temperature)
                    self.assert_prefix(model)
                    self.assert_prefix(model, training=True)
                    text = torch.cat(self.inputs("cuda"), dim=1)
                    model.train()
                    _, loss = model(text, torch.zeros_like(text))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    for param in model.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_bf16_predictions_and_backward(self):
        from minagi.precision import amp, compute_dtype, set_compute_dtype
        original_dtype = compute_dtype()
        set_compute_dtype("bf16")
        try:
            for temperature in (0, 0.7):
                model = self.model(device="cuda", capacity=1.5,
                                   temperature=temperature)
                with amp("cuda"):
                    self.assert_prefix(model)
                    self.assert_prefix(model, training=True)
                    text = torch.cat(self.inputs("cuda"), dim=1)
                    model.train()
                    _, loss = model(text, torch.zeros_like(text))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for param in model.parameters():
                    if param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all())
        finally:
            set_compute_dtype(original_dtype)


if __name__ == "__main__":
    unittest.main()
