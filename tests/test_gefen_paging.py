"""UID ownership and lossless optimizer serialization against stock Gefen.

CPU tests never use CUDA. CUDA qualification runs separately under monitoring.
"""

import copy
from dataclasses import asdict
import importlib.util
import importlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from minagi import optim_codec, store
from minagi.optimizers import EXPERT_PREFIX, NAMES, PagedGefen, make_optimizer
from minagi.paged import PagedPool
from minagi.pool import PooledMLP
from minagi.recur import RecurCoder, RecurConfig, load_any


def assert_tree(test, actual, expected):
    if torch.is_tensor(expected):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertTrue(torch.equal(actual.cpu(), expected.cpu()))
    elif isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_tree(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(type(actual), type(expected))
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_tree(test, a, b)
    else:
        test.assertEqual(actual, expected)


class OptimizerCodecTests(unittest.TestCase):
    def test_lossless_nested_dtype_roundtrip_without_pickle(self):
        tree = {7: (torch.arange(256, dtype=torch.uint8), torch.tensor([-.3, .012], dtype=torch.bfloat16)),
                "metadata": [None, True, 3, 1e-8, "expert"],
                "f32": torch.tensor([1e-30, 1.0000001]), "step": torch.tensor(17.)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.npz")
            np.savez(path, **optim_codec.encode(tree))
            with np.load(path, allow_pickle=False) as arrays:
                assert_tree(self, optim_codec.decode(arrays), tree)
        arrays = optim_codec.encode(tree)
        metadata = json.loads(arrays["__optim_json"].tobytes())
        metadata["version"] = 999
        arrays["__optim_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "version"):
            optim_codec.decode(arrays)


@unittest.skipUnless(importlib.util.find_spec("gefen"), "optional Gefen required")
class GefenPagingTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(prefix="minagi-gefen-")
        self.addCleanup(self.tmp.cleanup)
        torch.manual_seed(42)
        module = importlib.import_module("gefen.gefen")
        previous = module.FIND_PERIOD_BACKEND
        self.addCleanup(setattr, module, "FIND_PERIOD_BACKEND", previous)
        module.FIND_PERIOD_BACKEND = None

    def pool_model(self, experts=5, resident=2):
        model = torch.nn.Module()
        model.pool = PagedPool(os.path.join(self.tmp.name, "experts"), 16, 32, experts,
                               resident=resident, ram_capacity=1).to(self.device)
        os.makedirs(model.pool.path)
        model.aux = torch.nn.Parameter(torch.randn(3, device=self.device))
        originals = []
        for uid in range(experts):
            entry = {name: torch.randn(*getattr(model.pool, name).shape[1:]) * .1 for name in NAMES}
            originals.append(entry)
            model.pool.tiers.put(uid, entry)
        model.pool.tiers.flush()
        return model, originals

    def optimizer(self, model):
        opt = PagedGefen(model.parameters(), model.pool, lr=.001, weight_decay=.1, fused=self.device == "cuda")
        model.pool.attach_optimiser(opt)
        return opt

    def select(self, pool, ids):
        pool._rearrange(ids, {e: s for s, e in enumerate(pool.slots) if e >= 0})

    def recurrent_model(self):
        cfg = RecurConfig(vocab_size=265, d_model=16, n_head=2, d_ff=32,
                          block=64, n_prelude=1, n_recur=1, n_coda=0,
                          max_steps=2, min_steps=2, bptt_window=2,
                          train_steps_mean=0, use_pool=True, pool_experts=5,
                          pool_max=5, pool_d_ff=32, pool_top_k=2)
        model = RecurCoder(cfg)
        path = os.path.join(self.tmp.name, "experts")
        os.makedirs(path)
        pool = PagedPool(path, 16, 32, 5, resident=2, ram_capacity=1)
        for uid, expert in enumerate(model.pool.experts):
            pool.tiers.put(uid, {name: getattr(expert, name).weight.detach().clone() for name in NAMES})
        pool.tiers.flush()
        for site in model.modules():
            if isinstance(site, PooledMLP):
                site._pool[0] = pool
        model.pool = pool
        pool.attach_sites(model)
        return model.to(self.device)

    def deterministic_step(self, model, opt, tick):
        opt.zero_grad(set_to_none=True)
        for index, (name, p) in enumerate(model.named_parameters()):
            if opt.uses_gefen_parameter(p):
                p.grad = torch.empty_like(p)
                for slot, position in enumerate(model.pool.slots):
                    uid = model.pool._f(position)
                    generator = torch.Generator().manual_seed(tick * 1000 + uid * 10 + NAMES.index(name[5:]))
                    p.grad[slot] = torch.randn(p.shape[1:], generator=generator) * .01
            else:
                generator = torch.Generator().manual_seed(tick * 1000 + index)
                p.grad = torch.randn(p.shape, generator=generator).to(p.device) * .01
        opt.step()

    def test_growth_prune_and_directory_resume_with_different_slot_order(self):
        from train import _resync_opt
        model = self.recurrent_model()
        pool, opt = model.pool, self.optimizer(model)
        self.select(pool, (0, 1))
        self.deterministic_step(model, opt, 1)
        old_gate_m = opt.state[pool.gate]["exp_avg"].clone()
        old_router = pool._sites[0].router.weight
        old_router_m = opt.state[old_router]["exp_avg"].clone()
        pool.add_experts(1, seed_from=0, step=2)
        _resync_opt(opt, model, SimpleNamespace(lr=.001, wd=.1))
        self.assertTrue(torch.equal(opt.state[pool.gate]["exp_avg"][:-1], old_gate_m))
        self.assertEqual(float(opt.state[pool.gate]["exp_avg"][-1]), 0.)
        self.assertTrue(torch.equal(opt.state[pool._sites[0].router.weight]["exp_avg"][:-1], old_router_m))
        self.select(pool, (5, 1))
        self.deterministic_step(model, opt, 2)
        self.assertEqual(opt._gefen.state[opt._views[0][0]]["step"], 1)
        self.assertEqual(opt._gefen.state[opt._views[1][0]]["step"], 2)
        # Remove old nonresident positions; permanent expert UIDs stay 1 and 5.
        pool.segments = 100
        pool.last_seen.zero_()
        gone = pool.prune(step=100, survival=1)
        self.assertEqual(gone, 4)
        self.assertEqual(pool.uid.tolist(), [1, 5])
        _resync_opt(opt, model, SimpleNamespace(lr=.001, wd=.1))
        store.save(model, self.tmp.name, opt=opt, cfg=asdict(model.cfg), step=2)
        resumed, _ = load_any(self.tmp.name, torch.device(self.device), read_only=False)
        resumed_opt = self.optimizer(resumed)
        store._load_optim(resumed_opt, resumed, self.tmp.name)
        # Same experts, opposite physical slots; state follows UID.
        self.select(resumed.pool, tuple(reversed(pool.slots)))
        self.deterministic_step(model, opt, 3)
        self.deterministic_step(resumed, resumed_opt, 3)
        for name, p in model.named_parameters():
            if not opt.uses_gefen_parameter(p):
                other = dict(resumed.named_parameters())[name]
                self.assertTrue(torch.equal(p, other), name)
                assert_tree(self, opt.state[p], resumed_opt.state[other])
        for s, position in enumerate(pool.slots):
            other_slot = resumed.pool.slots.index(position)
            for j, name in enumerate(NAMES):
                self.assertTrue(torch.equal(getattr(pool, name)[s], getattr(resumed.pool, name)[other_slot]))
                assert_tree(self, opt._gefen.state[opt._views[s][j]],
                            resumed_opt._gefen.state[resumed_opt._views[other_slot][j]])
        assert_tree(self, opt._global(), resumed_opt._global())
        self.assertEqual([g["lr"] for g in opt.param_groups], [g["lr"] for g in resumed_opt.param_groups])
        wrong = torch.optim.AdamW(resumed.parameters())
        with self.assertRaisesRegex(ValueError, "select the Gefen"):
            store._load_optim(wrong, resumed, self.tmp.name)

    def test_recurrent_forward_backward_and_cache_refresh(self):
        from minagi.precision import compute_dtype, set_compute_dtype
        original = compute_dtype()
        self.addCleanup(set_compute_dtype, original)
        set_compute_dtype("fp32")
        model = self.recurrent_model().train()
        opt = self.optimizer(model)
        for tick in range(3):
            torch.manual_seed(10 + tick)
            ids = torch.randint(0, 265, (1, 17), device=self.device)
            _, loss = model(ids[:, :-1], ids[:, 1:])
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            cached = model.pool.inference_weights(torch.bfloat16)
            opt.step()
            self.assertIsNone(model.pool._infer)
            refreshed = model.pool.inference_weights(torch.bfloat16)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(cached, refreshed)))
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            opt.zero_grad(set_to_none=True)

    def test_legacy_adam_adoption_restores_auxiliary_but_starts_fresh_gefen(self):
        model, _ = self.pool_model()
        pool = model.pool
        self.select(pool, (0, 1))
        adam = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .95), fused=self.device == "cuda")
        pool.attach_optimiser(adam)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        adam.step()
        pool.flush()
        store._save_optim(adam, model, self.tmp.name)
        opt = self.optimizer(model)
        store._load_optim(opt, model, self.tmp.name)
        self.assertNotIn(pool.w1, opt.state)
        self.assertIn("exp_avg", opt.state[model.aux])
        self.assertTrue(torch.equal(opt.state[model.aux]["exp_avg"],
                                    adam.state[model.aux]["exp_avg"].bfloat16().float()))
        opt.step()
        for row in opt._views:
            for p in row:
                self.assertEqual(opt._gefen.state[p]["step"], 1)

    def test_evicted_history_has_recoverable_codebook_before_first_checkpoint(self):
        model, _ = self.pool_model()
        pool, opt = model.pool, self.optimizer(model)
        self.select(pool, (0, 1))
        for name in NAMES:
            getattr(pool, name).grad = torch.randn_like(getattr(pool, name))
        opt.step()
        self.assertTrue(opt._codebook_path.exists())
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "optim.npz")))
        self.select(pool, (2, 3))  # forces trained history to disk
        resumed = torch.nn.Module()
        resumed.pool = PagedPool(pool.path, 16, 32, 5, resident=2, ram_capacity=1).to(self.device)
        resumed.aux = torch.nn.Parameter(model.aux.detach().clone())
        resumed_opt = self.optimizer(resumed)
        self.assertTrue(torch.equal(opt._gefen._gefen_codebook, resumed_opt._gefen._gefen_codebook))
        for candidate, candidate_opt in ((model, opt), (resumed, resumed_opt)):
            self.select(candidate.pool, (0, 1))
            for name in NAMES:
                slab = getattr(candidate.pool, name)
                slab.grad = torch.full_like(slab, .02)
            candidate_opt.step()
        for name in NAMES:
            self.assertTrue(torch.equal(getattr(pool, name), getattr(resumed.pool, name)))
        for row, resumed_row in zip(opt._views, resumed_opt._views):
            for p, q in zip(row, resumed_row):
                assert_tree(self, opt._gefen.state[p], resumed_opt._gefen.state[q])

    def test_corrupt_geometry_and_partition_rejected_before_update(self):
        model, _ = self.pool_model()
        pool, opt = model.pool, self.optimizer(model)
        self.select(pool, (0, 1))
        for name in NAMES:
            getattr(pool, name).grad = torch.randn_like(getattr(pool, name))
        opt.step()
        pool.flush()
        self.select(pool, (2, 3))
        entry = dict(pool.tiers.fetch(0))
        payload = optim_codec.decode(entry, EXPERT_PREFIX)
        original = copy.deepcopy(payload)
        for corruption in ("uid", "shape", "period", "indices", "codebook"):
            bad = copy.deepcopy(original)
            if corruption == "uid":
                bad["uid"] = 999
            elif corruption == "shape":
                bad["shapes"][0] = (1, 1)
            elif corruption == "period":
                bad["states"]["w1"]["automatic_period"] = 0
            elif corruption == "indices":
                bad["states"]["w1"]["m_codebook"] = torch.ones(1, dtype=torch.float32)
            else:
                bad["codebook_id"] = "wrong"
            arrays = optim_codec.encode(bad, EXPERT_PREFIX)
            changed = {k: v for k, v in entry.items() if not k.startswith(EXPERT_PREFIX)}
            changed.update({k: torch.from_numpy(v) for k, v in arrays.items()})
            pool.tiers.put(0, changed)
            self.select(pool, (0, 3))
            before = pool.w1.detach().clone()
            with self.assertRaisesRegex(ValueError, "Gefen"):
                opt.step()
            self.assertTrue(torch.equal(pool.w1, before))
            self.select(pool, (2, 3))
        # Restore valid state so failure did not make it unrecoverable.
        arrays = optim_codec.encode(original, EXPERT_PREFIX)
        entry = {k: v for k, v in entry.items() if not k.startswith(EXPERT_PREFIX)}
        entry.update({k: torch.from_numpy(v) for k, v in arrays.items()})
        pool.tiers.put(0, entry)
        self.select(pool, (0, 3))
        opt.step()

    def test_eviction_disk_reload_matches_uninterrupted_stock_gefen(self):
        from gefen import Gefen
        model, originals = self.pool_model()
        pool, opt = model.pool, self.optimizer(model)
        reference = [{name: torch.nn.Parameter(entry[name].to(self.device).clone()) for name in NAMES}
                     for entry in originals]
        ref_opt = Gefen([p for entry in reference for p in entry.values()], lr=.001,
                        betas=(.9, .95), weight_decay=.1, fused=self.device == "cuda")
        aux = torch.nn.Parameter(model.aux.detach().clone())
        gate = torch.nn.Parameter(pool.gate.detach().clone())
        ref_adam = torch.optim.AdamW([aux, gate], lr=.001, betas=(.9, .95), weight_decay=.1, fused=self.device == "cuda")
        for tick, ids in enumerate(((0, 1), (0, 1), (2, 3), (4, 3), (0, 1), (2, 1))):
            if tick % 2 == 0:
                self.select(pool, ids)
            opt.zero_grad(set_to_none=True)
            ref_opt.zero_grad(set_to_none=True)
            for j, name in enumerate(NAMES):
                slab = getattr(pool, name)
                if tick == 3 and name == "w2":
                    continue   # no gradient means no projection update/counter
                slab.grad = torch.empty_like(slab)
                for slot, uid in enumerate(ids):
                    g = torch.Generator().manual_seed(1000 + tick * 100 + uid * 3 + j)
                    gradient = torch.randn(slab.shape[1:], generator=g) * .01
                    if tick == 4:
                        gradient.zero_()  # zero is an update, unlike grad=None
                    slab.grad[slot].copy_(gradient)
                    reference[uid][name].grad = gradient.to(self.device).clone()
            model.aux.grad = aux.grad = torch.full_like(aux, .03)
            pool.gate.grad = gate.grad = torch.linspace(-.02, .03, pool._n, device=self.device)
            opt.step((lambda: self.select(pool, ids)) if tick % 2 else None)
            ref_opt.step()
            ref_adam.step()
            self.assertEqual(opt._gefen._gefen_global_step, tick + 1)
            self.assertTrue(torch.equal(opt._gefen._gefen_codebook, ref_opt._gefen_codebook))
            self.assertTrue(torch.equal(model.aux, aux))
            self.assertTrue(torch.equal(pool.gate, gate))
            for s, uid in enumerate(ids):
                for j, name in enumerate(NAMES):
                    self.assertTrue(torch.equal(getattr(pool, name)[s], reference[uid][name]))
                    assert_tree(self, opt._gefen.state[opt._views[s][j]], ref_opt.state[reference[uid][name]])
            self.assertFalse(any(p in opt.state for p in (pool.w1, pool.w3, pool.w2)))
        pool.flush()
        self.assertGreater(pool.tiers.writebacks, pool._n)
        for uid in range(pool._n):
            entry = pool.tiers._from_disk(uid)
            payload = optim_codec.decode(entry, EXPERT_PREFIX)
            self.assertEqual(payload["uid"], uid)
            for name in NAMES:
                self.assertTrue(torch.equal(entry[name], reference[uid][name].cpu()))
                assert_tree(self, payload["states"][name], ref_opt.state[reference[uid][name]])

    def test_empty_slots_skipped_and_history_requires_matching_codebook(self):
        model, _ = self.pool_model()
        pool, opt = model.pool, self.optimizer(model)
        for name in NAMES:
            getattr(pool, name).grad = torch.ones_like(getattr(pool, name))
        opt.step()
        self.assertEqual(opt._gefen._gefen_global_step, 0)
        self.assertFalse(any("step" in st for st in opt._gefen.state.values()))
        self.select(pool, (0, -1))
        opt.step()
        self.assertEqual(opt._gefen._gefen_global_step, 1)
        self.assertNotIn("step", opt._gefen.state[opt._views[1][0]])
        self.select(pool, (1, -1))
        self.select(pool, (0, -1))
        saved = opt._gefen._gefen_codebook
        opt._gefen._gefen_codebook = None
        with self.assertRaisesRegex(ValueError, "matching optimizer codebook"):
            opt.step()
        opt._gefen._gefen_codebook = saved
        opt.step()
        pool.flush()
        adam = torch.optim.AdamW(model.parameters())
        pool.attach_optimiser(adam)
        with self.assertRaisesRegex(ValueError, "Gefen expert history"):
            adam.step()

    def test_state_dict_continuation_preserves_uint8_and_uid_mapping(self):
        model, _ = self.pool_model()
        pool, opt = model.pool, self.optimizer(model)
        self.select(pool, (0, 1))
        for name in NAMES:
            getattr(pool, name).grad = torch.randn_like(getattr(pool, name))
        opt.step()
        saved = copy.deepcopy(opt.state_dict())
        weights = [getattr(pool, name).detach().clone() for name in NAMES]
        opt.step()
        expected = [getattr(pool, name).detach().clone() for name in NAMES]
        expected_states = copy.deepcopy(opt.state_dict())
        for name, weight in zip(NAMES, weights):
            getattr(pool, name).data.copy_(weight)
        opt.load_state_dict(saved)
        opt.step()
        for name, value in zip(NAMES, expected):
            self.assertTrue(torch.equal(getattr(pool, name), value))
        assert_tree(self, opt.state_dict(), expected_states)


@unittest.skipUnless(importlib.util.find_spec("gefen") and torch.cuda.is_available(), "optional Gefen and CUDA required")
class CudaGefenPagingTests(GefenPagingTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
