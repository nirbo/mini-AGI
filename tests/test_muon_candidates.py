"""Research adapter qualification against independent, unpaged upstream optimizers."""

import ast
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from minagi import optim_codec
from minagi.optimizers import NAMES
from minagi.paged import PagedPool
from replication.muon_candidates import (ROOT, ResearchPagedMatrix, VARIANTS, matrix_optimizer,
                                          classic_ns5, upstream_plus, source_path)


@unittest.skipUnless(all((ROOT/'runs/third_party'/n).exists() for n in ('dion', 'MuonPlus')),
                     'run .venv/bin/python -m replication.muon_candidates to fetch pinned research sources')
class MuonCandidateTests(unittest.TestCase):
    device = 'cpu'

    def setUp(self):
        torch.set_num_threads(4)
        limit = torch._dynamo.config.recompile_limit
        torch._dynamo.config.recompile_limit = 64
        self.addCleanup(setattr, torch._dynamo.config, 'recompile_limit', limit)
        torch.manual_seed(72)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def model(self, suffix='a'):
        model = torch.nn.Module()
        model.pool = PagedPool(str(Path(self.tmp.name) / suffix), 16, 32, 5,
                               resident=2, ram_capacity=5, read_only=True).to(self.device)
        model.aux = torch.nn.Parameter(torch.randn(3, device=self.device))
        for uid in range(5):
            model.pool.tiers.put(uid, {n: torch.randn(*getattr(model.pool, n).shape[1:]) * .1 for n in NAMES})
        return model

    def optimizer(self, model, variant):
        opt = ResearchPagedMatrix(model.parameters(), model.pool, variant, .001,
                                   weight_decay=.1, betas=(.9, .95), fused=self.device == 'cuda')
        model.pool.attach_optimiser(opt)
        return opt

    def select(self, pool, plan):
        pool._rearrange(plan, {e: s for s, e in enumerate(pool.slots) if e >= 0})

    def gradients(self, model, tick, missing=None):
        for j, n in enumerate(NAMES):
            slab = getattr(model.pool, n)
            if n == missing:
                slab.grad = None
                continue
            slab.grad = torch.empty_like(slab)
            for slot, position in enumerate(model.pool.slots):
                generator = torch.Generator().manual_seed(1000*tick + 10*max(position, 0) + j)
                slab.grad[slot] = (torch.randn(slab.shape[1:], generator=generator) * .01).to(self.device)
            if tick == 3:
                slab.grad.zero_()
        model.aux.grad = torch.full_like(model.aux, tick * .001)

    def test_paging_swap_empty_zero_none_against_unpaged_upstream(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant)
                opt = self.optimizer(model, variant)
                reference = {}
                for uid in range(5):
                    ps = [torch.nn.Parameter(model.pool.tiers.fetch(uid)[n].to(self.device).clone()) for n in NAMES]
                    reference[uid] = ps
                engine = matrix_optimizer(variant, [p for ps in reference.values() for p in ps], .001)
                aux = torch.nn.Parameter(model.aux.detach().clone())
                aux_opt = torch.optim.AdamW([aux], lr=.001, betas=(.9, .95), weight_decay=.1,
                                           fused=self.device == 'cuda')
                # Retained experts keep their physical slot in the real pager.
                # Swap A/B only after evicting both, then readmit in reverse.
                plans = ((0, 1), (2, 3), (1, 0), (3, 4), (1, 2), (4, -1), (0, 3))
                counts = {}
                for tick, plan in enumerate(plans, 1):
                    self.select(model.pool, plan)
                    missing = 'w3' if tick == 4 else None
                    self.gradients(model, tick, missing)
                    engine.zero_grad(set_to_none=True)
                    for slot, uid in enumerate(plan):
                        if uid < 0:
                            continue
                        ps = reference[uid]
                        for n, p in zip(NAMES, ps):
                            g = getattr(model.pool, n).grad
                            p.grad = g[slot].clone() if g is not None else None
                            if g is not None:
                                counts[(uid, n)] = counts.get((uid, n), 0) + 1
                    engine.step()
                    aux.grad = model.aux.grad.clone()
                    aux_opt.step()
                    cached = model.pool.inference_weights(torch.bfloat16)
                    opt.step()
                    self.assertIsNone(model.pool._infer)
                    self.assertTrue(torch.equal(model.aux, aux))
                    for slot, uid in enumerate(plan):
                        if uid < 0:
                            continue
                        for n, p in zip(NAMES, reference[uid]):
                            torch.testing.assert_close(getattr(model.pool, n)[slot], p, rtol=0, atol=0)
                        for p, q in zip(opt._views[slot], reference[uid]):
                            for key, value in engine.state[q].items():
                                if torch.is_tensor(value):
                                    torch.testing.assert_close(opt._matrix.state[p][key], value, rtol=0, atol=0)
                    self.assertEqual(dict(opt._counts), counts)
                self.assertEqual(model.pool.tiers.writebacks, 0)

    def test_lossless_snapshot_resume_into_different_slots(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant)
                opt = self.optimizer(model, variant)
                for tick, plan in enumerate(((0, 1), (2, 3), (1, 4)), 1):
                    self.select(model.pool, plan)
                    self.gradients(model, tick)
                    opt.step()
                # Exercise the real lossless NPZ codec, not pickle.
                saved = optim_codec.decode(optim_codec.encode(opt.state_dict()))
                clone = self.model(variant + '-resume')
                clone.aux.data.copy_(model.aux)
                for uid in range(5):
                    if uid in model.pool.slots:
                        slot = model.pool.slots.index(uid)
                        weights = {n: getattr(model.pool, n)[slot].detach().cpu().clone() for n in NAMES}
                    else:
                        weights = {n: model.pool.tiers.fetch(uid)[n].clone() for n in NAMES}
                    clone.pool.tiers.put(uid, weights)
                resumed = self.optimizer(clone, variant)
                resumed.load_state_dict(saved)
                for tick, plan in enumerate(((0, 4), (2, 1)), 4):
                    self.select(model.pool, plan)
                    self.select(clone.pool, tuple(reversed(plan)))
                    self.gradients(model, tick)
                    self.gradients(clone, tick)
                    opt.step()
                    resumed.step()
                    torch.testing.assert_close(model.aux, clone.aux, rtol=0, atol=0)
                    for s, uid in enumerate(plan):
                        other = clone.pool.slots.index(uid)
                        for n in NAMES:
                            torch.testing.assert_close(getattr(model.pool, n)[s], getattr(clone.pool, n)[other], rtol=0, atol=0)
                    self.assertEqual(dict(opt._counts), dict(resumed._counts))

    @torch.no_grad()
    def test_classic_ns5_and_author_normalization(self):
        source = source_path('MuonPlus') / 'utils/optim/muon_plus.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'apply_post_polar_norm')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        author = upstream_plus()
        # Generalize only the author's 2D transpose check to trailing axes.
        # This lets both references use the SAME batch geometry and compiler,
        # instead of conflating BF16 GEMM selection with a formula difference.
        ns = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'zeropower_via_newtonschulz5')
        ns = copy.deepcopy(ns)
        ns.decorator_list = []
        class BatchAxes(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if (isinstance(node.func, ast.Attribute) and node.func.attr == 'size'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'G'
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)):
                    node.args[0] = ast.Constant(-2 if node.args[0].value == 0 else -1)
                return node
        ns = ast.fix_missing_locations(BatchAxes().visit(ns))
        exec(compile(ast.Module(body=[ns], type_ignores=[]), str(source), 'exec'), namespace)
        reference = torch.compile(namespace['zeropower_via_newtonschulz5'], fullgraph=True, dynamic=False)
        for shape in ((3, 16, 32), (3, 32, 16)):
            g = torch.randn(shape, device=self.device)
            actual = classic_ns5(g)
            torch.testing.assert_close(actual, reference(g, 5), rtol=0, atol=0)
            u = torch.randn_like(g).bfloat16()
            torch.testing.assert_close(author.apply_post_polar_norm(u, 'col_row'),
                                       namespace['apply_post_polar_norm'](u, 'col_row'), rtol=0, atol=0)

    def test_writable_store_and_wrong_variant_rejected(self):
        model = self.model()
        opt = self.optimizer(model, 'muon')
        other = self.optimizer(model, 'normuon')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            other.load_state_dict(opt.state_dict())
        model.pool.tiers.read_only = False
        with self.assertRaisesRegex(ValueError, 'read-only'):
            self.optimizer(model, 'muon')

    def test_closure_can_page_and_all_missing_gradients_pause_history(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant)
                opt = self.optimizer(model, variant)
                ps = [torch.nn.Parameter(model.pool.tiers.fetch(0)[n].to(self.device).clone()) for n in NAMES]
                reference = matrix_optimizer(variant, ps, .001)
                self.select(model.pool, (0, -1))
                self.gradients(model, 1)
                for n,p in zip(NAMES,ps):
                    p.grad = getattr(model.pool,n).grad[0].clone()
                reference.step()
                opt.step()
                self.select(model.pool, (1, -1))
                def closure():
                    self.select(model.pool, (0, -1))
                    self.gradients(model, 2)
                    for n,p in zip(NAMES,ps):
                        p.grad = getattr(model.pool,n).grad[0].clone()
                    reference.step()
                    return torch.tensor(7.)
                self.assertEqual(float(opt.step(closure)), 7.)
                for n,p in zip(NAMES,ps):
                    torch.testing.assert_close(getattr(model.pool,n)[0],p,rtol=0,atol=0)
                counts = dict(opt._counts)
                before = {n:getattr(model.pool,n).detach().clone() for n in NAMES}
                states = {p:copy.deepcopy(opt._matrix.state[p]) for p in opt._views[0]}
                opt.zero_grad(set_to_none=True)
                opt.step()
                self.assertEqual(dict(opt._counts), counts)
                for n in NAMES:
                    torch.testing.assert_close(getattr(model.pool,n),before[n],rtol=0,atol=0)
                for p,state in states.items():
                    for key,value in state.items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(opt._matrix.state[p][key],value,rtol=0,atol=0)

    def test_optimizer_preserves_model_routing_random_stream(self):
        # begin_text samples its routing seed from Torch's CPU RNG. Optimizer
        # initialization/compilation must not change the model's random stream.
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                p = torch.nn.Parameter(torch.randn(16,32,device=self.device))
                p.grad = torch.ones_like(p)*.001
                cpu_before = torch.get_rng_state().clone()
                gpu_before = torch.cuda.get_rng_state().clone() if self.device == 'cuda' else None
                opt = matrix_optimizer(variant,[p],.001)
                opt.step()
                self.assertTrue(torch.equal(torch.get_rng_state(),cpu_before))
                if gpu_before is not None:
                    self.assertTrue(torch.equal(torch.cuda.get_rng_state(),gpu_before))


if __name__ == '__main__':
    # CPU by default. CUDA qualification explicitly opts in under NVML wrapper.
    import os
    torch._dynamo.config.recompile_limit = 64
    if os.environ.get('MINAGI_MUON_TEST_CUDA') == '1':
        MuonCandidateTests.device = 'cuda'
    unittest.main()
